"""Build the tidy per-room and per-block tables for one block + time window.

The dataset authors already did most of the heavy lifting (outlier removal,
resampling, weather-variable selection, time dummies) in their own
processed_data/data-room{BLOCK}-10T.csv files. This script:
  1. Filters that table to the chosen block/window/rooms.
  2. Collapses multiple HVAC units per room into one row per (room, timestamp).
  3. Merges in presence (motion) and CO2 data, which the authors left
     unresampled/unmerged, resampling them onto the same 10-min grid.
  4. Writes a tidy per-room parquet table and a per-block consumption table.

Parameterized by block/window/rooms so re-running for more data later
(Step 1's "scale up") doesn't require rewriting this pipeline.
"""
import json

import pandas as pd

from config import (
    BLOCK,
    FOCUS_ROOMS,
    PROCESSED_SRC_DIR,
    RAW_DIR,
    ROOMS_TABLE_PATH,
    CONSUMPTION_TABLE_PATH,
    STRUCTURE_PATH,
    WINDOW_START,
    WINDOW_END,
)

HVAC_AGG = {
    "dif_cons": "first",
    "cons_total": "first",
    "V2": "first",  # room temperature -- identical across duplicate HVAC rows
    "V12": "mean",  # setpoint, averaged across units in the room
    "V4": "max",  # HVAC state -- 1 if ANY unit in the room is on
    "V26": "mean",
    "V5_0": "mean",
    "V5_1": "mean",
    "V5_2": "mean",
    "tmed": "first",
    "hrmed": "first",
    "radmed": "first",
    "vvmed": "first",
    "dvmed": "first",
    "prec": "first",
    "dewpt": "first",
    "dpv": "first",
    "Hour_1": "first",
    "Hour_2": "first",
    "Hour_3": "first",
    "Season_1": "first",
    "Season_2": "first",
    "Season_3": "first",
    "Season_4": "first",
}


def load_room_table(block: str, start: str, end: str, rooms: list[str]) -> pd.DataFrame:
    path = PROCESSED_SRC_DIR / f"data-room{block}-10T.csv"
    df = pd.read_csv(path, sep=";", parse_dates=["Date"])
    df["room"] = df["room"].astype(str)
    df = df[df["room"].isin(rooms)]
    df = df[(df["Date"] >= start) & (df["Date"] <= end)]
    collapsed = df.groupby(["Date", "room"], as_index=False).agg(HVAC_AGG)
    return collapsed


def load_presence_or_co2(kind: str, sensor_ids: dict[str, int], start: str, end: str, value_col: str, agg: str) -> pd.DataFrame:
    """kind: 'presence' or 'CO2'. sensor_ids: {room: device_id}."""
    path = RAW_DIR / f"data-{kind}.csv"
    sep = "," if kind == "presence" else ";"
    id_to_room = {v: k for k, v in sensor_ids.items()}
    wanted_ids = set(sensor_ids.values())

    chunks = []
    for chunk in pd.read_csv(path, sep=sep, chunksize=2_000_000):
        sub = chunk[chunk["IDdevice"].isin(wanted_ids)].copy()
        if len(sub):
            chunks.append(sub)
    if not chunks:
        raise ValueError(f"No {kind} rows found for requested sensors")
    raw = pd.concat(chunks, ignore_index=True)
    raw["Date"] = pd.to_datetime(raw["Date"], utc=True, format="mixed")
    raw = raw[(raw["Date"] >= start) & (raw["Date"] <= end)]
    raw["room"] = raw["IDdevice"].map(id_to_room)

    out = []
    for room, g in raw.groupby("room"):
        g = g.set_index("Date").sort_index()
        resampled = g[value_col].resample("10min").agg(agg)
        out.append(pd.DataFrame({"room": room, "Date": resampled.index, value_col: resampled.values}))
    result = pd.concat(out, ignore_index=True)
    return result


def build_rooms_table() -> pd.DataFrame:
    structure = json.loads(STRUCTURE_PATH.read_text())
    presence_ids = {r: v["presence_sensors"][0] for r, v in structure["rooms"].items() if v["presence_sensors"] and r in FOCUS_ROOMS}
    co2_ids = {r: v["co2_sensors"][0] for r, v in structure["rooms"].items() if v["co2_sensors"] and r in FOCUS_ROOMS}

    rooms = load_room_table(BLOCK, WINDOW_START, WINDOW_END, FOCUS_ROOMS)

    presence = load_presence_or_co2("presence", presence_ids, WINDOW_START, WINDOW_END, "V2", "max")
    presence = presence.rename(columns={"V2": "presence"})
    co2 = load_presence_or_co2("CO2", co2_ids, WINDOW_START, WINDOW_END, "V17", "mean")
    co2 = co2.rename(columns={"V17": "co2_ppm"})

    merged = rooms.merge(presence, on=["Date", "room"], how="left")
    merged = merged.merge(co2, on=["Date", "room"], how="left")
    merged["presence"] = merged["presence"].fillna(0).astype(int)
    merged["co2_ppm"] = merged.groupby("room")["co2_ppm"].transform(lambda s: s.ffill().bfill())

    return merged


def build_consumption_table() -> pd.DataFrame:
    path = PROCESSED_SRC_DIR / f"cons{BLOCK}-10T.csv"
    df = pd.read_csv(path, sep=";", parse_dates=["Date"])
    df = df[(df["Date"] >= WINDOW_START) & (df["Date"] <= WINDOW_END)]
    return df


if __name__ == "__main__":
    rooms_df = build_rooms_table()
    rooms_df.to_parquet(ROOMS_TABLE_PATH, index=False)
    print(f"rooms table: {rooms_df.shape} -> {ROOMS_TABLE_PATH}")
    print(rooms_df.isna().sum()[rooms_df.isna().sum() > 0])

    cons_df = build_consumption_table()
    cons_df.to_parquet(CONSUMPTION_TABLE_PATH, index=False)
    print(f"consumption table: {cons_df.shape} -> {CONSUMPTION_TABLE_PATH}")
