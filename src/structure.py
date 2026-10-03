"""Build the structural model (block -> room -> sensors/HVAC) for one block.

This is the digital twin's skeleton: it maps device IDs to rooms so the app
can label charts and populate room selectors, independent of any time window.
"""
import json
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "Data_Nature" / "Data_Nature" / "raw_data"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "structure.json"


def build_structure(block: str) -> dict:
    sensors = pd.read_csv(RAW_DIR / "relations-sensor.csv", sep=";")
    hvac = pd.read_csv(RAW_DIR / "relations-hvac.csv", sep=";")
    presence = pd.read_csv(RAW_DIR / "relations-presence.csv", sep=";")
    co2 = pd.read_csv(RAW_DIR / "relations-CO2.csv", sep=";")

    rooms = {}

    def add(df, kind):
        sub = df[df["block"] == block]
        for _, row in sub.iterrows():
            room = str(row["room"])
            rooms.setdefault(room, {"temp_sensors": [], "hvac_units": [], "presence_sensors": [], "co2_sensors": []})
            rooms[room][kind].append(int(row["ID"]))

    add(sensors, "temp_sensors")
    add(hvac, "hvac_units")
    add(presence, "presence_sensors")
    add(co2, "co2_sensors")

    structure = {
        "block": block,
        "rooms": rooms,
        "room_count": len(rooms),
    }
    return structure


if __name__ == "__main__":
    import sys

    block = sys.argv[1] if len(sys.argv) > 1 else "B"
    structure = build_structure(block)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(structure, indent=2))
    print(f"Block {block}: {structure['room_count']} rooms -> {OUT_PATH}")
    rooms_with_presence = [r for r, v in structure["rooms"].items() if v["presence_sensors"]]
    rooms_with_co2 = [r for r, v in structure["rooms"].items() if v["co2_sensors"]]
    print(f"Rooms with presence sensors: {rooms_with_presence}")
    print(f"Rooms with CO2 sensors: {rooms_with_co2}")
