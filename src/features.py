"""Feature engineering for Model A (indoor temp) and Model B (energy).

Season_1..4 dummies are dropped: the chosen 8-week window falls entirely in
winter (Season_1), so they carry zero information and would just confuse
feature-importance plots in the demo.
"""
import pandas as pd

TEMP_LAGS = [1, 2, 3, 6]  # 10-min steps -> 10/20/30/60 min of history

TEMP_FEATURE_COLS = [
    "V12", "V4", "V26", "V5_0", "V5_1", "V5_2",  # HVAC state/setpoint/mode/type
    "tmed", "hrmed", "radmed", "vvmed", "dvmed", "prec", "dewpt", "dpv",  # weather
    "presence", "co2_ppm",  # occupancy proxies
    "Hour_1", "Hour_2", "Hour_3", "hour", "dayofweek", "is_weekend",  # time
] + [f"V2_lag{l}" for l in TEMP_LAGS]

TEMP_TARGET = "V2_next"


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["Date"].dt.hour
    df["dayofweek"] = df["Date"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    return df


def make_temp_table(rooms_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (room, timestamp): lag features of V2 -> next-step V2."""
    df = add_time_features(rooms_df).sort_values(["room", "Date"])
    g = df.groupby("room")["V2"]
    for lag in TEMP_LAGS:
        df[f"V2_lag{lag}"] = g.shift(lag)
    df[TEMP_TARGET] = g.shift(-1)
    df = df.dropna(subset=[f"V2_lag{l}" for l in TEMP_LAGS] + [TEMP_TARGET])
    return df


ENERGY_FEATURE_COLS = [
    "V2", "V4", "V12", "V26", "V5_0", "V5_1", "V5_2",  # block avg temp + hvac-agg
    "tmed", "hrmed", "radmed", "vvmed", "dvmed", "prec", "dewpt", "dpv",
    "Hour_1", "Hour_2", "Hour_3", "hour", "dayofweek", "is_weekend",
]
ENERGY_TARGET = "dif_cons_real"


def make_energy_table(consumo_df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_features(consumo_df).sort_values("Date")
    df = df.dropna(subset=ENERGY_FEATURE_COLS + [ENERGY_TARGET])
    return df
