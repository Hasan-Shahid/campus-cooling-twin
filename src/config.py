"""Shared config for the v1 digital twin scope.

Chosen per the plan's Step 1: Block B (has presence+CO2 sensors, unlike C;
smaller/cleaner than A's 281MB room-level file), an 8-week winter window
(good heating-season HVAC activity, verified zero gaps/NaNs in consumption
and room temperature for the focus rooms).
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_NATURE = PROJECT_ROOT / "Data_Nature" / "Data_Nature"
RAW_DIR = DATA_NATURE / "raw_data"
PROCESSED_SRC_DIR = DATA_NATURE / "processed_data"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

BLOCK = "B"
WINDOW_START = "2021-02-01"
WINDOW_END = "2021-03-28"

# The 8 Block B rooms with temperature + HVAC + presence + CO2 sensors all
# present -- the only rooms where occupancy-aware anomaly detection is
# possible. Other Block B rooms exist (24 total) but lack presence/CO2.
FOCUS_ROOMS = ["9", "10", "11", "12", "15", "16", "17", "18"]

STRUCTURE_PATH = PROCESSED_DIR / "structure.json"
ROOMS_TABLE_PATH = PROCESSED_DIR / f"rooms_{BLOCK}_{WINDOW_START}_{WINDOW_END}.parquet"
CONSUMPTION_TABLE_PATH = PROCESSED_DIR / f"consumption_{BLOCK}_{WINDOW_START}_{WINDOW_END}.parquet"
