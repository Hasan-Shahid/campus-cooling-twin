"""Color constants, per the dataviz skill: fixed-order categorical hues,
one-hue sequential/diverging ramps, and a reserved status palette that's
never reused for series identity."""

CATEGORICAL = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]

DIVERGING_BLUE_RED = ["#104281", "#5598e7", "#cde2fb", "#f0efec", "#f6c2c1", "#e34948", "#9c1e1d"]

STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

# Dark-mode tokens (app now runs on a dark theme -- see .streamlit/config.toml).
CATEGORICAL_DARK = [
    "#3987e5",  # 1 blue
    "#d95926",  # 2 orange
    "#199e70",  # 3 aqua
    "#c98500",  # 4 yellow
    "#d55181",  # 5 magenta
    "#008300",  # 6 green
    "#9085e9",  # 7 violet
    "#e66767",  # 8 red
]
SEQUENTIAL_BLUE_DARK = ["#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#5598e7", "#86b6ef"]
SURFACE_DARK = "#1a1a19"
PAGE_DARK = "#0d0d0d"
INK_PRIMARY_DARK = "#ffffff"
INK_SECONDARY_DARK = "#c3c2b7"
GRIDLINE_DARK = "#2c2c2a"

# "HUD" chrome tokens for the futuristic theme -- decorative (background,
# panel fills, gridlines, borders), never used to encode data. The actual
# data-bearing colors stay CATEGORICAL_DARK/STATUS/SEQUENTIAL_BLUE_DARK above,
# which are the validated, colorblind-checked set.
VOID_BG = "#05070a"
PANEL_BG = "#0d141f"
NEON_CYAN = "#3987e5"
NEON_CYAN_BRIGHT = "#5bb8ff"
BORDER_GLOW = "rgba(57,135,229,0.35)"
GRIDLINE_NEON = "rgba(57,135,229,0.18)"
AXIS_LINE_NEON = "rgba(57,135,229,0.4)"
INK_HUD = "#9db8d9"

ROOM_COLOR = {room: CATEGORICAL_DARK[i % len(CATEGORICAL_DARK)] for i, room in enumerate(["9", "10", "11", "12", "15", "16", "17", "18"])}

# Schematic room layout (NOT to scale -- the dataset has no floor-plan
# geometry; see plan's "Structural model" discussion). Two corridors of 4.
ROOM_GRID_POSITIONS = {
    "9": (0, 1), "10": (1, 1), "11": (2, 1), "12": (3, 1),
    "15": (0, 0), "16": (1, 0), "17": (2, 0), "18": (3, 0),
}
