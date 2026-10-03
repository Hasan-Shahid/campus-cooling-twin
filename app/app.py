"""Digital twin for PLEIAData Block B (Pleiades building, Univ. of Murcia).

Feature set modeled on a reference campus-cooling-twin demo (replay timeline,
KPI/cost-carbon tiles, expected-vs-actual overlays, a diagnosis panel, model
evidence with ASHRAE Guideline 14 metrics, a sensors/real-data explainer) --
adapted to this project's REAL dataset and the models actually built in
src/models.py. Unlike that reference, this data is not synthetic and there is
no injected-fault answer key.
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from config import FOCUS_ROOMS, PROCESSED_DIR, ROOMS_TABLE_PATH, WINDOW_START, WINDOW_END, STRUCTURE_PATH
from diagnosis import TYPES
from features import ENERGY_FEATURE_COLS, TEMP_FEATURE_COLS, TEMP_TARGET, ENERGY_TARGET
from palette import (
    AXIS_LINE_NEON,
    BORDER_GLOW,
    CATEGORICAL_DARK as CATEGORICAL,
    GRIDLINE_NEON as GRIDLINE,
    INK_HUD as INK_SECONDARY,
    INK_PRIMARY_DARK,
    NEON_CYAN,
    NEON_CYAN_BRIGHT,
    PANEL_BG as SURFACE_DARK,
    ROOM_COLOR,
    ROOM_GRID_POSITIONS,
    SEQUENTIAL_BLUE_DARK as SEQUENTIAL_BLUE,
    STATUS,
    VOID_BG,
)

MODELS_DIR = PROCESSED_DIR / "models"

st.set_page_config(page_title="Pleiades Digital Twin -- Block B", page_icon="🛰️", layout="wide")


# ================================================================ loaders ==
@st.cache_data
def load_structure():
    return json.loads(STRUCTURE_PATH.read_text())


@st.cache_data
def load_rooms():
    df = pd.read_parquet(ROOMS_TABLE_PATH)
    df["day"] = df["Date"].dt.date
    return df


@st.cache_data
def load_temp_anomalies():
    df = pd.read_parquet(MODELS_DIR / "temp_anomalies.parquet")
    df["day"] = df["Date"].dt.date
    return df


@st.cache_data
def load_energy_anomalies():
    df = pd.read_parquet(MODELS_DIR / "energy_anomalies.parquet")
    df["day"] = df["Date"].dt.date
    return df


@st.cache_data
def load_prophet_comparison():
    return pd.read_parquet(MODELS_DIR / "prophet_comparison.parquet")


@st.cache_data
def load_metrics():
    return json.loads((MODELS_DIR / "metrics.json").read_text())


@st.cache_resource
def load_temp_model():
    return joblib.load(MODELS_DIR / "temp_model.joblib")


@st.cache_resource
def load_energy_model():
    return joblib.load(MODELS_DIR / "energy_model.joblib")


structure = load_structure()
rooms_df = load_rooms()
temp_df = load_temp_anomalies()
energy_df = load_energy_anomalies()
metrics = load_metrics()
DAYS = sorted(rooms_df["day"].unique())
N_DAYS = len(DAYS)

# ================================================================ state ====
ss = st.session_state
ss.setdefault("day_idx", N_DAYS - 1)
ss.setdefault("playing", False)
ss.setdefault("cop", 4.0)
ss.setdefault("tariff", 0.15)
ss.setdefault("ef", 0.25)
ss.setdefault("jump_room", None)
ss.setdefault("jump_day", None)
ss.setdefault("_pending_day_idx", None)

# A widget's session_state value can't be reassigned after that widget has
# already been instantiated in the same run (Streamlit raises
# StreamlitWidgetAlreadyInstantiatedError). So the play loop and preset
# buttons below stash the target day in `_pending_day_idx` and rerun; this
# consumes it BEFORE the "day_idx"-keyed slider widget is created.
if ss._pending_day_idx is not None:
    ss.day_idx = ss._pending_day_idx
    ss._pending_day_idx = None


def cost(kwh):
    return kwh / ss.cop * ss.tariff


def co2_kg(kwh):
    return kwh / ss.cop * ss.ef


def day_of(idx):
    return DAYS[max(0, min(N_DAYS - 1, idx))]


# ================================================================ chrome ===
def inject_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;700;800&family=Rajdhani:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

        html, body, [class*="css"] {{ font-family: 'JetBrains Mono', ui-monospace, monospace; }}

        /* page backdrop: animated futuristic scene (.bgfx, injected by inject_background) behind transparent chrome */
        .stApp {{ background: {VOID_BG}; }}
        [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stMainBlockContainer"] {{ background: transparent; position: relative; z-index: 1; }}
        [data-testid="stHeader"] {{ background: rgba(5,7,10,0.0); border-bottom: 1px solid {BORDER_GLOW}; z-index: 2; }}
        [data-testid="stSidebar"] {{ z-index: 2; }}

        .bgfx {{ position: fixed; inset: 0; z-index: 0; overflow: hidden; pointer-events: none;
            --glowA: rgba(51,225,255,.18); --glowB: rgba(140,90,255,.20); --gridline: rgba(51,225,255,.38);
            --trace: rgba(51,225,255,.16); --scan: rgba(51,225,255,.07); --horizon: rgba(51,225,255,.8); --floor: #070a10;
            --neon: #33e1ff;
            background: radial-gradient(70% 50% at 20% 0%, var(--glowA), transparent 70%),
                        radial-gradient(60% 45% at 90% 15%, var(--glowB), transparent 70%), {VOID_BG}; }}
        .bgfx .floor {{ position: absolute; left: -50%; right: -50%; bottom: -10%; height: 55%;
            background: linear-gradient(180deg, transparent, var(--floor) 85%),
                        linear-gradient(var(--gridline) 1.5px, transparent 1.5px) 0 0/60px 60px,
                        linear-gradient(90deg, var(--gridline) 1.5px, transparent 1.5px) 0 0/60px 60px;
            transform: perspective(420px) rotateX(62deg); transform-origin: 50% 0; animation: bgfx-floor 4s linear infinite;
            -webkit-mask-image: linear-gradient(180deg, transparent 0%, #000 30%); mask-image: linear-gradient(180deg, transparent 0%, #000 30%); }}
        @keyframes bgfx-floor {{ to {{ background-position: 0 0, 0 60px, 0 0; }} }}
        .bgfx .horizon {{ position: absolute; left: 0; right: 0; bottom: 45%; height: 2px;
            background: linear-gradient(90deg, transparent, var(--horizon), transparent); box-shadow: 0 0 18px 4px var(--glowA); }}
        .bgfx .scan {{ position: absolute; left: 0; right: 0; height: 140px; top: -140px;
            background: linear-gradient(180deg, transparent, var(--scan), transparent); animation: bgfx-scan 9s linear infinite; }}
        @keyframes bgfx-scan {{ to {{ transform: translateY(calc(100vh + 140px)); }} }}
        .bgfx .mote {{ position: absolute; bottom: -10px; width: 3px; height: 3px; border-radius: 50%;
            background: var(--neon); box-shadow: 0 0 6px var(--neon); opacity: .7; animation: bgfx-mote linear infinite; }}
        @keyframes bgfx-mote {{ to {{ transform: translateY(-105vh); opacity: 0; }} }}
        .bgfx .ring-hud {{ position: absolute; width: 520px; height: 520px; right: -180px; top: -160px; border-radius: 50%;
            border: 1px dashed var(--trace); outline: 1px solid var(--trace); outline-offset: -42px; animation: bgfx-spin 60s linear infinite; }}
        @keyframes bgfx-spin {{ to {{ transform: rotate(360deg); }} }}
        @media (prefers-reduced-motion: reduce) {{
            .bgfx .floor, .bgfx .scan, .bgfx .mote, .bgfx .ring-hud {{ animation: none; }}
            .bgfx .scan, .bgfx .mote {{ display: none; }}
        }}

        /* campus building tiles (inline SVG) */
        .twin-map {{ width: 100%; height: auto; display: block; background: rgba(8,13,22,0.55); border: 1px solid {BORDER_GLOW}; border-radius: 4px; }}
        .twin-map .road {{ stroke: rgba(91,184,255,0.22); stroke-width: 1; stroke-dasharray: 5 6; fill: none; }}
        .twin-map .walk {{ font: 8px 'JetBrains Mono', monospace; fill: #5f7fa3; letter-spacing: .12em; }}
        .twin-map .frame {{ fill: url(#tbp); stroke-width: 1.6; }}
        .twin-map .glow {{ stroke-width: 4; opacity: .28; filter: blur(3px); }}
        .twin-map .tb.crit .glow {{ animation: tb-pulse 1.6s ease-in-out infinite; }}
        .twin-map .roof {{ fill: none; stroke-width: 1.4; opacity: .85; }}
        .twin-map .led {{ fill: #35506f; }}
        .twin-map .led.on {{ fill: #33e1ff; filter: drop-shadow(0 0 3px #33e1ff); animation: tb-blink 1.4s steps(2, end) infinite; }}
        .twin-map text {{ font-family: 'JetBrains Mono', monospace; fill: #eaf4ff; pointer-events: none; }}
        .twin-map .t1 {{ font-size: 13px; font-weight: 700; letter-spacing: .08em; }}
        .twin-map .t2 {{ font-size: 9.5px; fill: #9db8d8; }}
        .twin-map .t3 {{ font-size: 10px; font-weight: 700; letter-spacing: .1em; }}
        .twin-map .t4 {{ font-size: 9px; fill: #7f9bbd; }}
        .twin-map .hair {{ stroke: rgba(91,184,255,0.25); stroke-width: 1; }}
        .twin-map .fl.pulse {{ animation: tb-pulse 1.2s ease-in-out infinite; }}
        @keyframes tb-pulse {{ 50% {{ opacity: .35; }} }}
        @keyframes tb-blink {{ 50% {{ opacity: .25; }} }}
        @media (prefers-reduced-motion: reduce) {{ .twin-map * {{ animation: none !important; }} }}

        h1, h2, h3 {{ font-family: 'Rajdhani', system-ui, sans-serif; letter-spacing: 0.03em; text-transform: uppercase; font-weight: 700; }}
        h1 {{
            font-family: 'Orbitron', 'Rajdhani', sans-serif; font-weight: 800; color: #eaf4ff;
            text-shadow: 0 0 10px rgba(57,135,229,0.65), 0 0 26px rgba(57,135,229,0.3);
        }}
        .stCaption, [data-testid="stCaptionContainer"], small {{ font-family: 'JetBrains Mono', monospace !important; color: {INK_SECONDARY} !important; }}

        /* KPI tiles: glass panel, neon left accent */
        [data-testid="stMetric"] {{
            background: linear-gradient(145deg, rgba(18,26,38,0.92), rgba(13,20,31,0.92));
            border: 1px solid {BORDER_GLOW}; border-left: 3px solid {NEON_CYAN}; border-radius: 4px;
            padding: 14px 16px; box-shadow: 0 0 18px rgba(57,135,229,0.08), inset 0 1px 0 rgba(255,255,255,0.03);
        }}
        [data-testid="stMetricValue"] {{ font-family: 'JetBrains Mono', monospace; text-shadow: 0 0 10px rgba(57,135,229,0.35); }}
        [data-testid="stMetricLabel"] {{ font-family: 'Rajdhani', sans-serif; letter-spacing: 0.08em; text-transform: uppercase; font-size: 0.8rem; color: #7fa6e0 !important; }}

        /* tabs: HUD nav bar */
        .stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {BORDER_GLOW}; }}
        .stTabs [data-baseweb="tab"] {{
            font-family: 'Rajdhani', sans-serif; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;
            color: #7fa6e0; background: transparent; border-radius: 4px 4px 0 0;
        }}
        .stTabs [aria-selected="true"] {{
            color: #eaf4ff !important; background: rgba(57,135,229,0.12);
            box-shadow: inset 0 -2px 0 {NEON_CYAN}, 0 0 14px rgba(57,135,229,0.22);
        }}

        /* buttons */
        div.stButton > button, div.stDownloadButton > button {{
            font-family: 'Rajdhani', sans-serif; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
            background: rgba(57,135,229,0.08); border: 1px solid {BORDER_GLOW}; color: #cfe3ff; border-radius: 3px;
            transition: all 0.15s ease;
        }}
        div.stButton > button:hover {{ border-color: {NEON_CYAN_BRIGHT}; box-shadow: 0 0 16px rgba(91,184,255,0.45); color: #ffffff; }}

        /* expander / containers as glass panels */
        div[data-testid="stExpander"] {{ background: rgba(13,20,31,0.55); border: 1px solid {BORDER_GLOW}; border-radius: 4px; }}

        /* dataframes */
        [data-testid="stDataFrame"] {{ border: 1px solid {BORDER_GLOW}; border-radius: 4px; overflow: hidden; }}

        /* sliders */
        [data-testid="stSlider"] [role="slider"] {{ box-shadow: 0 0 10px rgba(57,135,229,0.85); }}

        hr {{ border-color: {BORDER_GLOW} !important; }}

        ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
        ::-webkit-scrollbar-track {{ background: {VOID_BG}; }}
        ::-webkit-scrollbar-thumb {{ background: rgba(57,135,229,0.4); border-radius: 5px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: rgba(57,135,229,0.7); }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def campus_svg(tiles):
    """Futuristic building tiles as inline SVG, styled after the reference demo's building nodes:
    chamfered status-coloured frame, rooftop HVAC unit with blinking LED, text block, and a
    'floor stack' on the right (here six 4-hour blocks of the replay day; top = 20-24h)."""
    colors = (STATUS["good"], STATUS["warning"], STATUS["critical"])
    words = ("NORMAL", "LOW CONFIDENCE", "FLAGGED")
    W, H, CW, X0, Y_TOP, Y_BOT = 150, 112, 162, 10, 30, 172
    o = [
        '<svg class="twin-map" viewBox="0 0 660 300" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Block B room status">',
        '<defs><linearGradient id="tbp" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#132238"/><stop offset="1" stop-color="#0a121e"/></linearGradient></defs>',
        '<path class="road" d="M10,152 H650 M92,142 V162 M254,142 V162 M416,142 V162 M578,142 V162"/>',
        '<text class="walk" x="330" y="148" text-anchor="middle">CENTRAL WALKWAY</text>',
    ]
    for t in tiles:
        col, row = t["pos"]
        x, y = X0 + col * CW, (Y_TOP if row == 1 else Y_BOT)
        c, lvl = t["frame"], t["level"]
        poly = f"{x+12},{y} {x+W},{y} {x+W},{y+H-12} {x+W-12},{y+H} {x},{y+H} {x},{y+12}"
        hrs = "no alerts today" if t["hours"] == 0 else "%.1f h flagged today" % t["hours"]
        o.append(f'<g class="tb{" crit" if lvl == 2 else ""}">')
        o.append(f'<polygon class="glow" points="{poly}" style="stroke:{c};fill:none"/>')
        o.append(f'<polygon class="frame" points="{poly}" style="stroke:{c}"/>')
        # roof: antenna + HVAC unit with LED (blinks when the unit was on for most of the day)
        o.append(f'<path class="roof" style="stroke:{c}" d="M{x+24},{y} V{y-12} M{x+W-52},{y} V{y-7} H{x+W-20} V{y}"/>')
        o.append(f'<circle class="ant" cx="{x+24}" cy="{y-13}" r="2" style="fill:{c}"/>')
        o.append(f'<circle class="led{" on" if t["hvac"] > 0.5 else ""}" cx="{x+W-36}" cy="{y-3.5}" r="1.8"/>')
        o.append(f'<text class="t1" x="{x+14}" y="{y+24}">ROOM {t["room"]}</text>')
        o.append(f'<text class="t2" x="{x+14}" y="{y+40}">{t["temp"]:.1f} °C · HVAC {t["hvac"]*100:.0f}%</text>')
        o.append(f'<text class="t3" x="{x+14}" y="{y+64}" style="fill:{colors[lvl]}">{words[lvl]}</text>')
        o.append(f'<text class="t4" x="{x+14}" y="{y+80}">{hrs}</text>')
        o.append(f'<line class="hair" x1="{x+14}" y1="{y+92}" x2="{x+W-44}" y2="{y+92}"/>')
        for i, b in enumerate(reversed(t["blocks"])):
            o.append(f'<rect class="fl{" pulse" if b == 2 else ""}" x="{x+W-30}" y="{y+14+i*14}" width="16" height="10" rx="1" style="fill:{colors[b]};opacity:{1 if b else 0.4}"/>')
        o.append('</g>')
    o.append('</svg>')
    return "".join(o)


def inject_background():
    """Animated backdrop markup (styles live in inject_css). Motes use fixed pseudo-random
    positions/durations so the scene is identical on every rerun."""
    motes = "".join(
        f'<i class="mote" style="left:{(i * 37 + 11) % 100}%;animation-duration:{9 + (i * 5) % 10}s;'
        f'animation-delay:-{(i * 3) % 13}s"></i>'
        for i in range(14)
    )
    st.markdown(
        f'<div class="bgfx"><div class="floor"></div><div class="horizon"></div><div class="scan"></div>'
        f'<div class="ring-hud"></div>{motes}</div>',
        unsafe_allow_html=True,
    )


def base_layout(fig, title, yaxis_title, height=360):
    fig.update_layout(
        title=dict(text=title, font=dict(color=INK_PRIMARY_DARK, size=15, family="Rajdhani, sans-serif")),
        height=height,
        margin=dict(l=40, r=20, t=50, b=30),
        plot_bgcolor=SURFACE_DARK,
        paper_bgcolor=SURFACE_DARK,
        font=dict(color=INK_SECONDARY, family="JetBrains Mono, monospace"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(gridcolor=GRIDLINE, showline=True, linecolor=AXIS_LINE_NEON, zerolinecolor=GRIDLINE)
    fig.update_yaxes(title=yaxis_title, gridcolor=GRIDLINE, showline=True, linecolor=AXIS_LINE_NEON, zerolinecolor=GRIDLINE)
    return fig


inject_css()
inject_background()
st.title("◆ PLEIADES DIGITAL TWIN // BLOCK B")
st.caption(
    f"PLEIAData (Zenodo 7620136), Block B, {WINDOW_START} to {WINDOW_END} -- **real sensor data**, not synthetic. "
    "Historical replay + predictive simulation + anomaly detection; there is no injected-fault answer key here."
)

# ---------- replay bar ----------
# NOTE: the slider's key IS "day_idx" (no separate `value=` argument) so that
# programmatic jumps (play loop, preset buttons) which set st.session_state.day_idx
# directly are reflected in the slider on rerun. Passing both `value=` and `key=`
# would make Streamlit ignore the value on reruns, silently undoing those jumps.
def _toggle_play():
    # Callbacks run BEFORE the script reruns, so the button label below is always current
    # (rendering the label first and flipping state after the click left "Pause" stuck on screen).
    if not ss.playing and ss.day_idx >= N_DAYS - 1:
        ss._pending_day_idx = 0  # Play from the last day restarts instead of stopping at once
    ss.playing = not ss.playing


def _pause():
    ss.playing = False  # dragging the slider by hand takes over from the replay


rc1, rc2, rc3, rc4 = st.columns([3, 1, 1, 1])
with rc1:
    st.slider("Replay date", 0, N_DAYS - 1, key="day_idx", format="", on_change=_pause)
with rc2:
    st.markdown(f"**{day_of(ss.day_idx).strftime('%a %d %b %Y')}**")
with rc3:
    st.button("Pause" if ss.playing else "Play", key="play_btn", on_click=_toggle_play)
with rc4:
    preset_anom = temp_df.groupby("day")["any_anomaly"].sum().idxmax()
    if st.button("Jump to busiest anomaly day"):
        ss._pending_day_idx = DAYS.index(preset_anom)
        ss.playing = False
        st.rerun()
if ss.playing:
    st.caption("Replay running -- the other tabs are paused so playback stays smooth. Press Pause to explore them.")

day = day_of(ss.day_idx)
day_mask_rooms = rooms_df["day"] <= day
day_only_rooms = rooms_df["day"] == day
day_mask_temp = temp_df["day"] <= day
day_only_temp = temp_df["day"] == day
day_mask_energy = energy_df["day"] <= day
day_only_energy = energy_df["day"] == day

with st.expander("Cost & carbon assumptions (editable)"):
    c1, c2, c3 = st.columns(3)
    ss.cop = c1.number_input("Plant COP (thermal kWh -> electric kWh = / COP)", 1.0, 10.0, ss.cop, 0.1)
    ss.tariff = c2.number_input("Tariff ($/kWh electric)", 0.0, 2.0, ss.tariff, 0.01)
    ss.ef = c3.number_input("Grid emission factor (kg CO2/kWh electric)", 0.0, 2.0, ss.ef, 0.01)
    st.caption("These are placeholders, not measured utility rates. Applied only to Model B's energy-spike excess (kWh) -- room-level HVAC/temp flags have no per-room energy meter in this dataset, so no $ figure is fabricated for them.")

tab_overview, tab_rooms, tab_compare, tab_whatif, tab_model, tab_about = st.tabs(
    ["◉ Overview", "⬡ Rooms & Diagnosis", "⇄ Compare Rooms", "∿ Weather What-if", "⌬ Model & Evidence", "ℹ About / Real Data"]
)

# ============================================================ OVERVIEW ====
with tab_overview:
    energy_ytd = rooms_df.loc[day_mask_rooms].drop_duplicates(["Date"])["dif_cons"].sum()
    flagged_rooms_sofar = temp_df.loc[day_mask_temp, "room"][temp_df.loc[day_mask_temp, "any_anomaly"]].nunique()
    n_room_anoms_sofar = int(temp_df.loc[day_mask_temp, "any_anomaly"].sum())
    spike_rows = energy_df.loc[day_mask_energy & energy_df["anomaly_energy_spike"]]
    excess_kwh = float(spike_rows["energy_residual"].clip(lower=0).sum())
    today_flagged_rooms = temp_df.loc[day_only_temp & temp_df["any_anomaly"], "room"].unique().tolist()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("⚡ ENERGY SO FAR IN WINDOW", f"{energy_ytd:,.0f} kWh")
    k2.metric("▲ ROOM-TIMESTAMPS FLAGGED", f"{n_room_anoms_sofar:,}", help=f"{flagged_rooms_sofar}/{len(FOCUS_ROOMS)} rooms affected at least once")
    k3.metric("$ EST. COST OF ENERGY SPIKES", f"${cost(excess_kwh):,.0f}", help=f"{co2_kg(excess_kwh):.1f} kg CO2, at COP {ss.cop:g}")
    k4.metric("⬡ ROOMS FLAGGED TODAY", f"{len(today_flagged_rooms)} / {len(FOCUS_ROOMS)}")

    col_map, col_feed = st.columns([3, 2])
    with col_map:
        st.subheader("Campus status -- schematic layout (not to scale)")
        st.caption(
            "Each tile below is one monitored room in Block B (there's no real floor plan in the dataset, so this is a "
            "schematic grid, not true geometry). Pick what the color represents:"
        )
        color_by = st.radio("Color by", ["Anomaly status (today)", "Avg. temperature (today)"], horizontal=True, key="ov_colorby")
        today_rooms = rooms_df.loc[day_only_rooms]
        today_anoms = temp_df.loc[day_only_temp]
        tmin, tmax = rooms_df["V2"].min(), rooms_df["V2"].max()
        by_anoms = color_by.startswith("Anomaly")
        tiles = []
        for room in FOCUS_ROOMS:
            rrow = today_rooms[today_rooms["room"] == room]
            if rrow.empty:
                continue
            tr = today_anoms[today_anoms["room"] == room]
            blocks = [0] * 6  # six 4-hour blocks of the day (0 = 00-04h ... 5 = 20-24h) -> the building's "floor stack"
            if not tr.empty:
                crit = (tr["anomaly_setpoint_gap"] | tr["anomaly_temp_residual"]).to_numpy()
                warn = tr["anomaly_hvac_no_occupancy"].to_numpy()
                blk = (tr["Date"].dt.hour // 4).to_numpy()
                blocks = [2 if crit[blk == k].any() else (1 if warn[blk == k].any() else 0) for k in range(6)]
            level = max(blocks)
            avg_temp = float(rrow["V2"].mean())
            if by_anoms:
                frame = (STATUS["good"], STATUS["warning"], STATUS["critical"])[level]
            else:
                frac = (avg_temp - tmin) / (tmax - tmin)
                frame = SEQUENTIAL_BLUE[min(int(frac * (len(SEQUENTIAL_BLUE) - 1)), len(SEQUENTIAL_BLUE) - 1)]
            tiles.append(dict(
                room=room, pos=ROOM_GRID_POSITIONS[room], frame=frame, level=level, blocks=blocks, temp=avg_temp,
                hvac=float((rrow["V4"] > 0).mean()), hours=float(tr["any_anomaly"].sum()) * 10 / 60 if not tr.empty else 0.0,
            ))
        st.markdown(campus_svg(tiles), unsafe_allow_html=True)
        st.markdown(
            f"<span style='color:{STATUS['good']}'>&#9679;</span> normal &nbsp; "
            f"<span style='color:{STATUS['warning']}'>&#9679;</span> HVAC on, no motion and no CO2 rise (low confidence) &nbsp; "
            f"<span style='color:{STATUS['critical']}'>&#9679;</span> setpoint gap or unexpected temp",
            unsafe_allow_html=True,
        )
        st.caption(
            "Side stack on each building = six 4-hour blocks of the replay day (top = 20:00-24:00, bottom = 00:00-04:00), "
            "coloured by the worst anomaly in that block. The rooftop light blinks when the HVAC unit ran for over half the day."
        )

    with col_feed:
        st.subheader(f"Active alerts on {day.strftime('%d %b')}")
        if not today_flagged_rooms:
            st.info("No rooms are outside expected range today.")
        else:
            for room in today_flagged_rooms:
                tr = today_anoms[today_anoms["room"] == room]
                dominant = tr["anomaly_type"].mode().iloc[0] if not tr["anomaly_type"].mode().empty else None
                ty = TYPES.get(dominant, {})
                hours_flagged = tr["any_anomaly"].sum() * 10 / 60
                if st.button(f"Room {room} -- {ty.get('name', 'flagged')} ({hours_flagged:.1f} h)", key=f"feed_{room}"):
                    ss.jump_room = room
                    ss.jump_day = day

# ---- replay loop ----
# Placed right after the Overview tab on purpose: while playing, each tick reruns the script only up to
# here (Overview = the only thing that animates) and skips building the Plotly-heavy tabs below.
if ss.playing:
    if ss.day_idx < N_DAYS - 1:
        time.sleep(0.12)
        ss._pending_day_idx = ss.day_idx + 1
    else:
        ss.playing = False  # reached the end: rerun once more to draw the full page
    st.rerun()

# ======================================================= ROOMS & DIAGNOSIS =
with tab_rooms:
    default_room = ss.jump_room or FOCUS_ROOMS[0]
    sel_room = st.selectbox("Room", FOCUS_ROOMS, index=FOCUS_ROOMS.index(default_room))
    range_choice = st.radio("Window", ["7 days ending on replay date", "Full window"], horizontal=True)

    room_temp = temp_df[temp_df["room"] == sel_room].sort_values("Date")
    if range_choice.startswith("7"):
        start = pd.Timestamp(day) - pd.Timedelta(days=6)
        end = pd.Timestamp(day) + pd.Timedelta(days=1)
        rt = room_temp[(room_temp["Date"] >= start.tz_localize("UTC")) & (room_temp["Date"] < end.tz_localize("UTC"))]
    else:
        rt = room_temp

    st.caption(
        f"**What this shows:** the white line is Room {sel_room}'s actual recorded temperature. The blue line is what "
        "Model A *expected* it to be, given the weather, HVAC state, and occupancy at the time. Red-shaded periods are "
        "where actual departed enough from expected to be flagged -- those are the anomalies you can diagnose below."
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rt["Date"], y=rt[TEMP_TARGET], name="Actual", mode="lines", line=dict(color="#ffffff", width=1.6)))
    fig.add_trace(go.Scatter(x=rt["Date"], y=rt["V2_next_pred"], name="Expected (Model A)", mode="lines", line=dict(color=CATEGORICAL[0], width=1.6)))
    excess = rt[TEMP_TARGET] - rt["V2_next_pred"]
    fig.add_trace(go.Scatter(x=rt["Date"], y=np.where(rt["any_anomaly"], rt[TEMP_TARGET], rt["V2_next_pred"]), name="Flagged", mode="markers", marker=dict(color=STATUS["critical"], size=5), opacity=0.0, showlegend=False))
    for _, grp in rt[rt["any_anomaly"]].groupby((~rt["any_anomaly"]).cumsum()):
        if grp.empty or not grp["any_anomaly"].iloc[0]:
            continue
        fig.add_vrect(x0=grp["Date"].iloc[0], x1=grp["Date"].iloc[-1], fillcolor=STATUS["critical"], opacity=0.12, line_width=0)
    base_layout(fig, f"Room {sel_room}: actual vs. expected temperature (red shading = flagged)", "Temperature (deg C)")
    st.plotly_chart(fig, width="stretch")

    st.caption(
        f"**What this shows:** whether Room {sel_room}'s HVAC unit was running (top) and whether motion was detected "
        "(bottom) across the same time window, so you can see if the HVAC was on while the room looked empty."
    )
    fig_state = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("HVAC state (1 = on, 0 = off)", "Motion detected (1 = yes, 0 = no)"), vertical_spacing=0.15)
    fig_state.add_trace(go.Scatter(x=rt["Date"], y=rt["V4"], mode="lines", line=dict(color=ROOM_COLOR[sel_room], shape="hv", width=2), showlegend=False), row=1, col=1)
    fig_state.add_trace(go.Scatter(x=rt["Date"], y=rt["presence"], mode="lines", line=dict(color=ROOM_COLOR[sel_room], shape="hv", width=2), showlegend=False), row=2, col=1)
    fig_state.update_layout(height=340, margin=dict(l=40, r=20, t=50, b=30), plot_bgcolor="#0d141f", paper_bgcolor="#0d141f", font=dict(color=INK_SECONDARY))
    for ann in fig_state["layout"]["annotations"]:
        ann["font"] = dict(color=INK_PRIMARY_DARK, size=13, family="Rajdhani, sans-serif")
    fig_state.update_xaxes(gridcolor=GRIDLINE)
    fig_state.update_yaxes(gridcolor=GRIDLINE, range=[-0.1, 1.1])
    st.plotly_chart(fig_state, width="stretch")

    st.subheader("Alerts for this room")
    room_alerts = rt[rt["any_anomaly"]].copy()
    if room_alerts.empty:
        st.info("No flagged periods for this room in the selected window.")
    else:
        room_alerts["block"] = (~room_alerts["any_anomaly"]).cumsum()
        groups = [g for _, g in room_alerts.groupby("block") if g["any_anomaly"].iloc[0]]
        options = [f"{g['Date'].iloc[0].strftime('%m/%d %H:%M')} -> {g['Date'].iloc[-1].strftime('%m/%d %H:%M')} ({g['anomaly_type'].mode().iloc[0]})" for g in groups]
        sel_idx = st.selectbox("Select an alert to diagnose", range(len(options)), format_func=lambda i: options[i])
        g = groups[sel_idx]
        dominant = g["anomaly_type"].mode().iloc[0]
        ty = TYPES[dominant]
        hours = len(g) * 10 / 60
        st.markdown(f"#### Room {sel_room} -- {ty['name']}")
        st.caption(f"{g['Date'].iloc[0].strftime('%a %d %b %H:%M')} -> {g['Date'].iloc[-1].strftime('%a %d %b %H:%M')} ({hours:.1f} h)")
        m1, m2, m3 = st.columns(3)
        m1.metric("Avg temp during alert", f"{g['V2'].mean():.1f} C")
        m2.metric("Avg setpoint", f"{g['V12'].mean():.1f} C")
        m3.metric("Motion detected?", "Yes" if g["presence"].max() else "No")
        st.markdown(f"**Why it was flagged:** {ty['why']}")
        st.markdown(f"**What to check:** {ty['check']}")
        st.caption("This is a rule-based hint from the pattern of the deviation, not a confirmed cause.")

# ============================================================ COMPARE =====
with tab_compare:
    st.caption(
        "Why not just rank rooms by average temperature? A south-facing or corner room legitimately runs warmer/cooler "
        "than an interior one, so raw averages rank room *orientation*, not *problems*. Comparing each room's own "
        "flagged-anomaly rate is a fairer test of where something is actually going wrong."
    )
    rows = []
    for room in FOCUS_ROOMS:
        rt = temp_df[temp_df["room"] == room]
        rows.append({"room": room, "avg_temp": rt["V2"].mean(), "anomaly_rate": rt["any_anomaly"].mean() * 100, "hvac_on_pct": (rt["V4"] > 0).mean() * 100})
    cmp_df = pd.DataFrame(rows)

    c1, c2 = st.columns(2)
    with c1:
        d = cmp_df.sort_values("avg_temp", ascending=False)
        fig1 = go.Figure(go.Bar(x=d["avg_temp"], y=[f"Room {r}" for r in d["room"]], orientation="h", marker_color=CATEGORICAL[0]))
        base_layout(fig1, "Method 1: raw average temperature per room", "Avg. temperature (deg C)", height=280)
        st.plotly_chart(fig1, width="stretch")
        st.caption("Longer bar = warmer room on average. This ranks rooms by orientation/use, even when nothing is actually wrong -- see the caption above.")
    with c2:
        d = cmp_df.sort_values("anomaly_rate", ascending=False)
        fig2 = go.Figure(go.Bar(x=d["anomaly_rate"], y=[f"Room {r}" for r in d["room"]], orientation="h", marker_color=STATUS["critical"]))
        base_layout(fig2, "Method 2: % of time each room was flagged", "% of time flagged as anomalous", height=280)
        st.plotly_chart(fig2, width="stretch")
        st.caption("Longer bar = more of this room's own history looked abnormal relative to its own expected behavior -- a fairer 'which room has a problem' signal.")

    st.dataframe(cmp_df.rename(columns={"avg_temp": "avg temp (C)", "anomaly_rate": "% time flagged", "hvac_on_pct": "% time HVAC on"}).round(2), width="stretch")

# ============================================================ WHAT-IF =====
with tab_whatif:
    st.caption(
        "**What this can and can't answer.** Model A/B were trained on weather + HVAC state + occupancy as observed, "
        "so they can estimate the effect of a *weather* shift. They can't reliably answer 'what if we changed the "
        "schedule?' -- that was never varied in the data. XGBoost also doesn't extrapolate like a physics model: a "
        "shift that pushes outdoor temperature past the training range (here, {:.1f} to {:.1f} C) will under-react, "
        "since tree models plateau at the edges of what they saw.".format(float(rooms_df["tmed"].min()), float(rooms_df["tmed"].max()))
    )

    st.subheader("Room-level what-if (single timestamp)")
    sim_room = st.selectbox("Room", FOCUS_ROOMS, key="sim_room")
    room_temp_df = temp_df[temp_df["room"] == sim_room].sort_values("Date")
    baseline_idx = st.select_slider("Baseline timestamp", options=list(room_temp_df["Date"]), value=room_temp_df["Date"].iloc[len(room_temp_df) // 2], format_func=lambda d: pd.Timestamp(d).strftime("%m/%d %H:%M"))
    baseline_row = room_temp_df[room_temp_df["Date"] == baseline_idx].iloc[0]

    c1, c2, c3 = st.columns(3)
    with c1:
        sim_outdoor_temp = st.slider("Outdoor temp (C)", float(rooms_df["tmed"].min()), float(rooms_df["tmed"].max()), float(baseline_row["tmed"]))
    with c2:
        sim_setpoint = st.slider("HVAC setpoint (C)", 15.0, 30.0, float(baseline_row["V12"]))
    with c3:
        sim_presence = st.toggle("Occupied (motion detected)", value=bool(baseline_row["presence"]))

    model_a = load_temp_model()
    sim_features = baseline_row[TEMP_FEATURE_COLS].copy()
    sim_features["tmed"] = sim_outdoor_temp
    sim_features["V12"] = sim_setpoint
    sim_features["presence"] = int(sim_presence)
    pred = model_a.predict(pd.DataFrame([sim_features]))[0]

    m1, m2, m3 = st.columns(3)
    m1.metric("Actual temp now", f"{baseline_row['V2']:.2f} C")
    m2.metric("Actual next-step temp", f"{baseline_row[TEMP_TARGET]:.2f} C")
    m3.metric("Simulated next-step temp", f"{pred:.2f} C", delta=f"{pred - baseline_row['V2']:.2f} C vs current")

    st.caption(
        "**What this shows:** each point is Model A's predicted temperature 10 minutes from now if the thermostat were "
        "set to that value on the x-axis, with everything else (outdoor temp, occupancy, time) held at the values set above."
    )
    setpoints = list(range(16, 29))
    preds = [model_a.predict(pd.DataFrame([{**sim_features.to_dict(), "V12": sp}]))[0] for sp in setpoints]
    fig_sens = go.Figure(go.Scatter(x=setpoints, y=preds, mode="lines+markers", line=dict(color=CATEGORICAL[0], width=2), marker=dict(size=8)))
    base_layout(fig_sens, f"Room {sim_room}: how predicted temperature responds to setpoint", "Predicted temp 10 min ahead (deg C)", height=300)
    fig_sens.update_xaxes(title="HVAC setpoint (deg C)")
    st.plotly_chart(fig_sens, width="stretch")

    st.divider()
    st.subheader("Block-level weather scenario (whole 8-week window)")
    model_b = load_energy_model()
    base_table = energy_df.copy()
    shift_options = [-2, -1, 0, 1, 2, 3, 4]
    shift = st.radio("Outdoor temperature shift applied to the whole window", shift_options, index=2, horizontal=True, format_func=lambda v: f"{'+' if v > 0 else ''}{v} C")

    scenario_table = base_table.copy()
    scenario_table["tmed"] = scenario_table["tmed"] + shift
    scenario_pred = model_b.predict(scenario_table[ENERGY_FEATURE_COLS])
    baseline_total = base_table["energy_pred"].sum()
    scenario_total = scenario_pred.sum()
    delta = scenario_total - baseline_total

    w1, w2, w3 = st.columns(3)
    w1.metric("Baseline expected energy (window)", f"{baseline_total:,.0f} kWh")
    w2.metric(f"Scenario ({'+' if shift > 0 else ''}{shift} C) expected energy", f"{scenario_total:,.0f} kWh", delta=f"{delta:+,.0f} kWh")
    w3.metric("Est. cost / carbon change", f"${cost(delta):+,.0f}", help=f"{co2_kg(delta):+.0f} kg CO2")

    base_table["week"] = base_table["Date"].dt.isocalendar().week
    scenario_table["week"] = base_table["week"]
    scenario_table["energy_pred_scenario"] = scenario_pred
    by_week = base_table.groupby("week")["energy_pred"].sum().reset_index()
    by_week_scenario = scenario_table.groupby("week")["energy_pred_scenario"].sum().reset_index()
    by_week = by_week.merge(by_week_scenario, on="week")
    st.caption(
        f"**What this shows:** total predicted block energy for each week of the 8-week window, comparing the actual "
        f"recorded weather (Baseline) against the same weeks with outdoor temperature shifted by {'+' if shift > 0 else ''}{shift} C (Scenario)."
    )
    fig_wi = go.Figure()
    fig_wi.add_trace(go.Bar(x=by_week["week"], y=by_week["energy_pred"], name="Baseline (actual weather)", marker_color=CATEGORICAL[0]))
    fig_wi.add_trace(go.Bar(x=by_week["week"], y=by_week["energy_pred_scenario"], name=f"Scenario ({'+' if shift > 0 else ''}{shift} C)", marker_color=CATEGORICAL[1]))
    base_layout(fig_wi, "Predicted block energy by week: baseline vs. scenario", "Energy (kWh)", height=320)
    fig_wi.update_xaxes(title="ISO week number")
    st.plotly_chart(fig_wi, width="stretch")

# ============================================================ MODEL =======
with tab_model:
    st.subheader("How the expected value is computed")
    st.markdown(
        "- **Train** Model A/B (XGBoost) on this window's room-level and block-level data: weather, HVAC state/setpoint/mode, occupancy proxies, and time features.\n"
        "- **Predict** each step's next value (10-min indoor temp per room; next-hour block energy) from the current conditions.\n"
        "- **Compare** actual to expected; flag large residuals (z > 3) and two rule-based checks (HVAC on with no motion and no CO2 rise, setpoint gap) as anomalies -- see the Rooms & Diagnosis tab."
    )
    st.caption("Cause labels in the diagnosis panel are rule-based hints from the shape of the deviation, not confirmed diagnoses.")

    st.subheader("Prediction accuracy (ASHRAE Guideline 14 metrics)")
    st.caption("CV(RMSE) = typical error as % of mean actual value; NMBE = overall bias as % of mean actual. Guideline 14 accepts hourly models under 30% CV(RMSE) and within +-10% NMBE.")
    rows = [
        ("Model A: indoor temp (XGBoost)", metrics["temp_model"]),
        ("Model B: energy (XGBoost)", metrics["energy_model_xgboost"]),
        ("Model B baseline: energy (Prophet)", metrics["energy_model_prophet_baseline"]),
    ]
    mtable = pd.DataFrame(
        [{"Model": name, "RMSE": m["rmse"], "CV(RMSE) %": m["cv_rmse"], "NMBE %": m["nmbe"], "Guideline 14 pass?": "Yes" if abs(m["cv_rmse"]) < 30 and abs(m["nmbe"]) < 10 else "No", "n_test": m["n_test"]} for name, m in rows]
    )
    st.dataframe(mtable.round(2), width="stretch")

    st.subheader("Ablation: does occupancy (presence + CO2) actually help Model A?")
    ab = metrics["occupancy_ablation"]
    a1, a2 = st.columns(2)
    a1.metric("RMSE with presence + CO2", f"{ab['with_occupancy']['rmse']:.3f} C")
    a2.metric("RMSE without presence + CO2", f"{ab['without_occupancy']['rmse']:.3f} C")
    improvement = (ab["without_occupancy"]["rmse"] - ab["with_occupancy"]["rmse"]) / ab["without_occupancy"]["rmse"] * 100
    st.caption(f"Including occupancy features reduces RMSE by about {improvement:.1f}% at this 10-minute forecast horizon -- a real but modest effect: current temperature and HVAC state dominate the next-step prediction, occupancy is a second-order driver here. Mirrors the honest finding pattern of ablation studies in general: not every plausible feature earns its keep.")

    st.subheader("Energy model: driver-aware (XGBoost) vs. classical baseline (Prophet)")
    st.caption(
        "**What this shows:** on data the models never trained on (the held-out test period, the last ~20% of the "
        "window by time), the blue line is what actually happened, orange is XGBoost's prediction (using weather/HVAC/"
        "occupancy), and the dotted teal line is Prophet's prediction (using only the calendar pattern, no weather or "
        "HVAC data). The closer a line tracks the blue line, the more accurate that model is."
    )
    prophet_cmp = load_prophet_comparison()
    xgb_test = energy_df[energy_df["Date"].isin(pd.to_datetime(prophet_cmp["ds"]).dt.tz_localize("UTC"))][["Date", "energy_pred"]]
    merged = prophet_cmp.copy()
    merged["ds"] = pd.to_datetime(merged["ds"]).dt.tz_localize("UTC")
    merged = merged.merge(xgb_test, left_on="ds", right_on="Date", how="left")
    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Scatter(x=merged["ds"], y=merged["y"], name="Actual", mode="lines", line=dict(color=CATEGORICAL[0], width=2)))
    fig_cmp.add_trace(go.Scatter(x=merged["ds"], y=merged["energy_pred"], name="XGBoost (driver-aware)", mode="lines", line=dict(color=CATEGORICAL[1], width=2)))
    fig_cmp.add_trace(go.Scatter(x=merged["ds"], y=merged["prophet_pred"], name="Prophet (baseline)", mode="lines", line=dict(color=CATEGORICAL[2], width=2, dash="dot")))
    base_layout(fig_cmp, "Held-out test period: actual vs. predicted energy", "kWh", height=340)
    st.plotly_chart(fig_cmp, width="stretch")

    st.subheader("Known limitations")
    st.markdown(
        "- Occupancy is inferred from motion sensors that time out after seconds-to-minutes (per the dataset's own documentation) -- they undercount true presence. The HVAC-without-occupancy rule therefore also requires no CO2 rise (over 20 ppm above the room's own 24h baseline in the last 2 hours); this threshold is not calibrated against ground truth.\n"
        "- Scoped to Block B / 8 weeks / 8 rooms. The ingestion pipeline is parameterized by block and date range specifically so this can be re-run for more rooms/blocks/time.\n"
        "- No real floor-plan geometry exists in the dataset, so the twin's layout is a schematic 2D grid, not true spatial geometry.\n"
        "- XGBoost extrapolates poorly past the training data's feature range -- see the what-if tab's caveat.\n"
        "- There is no injected-fault ground truth here (unlike a synthetic-data demo), so detection accuracy can't be scored against known answers -- only plausibility-checked by inspection."
    )

# ============================================================ ABOUT =======
with tab_about:
    st.subheader("What sensor data this twin actually uses")
    st.dataframe(
        pd.DataFrame(
            [
                ("Block-level chilled/heating energy (kWh)", "Used (Model B target)"),
                ("Per-room indoor temperature", "Used (Model A target + feature)"),
                ("Per-room HVAC state/setpoint/mode/type", "Used"),
                ("Outdoor weather (temp, humidity, radiation, wind, precipitation, dew point)", "Used"),
                ("Per-room motion/presence", "Used, with the undercounting caveat noted throughout"),
                ("Per-room CO2", "Used as a model input and, as excess over a per-room baseline, to confirm rooms are empty in the HVAC-without-occupancy rule"),
                ("Per-room energy sub-metering", "Not available -- only block-level energy exists, hence no $ cost shown for room-level flags"),
                ("Floor-plan geometry / room coordinates", "Not available -- hence the schematic 2D grid, not a true floor plan"),
            ],
            columns=["Reading", "Status in this twin"],
        ),
        width="stretch",
    )
    st.subheader("Extending this twin")
    st.markdown(
        "- **More rooms/blocks:** Block A (40 rooms, no presence/CO2) and Block C (4 rooms, no presence/CO2) are in the same dataset; "
        "`src/ingest.py` is parameterized by block so this is a config change, not a rewrite.\n"
        "- **More time:** the dataset spans Jan 1 - Dec 18, 2021; this twin uses an 8-week winter slice. Re-running `src/ingest.py` and "
        "`src/models.py` with a wider `WINDOW_START`/`WINDOW_END` in `src/config.py` extends coverage (e.g. to compare winter vs. summer HVAC behavior).\n"
        "- **Real floor plan:** if building floor plans become available, the schematic grid in `palette.py` (`ROOM_GRID_POSITIONS`) can be replaced "
        "with true coordinates without touching the data pipeline."
    )
    st.caption("See README.md and the project plan file for full design rationale (why Block B, why a 2D grid not 3D, why XGBoost over ARIMA/Prophet).")
