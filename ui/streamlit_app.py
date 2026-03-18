"""
NYC Traffic Insight — Mission Control Dashboard.

AFSM-inspired dark mission control UI with:
    🚦 Live telemetry-style metric cards
    🗺️ Interactive traffic heatmap (CartoDB dark_matter)
    📈 ML-powered traffic volume prediction
    📊 Borough-level analysis with Plotly charts
"""

import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── Ensure project root is importable ────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (  # noqa: E402
    BOROUGHS,
    CYCLIC_FEATURES,
    DATA_YEAR_RANGE,
    FEATURE_COLS,
    LAG_FEATURES,
    MODEL_BUCKET,
    MODEL_DIR,
    MODEL_PREFIX,
)
from src import model_loader  # noqa: E402

logging.basicConfig(level=logging.INFO)
st.set_page_config(
    page_title="NYC Traffic Insight",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ══════════════════════════════════════════════════════════════════
#  AFSM-INSPIRED CSS — Mission Control Theme
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

:root {
  --bg: #080c10;
  --bg2: #0d1318;
  --bg3: #131a21;
  --border: rgba(0,255,180,0.12);
  --border2: rgba(0,255,180,0.22);
  --green: #00ffb4;
  --green-dim: rgba(0,255,180,0.15);
  --amber: #ffb700;
  --amber-dim: rgba(255,183,0,0.15);
  --red: #ff4444;
  --red-dim: rgba(255,68,68,0.15);
  --blue: #4db8ff;
  --blue-dim: rgba(77,184,255,0.12);
  --text: #c8d8e8;
  --text-dim: #5a7080;
  --text-bright: #e8f4ff;
  --mono: 'IBM Plex Mono', 'Courier New', monospace;
  --sans: 'IBM Plex Sans', 'Helvetica Neue', sans-serif;
}

/* === GLOBAL === */
.stApp {
  background: var(--bg) !important;
  font-family: var(--sans) !important;
  color: var(--text) !important;
}

/* Scan line overlay */
.stApp::before {
  content: '';
  position: fixed;
  inset: 0;
  background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.03) 2px, rgba(0,0,0,0.03) 4px);
  pointer-events: none;
  z-index: 1000;
}

.block-container {
  padding-top: 0 !important;
  max-width: 100% !important;
  padding-left: 2rem !important;
  padding-right: 2rem !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

/* === HEADER BAR === */
.hdr-bar {
  background: var(--bg2);
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  padding: 10px 0;
  gap: 16px;
  margin: -1rem -2rem 0 -2rem;
  padding: 10px 2rem;
}
.hdr-logo {
  font-family: var(--mono);
  font-size: 11px;
  font-weight: 500;
  color: var(--green);
  letter-spacing: 0.15em;
}
.hdr-sub {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--text-dim);
  letter-spacing: 0.1em;
}
.hdr-spacer { flex: 1; }
.hdr-stat {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-family: var(--mono);
  font-size: 10px;
  color: var(--text-dim);
  padding: 0 12px;
  border-left: 1px solid var(--border);
}
.hdr-stat .val { color: var(--green); font-weight: 500; }

/* Pulse dot */
.pulse-dot {
  display: inline-block;
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--green);
  animation: pulse-anim 1.5s infinite;
  vertical-align: middle;
  margin-right: 4px;
}
@keyframes pulse-anim {
  0%,100% { opacity: 1; box-shadow: 0 0 0 0 rgba(0,255,180,0.4); }
  50% { opacity: 0.6; box-shadow: 0 0 0 4px rgba(0,255,180,0); }
}

/* === PANEL HEADERS === */
.panel-hdr {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 0.15em;
  color: var(--text-dim);
  padding: 10px 0 8px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 8px;
  text-transform: uppercase;
  margin-bottom: 12px;
}
.panel-hdr .accent { color: var(--green); }

/* === METRIC CARDS (AFSM STYLE) === */
.mc-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 8px;
  margin: 12px 0 16px;
}
.mc {
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 10px 12px;
  position: relative;
  overflow: hidden;
  transition: border-color 0.2s;
}
.mc:hover { border-color: var(--border2); }
.mc::after {
  content: '';
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 2px;
  transform: scaleX(1);
  transform-origin: left;
}
.mc.mc-green::after { background: var(--green); }
.mc.mc-amber::after { background: var(--amber); }
.mc.mc-red::after { background: var(--red); }
.mc.mc-blue::after { background: var(--blue); }
.mc-label {
  font-family: var(--mono);
  font-size: 8px;
  letter-spacing: 0.12em;
  color: var(--text-dim);
  text-transform: uppercase;
  margin-bottom: 4px;
}
.mc-value {
  font-family: var(--mono);
  font-size: 22px;
  font-weight: 500;
  color: var(--text-bright);
  line-height: 1;
}
.mc-unit {
  font-size: 10px;
  color: var(--text-dim);
  margin-left: 2px;
}
.mc-bar {
  height: 2px;
  background: var(--border);
  border-radius: 1px;
  margin-top: 8px;
}
.mc-fill {
  height: 100%;
  border-radius: 1px;
  transition: width 0.5s ease;
}
.mc-trend {
  font-family: var(--mono);
  font-size: 8px;
  margin-top: 4px;
  color: var(--text-dim);
}

/* === CONGESTION GAUGE (Risk-style) === */
.gauge-wrap {
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 14px;
  margin: 12px 0;
}
.gauge-row {
  display: flex;
  align-items: center;
  gap: 14px;
}
.gauge-score {
  font-family: var(--mono);
  font-size: 36px;
  font-weight: 500;
  line-height: 1;
}
.gauge-info { flex: 1; }
.gauge-status {
  font-family: var(--mono);
  font-size: 10px;
  letter-spacing: 0.1em;
  margin-bottom: 4px;
}
.gauge-bar {
  height: 4px;
  border-radius: 2px;
  background: linear-gradient(to right, #00ffb4 0%, #ffb700 50%, #ff4444 100%);
  position: relative;
  margin-top: 8px;
}
.gauge-ptr {
  position: absolute;
  top: -3px;
  width: 10px; height: 10px;
  border-radius: 50%;
  background: white;
  border: 2px solid var(--bg);
  transform: translateX(-50%);
  box-shadow: 0 0 6px rgba(255,255,255,0.5);
}
.mode-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 0.12em;
  padding: 3px 8px;
  border-radius: 2px;
}
.mode-low { border: 1px solid var(--green); color: var(--green); background: var(--green-dim); }
.mode-mod { border: 1px solid var(--amber); color: var(--amber); background: var(--amber-dim); }
.mode-heavy { border: 1px solid var(--red); color: var(--red); background: var(--red-dim); animation: flash-b 0.6s infinite; }
@keyframes flash-b { 0%,100%{border-color:var(--red);} 50%{border-color:rgba(255,68,68,0.3);} }

/* === PREDICTION CARD (Agent Card style) === */
.pred-card {
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 4px;
  overflow: hidden;
  margin: 12px 0;
}
.pred-card.pred-ok { border-color: rgba(0,255,180,0.3); }
.pred-card.pred-warn { border-color: rgba(255,183,0,0.3); }
.pred-card.pred-crit { border-color: rgba(255,68,68,0.4); }
.pred-hdr {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 12px;
  border-bottom: 1px solid var(--border);
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 0.1em;
}
.pred-icon { font-size: 14px; }
.pred-title { flex: 1; font-weight: 500; color: var(--text-bright); }
.pred-score {
  font-size: 11px;
  padding: 2px 6px;
  border-radius: 2px;
  font-family: var(--mono);
}
.score-ok { background: var(--green-dim); color: var(--green); }
.score-warn { background: var(--amber-dim); color: var(--amber); }
.score-crit { background: var(--red-dim); color: var(--red); }
.pred-body {
  padding: 12px;
  font-family: var(--mono);
  text-align: center;
}
.pred-vol {
  font-size: 40px;
  font-weight: 500;
  line-height: 1;
}
.pred-vol-unit {
  font-size: 11px;
  color: var(--text-dim);
  margin-top: 4px;
  display: block;
}
.pred-evidence {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--text-dim);
  margin-top: 8px;
  border-top: 1px solid var(--border);
  padding-top: 8px;
}
.pred-cmd {
  background: var(--bg);
  border-top: 1px solid var(--border);
  padding: 8px 12px;
  font-family: var(--mono);
  font-size: 10px;
  color: var(--blue);
}
.pred-cmd .cmd-lbl {
  color: var(--text-dim);
  font-size: 8px;
  letter-spacing: 0.1em;
  margin-bottom: 3px;
}

/* === LOG ENTRIES === */
.log-entry {
  display: flex;
  gap: 8px;
  padding: 5px 0;
  border-left: 2px solid transparent;
  font-family: var(--mono);
  font-size: 11px;
  animation: log-in 0.3s ease;
}
@keyframes log-in { from { opacity: 0; transform: translateX(-8px); } to { opacity: 1; transform: none; } }
.log-entry.le-ok { border-left-color: var(--green); }
.log-entry.le-warn { border-left-color: var(--amber); }
.log-entry.le-crit { border-left-color: var(--red); }
.log-time { color: var(--text-dim); min-width: 48px; font-size: 10px; padding-left: 8px; }
.log-badge {
  font-size: 8px;
  padding: 1px 5px;
  border-radius: 2px;
  font-weight: 500;
  letter-spacing: 0.08em;
  min-width: 40px;
  text-align: center;
}
.lb-ok { background: rgba(0,255,180,0.15); color: var(--green); }
.lb-warn { background: rgba(255,183,0,0.15); color: var(--amber); }
.lb-crit { background: rgba(255,68,68,0.15); color: var(--red); }
.lb-cmd { background: rgba(77,184,255,0.15); color: var(--blue); }
.log-msg { flex: 1; color: var(--text); line-height: 1.4; }
.kw-val { color: var(--green); font-weight: 500; }
.kw-warn { color: var(--amber); font-weight: 500; }
.kw-crit { color: var(--red); font-weight: 500; }
.kw-cmd { color: var(--blue); font-weight: 500; }

/* === STABS OVERRIDES === */
.stTabs [data-baseweb="tab-list"] {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 4px;
  gap: 0;
  padding: 2px;
}
.stTabs [data-baseweb="tab"] {
  font-family: var(--mono);
  font-size: 10px;
  letter-spacing: 0.08em;
  border-radius: 3px;
  color: var(--text-dim) !important;
  padding: 6px 14px;
}
.stTabs [aria-selected="true"] {
  background: var(--bg3) !important;
  color: var(--green) !important;
}
.stTabs [data-baseweb="tab-highlight"] { background: transparent !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* Selectbox, slider, inputs */
[data-baseweb="select"] > div {
  background: var(--bg3) !important;
  border-color: var(--border) !important;
  font-family: var(--mono) !important;
  font-size: 11px !important;
  color: var(--text) !important;
}
.stSlider > div > div > div { background: var(--border) !important; }
.stSlider [data-baseweb="slider"] div[role="slider"] { background: var(--green) !important; }
.stNumberInput input {
  background: var(--bg3) !important;
  border-color: var(--border) !important;
  color: var(--text-bright) !important;
  font-family: var(--mono) !important;
}

/* Buttons */
.stButton > button[kind="primary"] {
  background: transparent !important;
  border: 1px solid var(--green) !important;
  color: var(--green) !important;
  font-family: var(--mono) !important;
  font-size: 10px !important;
  letter-spacing: 0.1em !important;
  border-radius: 3px !important;
  transition: all 0.15s !important;
}
.stButton > button[kind="primary"]:hover {
  background: var(--green-dim) !important;
  box-shadow: 0 0 12px rgba(0,255,180,0.15) !important;
}
.stButton > button {
  font-family: var(--mono) !important;
  font-size: 10px !important;
  letter-spacing: 0.1em !important;
  border-radius: 3px !important;
  border-color: var(--border) !important;
  color: var(--text-dim) !important;
  background: var(--bg3) !important;
}
.stButton > button:hover {
  border-color: var(--green) !important;
  color: var(--green) !important;
}

/* Expander */
.streamlit-expanderHeader {
  font-family: var(--mono) !important;
  font-size: 10px !important;
  color: var(--text-dim) !important;
  border: 1px solid var(--border) !important;
  background: var(--bg3) !important;
}

/* Metric labels */
[data-testid="stMetric"] {
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 8px 12px;
}
[data-testid="stMetricLabel"] { font-family: var(--mono) !important; font-size: 9px !important; letter-spacing: 0.1em !important; color: var(--text-dim) !important; }
[data-testid="stMetricValue"] { font-family: var(--mono) !important; color: var(--text-bright) !important; }

/* Checkbox */
.stCheckbox label { font-family: var(--mono) !important; font-size: 11px !important; color: var(--text) !important; }

/* Caption */
.stCaption, .stMarkdown small { font-family: var(--mono) !important; color: var(--text-dim) !important; }

/* Plotly chart backgrounds */
.stPlotlyChart { border: 1px solid var(--border); border-radius: 4px; }

/* Scrollbar */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════

def _get_gcp_credentials():
    try:
        from google.oauth2.service_account import Credentials
        if hasattr(st, "secrets") and "gcp_service_account" in st.secrets:
            return Credentials.from_service_account_info(st.secrets["gcp_service_account"])
    except Exception:
        pass
    return None


@st.cache_resource(show_spinner=False)
def _load_models():
    creds = _get_gcp_credentials()
    return model_loader.ensure_loaded(credentials=creds)


@st.cache_data(show_spinner=False, ttl=600)
def _get_summary_stats():
    data_path = _PROJECT_ROOT / "data" / "processed" / "merged_weather_traffic.csv"
    if not data_path.exists():
        return {}
    df = pd.read_csv(data_path, usecols=["Timestamp", "Boro", "Vol", "segmentid", "Street"],
                     parse_dates=["Timestamp"], nrows=200000)
    return {
        "total_rows": 1_133_538,
        "segments": df["segmentid"].nunique(),
        "avg_vol": df["Vol"].mean(),
        "streets": df["Street"].nunique(),
        "year_min": df["Timestamp"].min().year,
        "year_max": df["Timestamp"].max().year,
        "boroughs": df["Boro"].nunique(),
    }


@st.cache_data(show_spinner=False, ttl=600)
def _get_hourly_stats():
    data_path = _PROJECT_ROOT / "data" / "processed" / "merged_weather_traffic.csv"
    if not data_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(data_path, usecols=["Timestamp", "Boro", "Vol"],
                     parse_dates=["Timestamp"])
    df["Hour"] = df["Timestamp"].dt.hour
    return df.groupby(["Boro", "Hour"]).agg(
        avg_vol=("Vol", "mean"),
        max_vol=("Vol", "max"),
        count=("Vol", "count"),
    ).reset_index()


def _congestion(vol: float):
    """Return (label, css_class, color_var, score_pct)."""
    if vol < 50:
        return "LOW TRAFFIC", "mc-green", "var(--green)", int(vol / 300 * 100)
    elif vol < 150:
        return "MODERATE", "mc-amber", "var(--amber)", int(vol / 300 * 100)
    else:
        return "HEAVY CONGESTION", "mc-red", "var(--red)", min(95, int(vol / 300 * 100))


def _is_segmented(m):
    try:
        return m.__class__.__name__ == "SegmentedModel"
    except Exception:
        return False


def _build_payload(hour, weekday, month, vol_lag_1, vol_roll_3h, vol_roll_24h,
                   is_holiday=False, heavy_snow=False):
    return {
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "wd_sin": math.sin(2 * math.pi * weekday / 7),
        "wd_cos": math.cos(2 * math.pi * weekday / 7),
        "month_sin": math.sin(2 * math.pi * month / 12),
        "month_cos": math.cos(2 * math.pi * month / 12),
        "vol_lag_1": vol_lag_1,
        "vol_roll_3h": vol_roll_3h,
        "vol_roll_24h": vol_roll_24h,
        "is_holiday": int(is_holiday),
        "heavy_snow": int(heavy_snow),
    }


def _predict_volume(models, payload, model_key):
    m = models.get(model_key)
    if m is None:
        raise RuntimeError(f"Model '{model_key}' not loaded.")
    df = pd.DataFrame([payload])
    if _is_segmented(m):
        return float(m.predict(df)[0])
    df_feat = df[FEATURE_COLS]
    yhat = m.predict(df_feat)[0]
    try:
        return max(0, float(np.expm1(yhat)))
    except Exception:
        return max(0, float(yhat))


BORO_MAP = {"Mn": "Manhattan", "Bk": "Brooklyn", "Qn": "Queens",
            "Bx": "Bronx", "SI": "Staten Island",
            "Manhattan": "Manhattan", "Brooklyn": "Brooklyn",
            "Queens": "Queens", "Bronx": "Bronx", "Staten Island": "Staten Island"}


# ══════════════════════════════════════════════════════════════════
#  HEADER BAR
# ══════════════════════════════════════════════════════════════════

stats = _get_summary_stats()
models = _load_models()

st.markdown(f"""
<div class="hdr-bar">
    <span class="hdr-logo">NYC-TI v2.0</span>
    <span class="hdr-sub">TRAFFIC INSIGHT · MONITOR</span>
    <div class="hdr-spacer"></div>
    <div class="hdr-stat"><span class="pulse-dot"></span><span>STATUS</span><span class="val">ONLINE</span></div>
    <div class="hdr-stat"><span>RECORDS</span><span class="val">1.13M</span></div>
    <div class="hdr-stat"><span>SEGMENTS</span><span class="val">{stats.get("segments", "—")}</span></div>
    <div class="hdr-stat"><span>MODELS</span><span class="val">{len(models)}</span></div>
    <div class="hdr-stat"><span>DATA</span><span class="val">{stats.get("year_min","?")}–{stats.get("year_max","?")}</span></div>
</div>
""", unsafe_allow_html=True)

# ═══ METRIC CARDS ════════════════════════════════════════════════

avg_vol = stats.get("avg_vol", 0)
cong_label, cong_cls, cong_color, cong_pct = _congestion(avg_vol)

st.markdown(f"""
<div class="mc-grid">
    <div class="mc mc-green">
        <div class="mc-label">Total Records</div>
        <div class="mc-value">1.13M</div>
        <div class="mc-bar"><div class="mc-fill" style="width:85%;background:var(--green)"></div></div>
        <div class="mc-trend">▲ 2014–2025 coverage</div>
    </div>
    <div class="mc mc-blue">
        <div class="mc-label">Road Segments</div>
        <div class="mc-value">{stats.get("segments", "—")}</div>
        <div class="mc-bar"><div class="mc-fill" style="width:72%;background:var(--blue)"></div></div>
        <div class="mc-trend">across 5 boroughs</div>
    </div>
    <div class="mc {cong_cls}">
        <div class="mc-label">Avg Volume / hr</div>
        <div class="mc-value">{avg_vol:.0f}<span class="mc-unit">veh</span></div>
        <div class="mc-bar"><div class="mc-fill" style="width:{cong_pct}%;background:{cong_color}"></div></div>
        <div class="mc-trend">{cong_label}</div>
    </div>
    <div class="mc mc-green">
        <div class="mc-label">Unique Streets</div>
        <div class="mc-value">{stats.get("streets", "—")}</div>
        <div class="mc-bar"><div class="mc-fill" style="width:60%;background:var(--green)"></div></div>
        <div class="mc-trend">monitored locations</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════

tab_map, tab_pred, tab_trends, tab_compare, tab_diag = st.tabs(
    ["▣ HEATMAP", "◈ PREDICT", "◎ TRENDS", "◆ COMPARE", "⚙ SYSTEM"]
)

# ─── HEATMAP TAB ─────────────────────────────────────────────────
with tab_map:
    st.markdown('<div class="panel-hdr"><span class="accent">▣</span> TRAFFIC CONGESTION HEATMAP · LIVE</div>',
                unsafe_allow_html=True)

    hm1, hm2, hm3 = st.columns([1, 1, 1])
    with hm1:
        hm_borough = st.selectbox("BOROUGH", ["All"] + BOROUGHS, key="hm_b")
    with hm2:
        hm_year = st.selectbox("YEAR", list(range(2024, 2013, -1)), key="hm_y")
    with hm3:
        hm_hour = st.slider("HOUR FILTER", 0, 23, 8, key="hm_h")

    if st.button("▶ GENERATE HEATMAP", type="primary", key="hm_btn"):
        with st.spinner("Building heatmap..."):
            try:
                import folium
                from folium.plugins import HeatMap
                from streamlit_folium import st_folium

                data_path = _PROJECT_ROOT / "data" / "processed" / "merged_weather_traffic.csv"
                if data_path.exists():
                    df_map = pd.read_csv(data_path, parse_dates=["Timestamp"])
                    df_map = df_map[df_map["Timestamp"].dt.year == hm_year]
                    df_map = df_map[df_map["Timestamp"].dt.hour == hm_hour]
                    if hm_borough != "All":
                        df_map["boro_norm"] = df_map["Boro"].map(BORO_MAP)
                        df_map = df_map[df_map["boro_norm"] == hm_borough]

                    if len(df_map) > 0 and "latitude" in df_map.columns:
                        agg = df_map.groupby(["latitude", "longitude"]).agg(
                            avg_vol=("Vol", "mean")
                        ).reset_index().dropna()

                        m = folium.Map(location=[40.739, -73.952], zoom_start=11,
                                       tiles="CartoDB dark_matter", control_scale=True)
                        HeatMap(
                            agg[["latitude", "longitude", "avg_vol"]].values.tolist(),
                            min_opacity=0.3, radius=18, blur=15,
                            gradient={0.2: "#00ffb4", 0.5: "#ffb700", 0.8: "#ff4444", 1.0: "#ff0000"},
                        ).add_to(m)

                        label, cls, color, pct = _congestion(agg["avg_vol"].mean())
                        s1, s2, s3 = st.columns(3)
                        with s1:
                            st.metric("DATA POINTS", f"{len(agg):,}")
                        with s2:
                            st.metric("AVG VOLUME", f"{agg['avg_vol'].mean():.0f}")
                        with s3:
                            st.metric("STATUS", label)

                        st_folium(m, height=520, width=None, returned_objects=[])
                    else:
                        st.warning("No data for selected filters.")
                else:
                    st.error("Dataset not found — run `python pipelines/download_data.py`")
            except ImportError as e:
                st.error(f"Missing library: {e}")

# ─── PREDICT TAB ─────────────────────────────────────────────────
with tab_pred:
    st.markdown('<div class="panel-hdr"><span class="accent">◈</span> TRAFFIC VOLUME PREDICTION ENGINE</div>',
                unsafe_allow_html=True)

    available = sorted(models.keys()) or ["HistGradientBoosting"]

    p1, p2 = st.columns([3, 1])
    with p2:
        model_key = st.selectbox("MODEL", available, key="pred_m")
        is_holiday = st.checkbox("🎄 Holiday", value=False)
        heavy_snow = st.checkbox("❄️ Heavy snow", value=False)

    with p1:
        c1, c2, c3 = st.columns(3)
        with c1:
            hour = st.slider("HOUR", 0, 23, 8, help="0=midnight, 12=noon, 17=rush hour")
        with c2:
            weekday = st.selectbox("DAY", list(range(7)), index=0,
                                   format_func=lambda d: ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"][d])
        with c3:
            month = st.slider("MONTH", 1, 12, 6)

        with st.expander("⚙ ADVANCED: Historical volume inputs"):
            st.caption("⚠ Data leakage — set to 0 if unavailable")
            v1, v2, v3 = st.columns(3)
            with v1:
                vol_lag_1 = st.number_input("VOL LAG 1H", 0.0, 5000.0, 100.0, step=10.0)
            with v2:
                vol_roll_3h = st.number_input("VOL ROLL 3H", 0.0, 5000.0, 100.0, step=10.0)
            with v3:
                vol_roll_24h = st.number_input("VOL ROLL 24H", 0.0, 5000.0, 100.0, step=10.0)

    if st.button("▶ EXECUTE PREDICTION", type="primary", use_container_width=True):
        payload = _build_payload(hour, weekday, month, vol_lag_1, vol_roll_3h, vol_roll_24h,
                                 is_holiday, heavy_snow)
        try:
            y = _predict_volume(models, payload, model_key=model_key)
            label, _, color, pct = _congestion(y)
            day_names = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]

            if y < 50:
                sev, sev_icon, score_cls, card_cls, mode_cls = "OK", "✓", "score-ok", "pred-ok", "mode-low"
            elif y < 150:
                sev, sev_icon, score_cls, card_cls, mode_cls = "WARN", "▲", "score-warn", "pred-warn", "mode-mod"
            else:
                sev, sev_icon, score_cls, card_cls, mode_cls = "CRIT", "⚠", "score-crit", "pred-crit", "mode-heavy"

            st.markdown(f"""
            <div class="pred-card {card_cls}">
                <div class="pred-hdr">
                    <span class="pred-icon">{sev_icon}</span>
                    <span class="pred-title">PREDICTION — {day_names[weekday]} {hour:02d}:00</span>
                    <span class="pred-score {score_cls}">{label}</span>
                </div>
                <div class="pred-body">
                    <div class="pred-vol" style="color:{color};">{y:,.0f}</div>
                    <span class="pred-vol-unit">VEHICLES PER HOUR</span>
                    <div class="pred-evidence">
                        Model: <span class="kw-val">{model_key}</span> ·
                        Hour: <span class="kw-val">{hour:02d}:00</span> ·
                        Day: <span class="kw-val">{day_names[weekday]}</span> ·
                        Month: <span class="kw-val">{month}</span>
                    </div>
                </div>
                <div class="pred-cmd">
                    <div class="cmd-lbl">DECISION</div>
                    CMD: PREDICT model={model_key} hour={hour} day={weekday} month={month} → volume={y:,.0f}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Congestion gauge
            st.markdown(f"""
            <div class="gauge-wrap">
                <div class="gauge-row">
                    <div class="gauge-score" style="color:{color};">{y:,.0f}</div>
                    <div class="gauge-info">
                        <div class="gauge-status" style="color:{color};">{label}</div>
                        <div class="mode-badge {mode_cls}"><span class="pulse-dot"></span> {label}</div>
                    </div>
                </div>
                <div class="gauge-bar">
                    <div class="gauge-ptr" style="left:{pct}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Log-style output
            from datetime import datetime
            ts = datetime.now().strftime("%M:%S.%f")[:7]
            st.markdown(f"""
            <div style="margin-top:12px;">
                <div class="log-entry le-ok">
                    <span class="log-time">{ts}</span>
                    <span class="log-badge lb-cmd">CMD</span>
                    <span class="log-msg">Prediction executed — <span class="kw-val">{y:,.0f} veh/hr</span> via {model_key}</span>
                </div>
                <div class="log-entry le-{'ok' if y < 50 else 'warn' if y < 150 else 'crit'}">
                    <span class="log-time">{ts}</span>
                    <span class="log-badge lb-{'ok' if y < 50 else 'warn' if y < 150 else 'crit'}">{sev}</span>
                    <span class="log-msg">Congestion assessment: <span class="kw-{'warn' if y>=50 else 'val'}">{label}</span> — {day_names[weekday]} {hour:02d}:00 Month {month}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(str(e))

# ─── TRENDS TAB ──────────────────────────────────────────────────
with tab_trends:
    st.markdown('<div class="panel-hdr"><span class="accent">◎</span> BOROUGH TREND ANALYSIS</div>',
                unsafe_allow_html=True)

    hourly_stats = _get_hourly_stats()
    if hourly_stats.empty:
        st.warning("No data available.")
    else:
        import plotly.express as px

        hourly_stats["Borough"] = hourly_stats["Boro"].map(BORO_MAP)

        fig = px.line(
            hourly_stats, x="Hour", y="avg_vol", color="Borough",
            labels={"avg_vol": "Avg Vehicles/hr", "Hour": "Hour of Day"},
            color_discrete_map={
                "Manhattan": "#ff4444", "Brooklyn": "#00ffb4", "Queens": "#ffb700",
                "Bronx": "#c87eff", "Staten Island": "#4db8ff",
            },
        )
        fig.update_layout(
            plot_bgcolor="#080c10", paper_bgcolor="#080c10",
            font=dict(family="IBM Plex Mono", color="#c8d8e8", size=10),
            legend=dict(orientation="h", y=-0.15, font=dict(size=9)),
            xaxis=dict(dtick=2, gridcolor="rgba(0,255,180,0.06)", showline=True, linecolor="rgba(0,255,180,0.12)"),
            yaxis=dict(gridcolor="rgba(0,255,180,0.06)", showline=True, linecolor="rgba(0,255,180,0.12)"),
            height=380, margin=dict(t=10, l=50, r=10, b=50),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Peak volume bar chart
        st.markdown('<div class="panel-hdr"><span class="accent">◎</span> PEAK HOUR VOLUME</div>',
                    unsafe_allow_html=True)
        peak = hourly_stats.loc[hourly_stats.groupby("Borough")["avg_vol"].idxmax()]
        fig2 = px.bar(
            peak.sort_values("avg_vol", ascending=True),
            y="Borough", x="avg_vol", orientation="h", color="Borough",
            color_discrete_map={
                "Manhattan": "#ff4444", "Brooklyn": "#00ffb4", "Queens": "#ffb700",
                "Bronx": "#c87eff", "Staten Island": "#4db8ff",
            },
            labels={"avg_vol": "Peak Avg Volume/hr", "Borough": ""},
        )
        fig2.update_layout(
            plot_bgcolor="#080c10", paper_bgcolor="#080c10",
            font=dict(family="IBM Plex Mono", color="#c8d8e8", size=10),
            showlegend=False, height=260, margin=dict(t=10, l=0, r=10, b=30),
            xaxis=dict(gridcolor="rgba(0,255,180,0.06)"),
        )
        fig2.update_traces(
            text=[f"{v:.0f}" for v in peak.sort_values("avg_vol", ascending=True)["avg_vol"]],
            textposition="outside", textfont=dict(color="#c8d8e8", size=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

# ─── COMPARE TAB ─────────────────────────────────────────────────
with tab_compare:
    st.markdown('<div class="panel-hdr"><span class="accent">◆</span> MODEL COMPARISON · SIDE-BY-SIDE</div>',
                unsafe_allow_html=True)

    if not models:
        st.warning("No models loaded.")
    else:
        cc1, cc2, cc3, cc4 = st.columns(4)
        with cc1:
            cmp_hour = st.slider("HOUR", 0, 23, 8, key="cmp_h")
        with cc2:
            cmp_wd = st.selectbox("DAY", list(range(7)), key="cmp_d",
                                   format_func=lambda d: ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"][d])
        with cc3:
            cmp_month = st.slider("MONTH", 1, 12, 6, key="cmp_m")
        with cc4:
            cmp_lag = st.number_input("VOL LAG", 0.0, 5000.0, 100.0, key="cmp_l")

        if st.button("▶ COMPARE ALL MODELS", type="primary", use_container_width=True):
            payload = _build_payload(cmp_hour, cmp_wd, cmp_month, cmp_lag, cmp_lag, cmp_lag)
            results = []
            for name in sorted(models.keys()):
                try:
                    vol = _predict_volume(models, payload, model_key=name)
                    label, _, color, _ = _congestion(vol)
                    results.append({"name": name, "vol": vol, "label": label, "color": color, "ok": True})
                except Exception as e:
                    results.append({"name": name, "vol": 0, "label": f"ERROR: {e}", "color": "var(--red)", "ok": False})

            # Agent-style cards
            cards_html = ""
            for r in results:
                if r["ok"]:
                    sev = "pred-ok" if r["vol"] < 50 else "pred-warn" if r["vol"] < 150 else "pred-crit"
                    scls = "score-ok" if r["vol"] < 50 else "score-warn" if r["vol"] < 150 else "score-crit"
                    cards_html += f"""
                    <div class="pred-card {sev}" style="margin-bottom:8px;">
                        <div class="pred-hdr">
                            <span class="pred-title">{r['name']}</span>
                            <span class="pred-score {scls}">{r['label']}</span>
                        </div>
                        <div class="pred-body">
                            <div class="pred-vol" style="color:{r['color']};">{r['vol']:,.0f}</div>
                            <span class="pred-vol-unit">VEHICLES PER HOUR</span>
                        </div>
                    </div>
                    """
            st.markdown(cards_html, unsafe_allow_html=True)

# ─── SYSTEM TAB ──────────────────────────────────────────────────
with tab_diag:
    st.markdown('<div class="panel-hdr"><span class="accent">⚙</span> SYSTEM DIAGNOSTICS</div>',
                unsafe_allow_html=True)

    d1, d2 = st.columns(2)
    with d1:
        if st.button("↻ RELOAD MODELS"):
            model_loader.reset()
            st.cache_resource.clear()
            _load_models()
            st.success("Models reloaded.")
    with d2:
        if st.button("⊘ CLEAR CACHES"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Caches cleared.")

    with st.expander("◈ MODEL FILES"):
        paths = model_loader._model_paths(model_loader.MODELS_PATH)
        log_html = ""
        for name, path in paths.items():
            exists = path.exists()
            size = f"{path.stat().st_size / 1024:.0f} KB" if exists else "—"
            badge = "lb-ok" if exists else "lb-crit"
            status = "LOADED" if exists else "MISSING"
            log_html += f"""
            <div class="log-entry le-{'ok' if exists else 'crit'}">
                <span class="log-badge {badge}">{status}</span>
                <span class="log-msg"><span class="kw-val">{name}</span> — {path.name} ({size})</span>
            </div>
            """
        st.markdown(log_html, unsafe_allow_html=True)

    with st.expander("◎ CONFIGURATION"):
        has_secrets = False
        try:
            has_secrets = "gcp_service_account" in st.secrets
        except Exception:
            pass
        cfg_html = ""
        for k, v in [
            ("MODEL_BUCKET", MODEL_BUCKET or "(not set)"),
            ("MODEL_PREFIX", MODEL_PREFIX or "(not set)"),
            ("MODEL_DIR", str(MODEL_DIR)),
            ("GCP_SECRETS", "CONFIGURED" if has_secrets else "NOT SET"),
        ]:
            badge = "lb-ok" if v not in ["(not set)", "NOT SET"] else "lb-warn"
            cfg_html += f"""
            <div class="log-entry le-ok">
                <span class="log-badge {badge}">CFG</span>
                <span class="log-msg"><span class="kw-val">{k}</span> = {v}</span>
            </div>
            """
        st.markdown(cfg_html, unsafe_allow_html=True)
