"""
NYC Traffic Insight — Premium Streamlit Dashboard.

A visually stunning traffic analysis dashboard featuring:
    🏙️ Hero section with real-time-style key metrics
    🗺️ Interactive traffic heatmap of NYC
    📈 Prediction engine with human-friendly inputs
    📊 Borough traffic breakdown with charts
    🧪 Model diagnostics
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
    EXPECTED_MODELS,
    FEATURE_COLS,
    LAG_FEATURES,
    MODEL_BUCKET,
    MODEL_DIR,
    MODEL_PREFIX,
    GDRIVE_GEOJSON_FILE_ID,
)
from src import model_loader  # noqa: E402

logging.basicConfig(level=logging.INFO)
st.set_page_config(
    page_title="NYC Traffic Insight",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── Premium CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* Global */
.stApp { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1rem !important; max-width: 1200px; }

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(255,107,107,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -30%;
    left: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(78,205,196,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #FF6B6B, #FFE66D);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.hero-subtitle {
    color: rgba(255,255,255,0.7);
    font-size: 1.05rem;
    font-weight: 400;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1a1d24 0%, #252830 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.3);
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0.3rem 0;
}
.metric-label {
    font-size: 0.8rem;
    color: rgba(255,255,255,0.5);
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Section headers */
.section-header {
    font-size: 1.3rem;
    font-weight: 700;
    color: #FAFAFA;
    margin: 1.5rem 0 0.8rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid rgba(255,107,107,0.3);
}

/* Prediction result */
.prediction-result {
    background: linear-gradient(135deg, #0a2e1f, #1a4a35);
    border: 1px solid rgba(78,205,196,0.3);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    margin: 1rem 0;
}
.prediction-value {
    font-size: 2.5rem;
    font-weight: 800;
    color: #4ECDC4;
}
.prediction-label {
    color: rgba(255,255,255,0.6);
    font-size: 0.9rem;
}

/* Congestion badges */
.congestion-low { color: #4ECDC4; }
.congestion-med { color: #FFE66D; }
.congestion-high { color: #FF6B6B; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 0.3rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 0.5rem 1.2rem;
    font-weight: 600;
}

/* Hide Streamlit branding */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────

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


@st.cache_data(show_spinner=True, ttl=600)
def _load_sample_data(n_sample: int = 50000):
    """Load a sample of the merged data for visualizations."""
    data_path = _PROJECT_ROOT / "data" / "processed" / "merged_weather_traffic.csv"
    if not data_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(data_path, parse_dates=["Timestamp"])
    if len(df) > n_sample:
        df = df.sample(n_sample, random_state=42)
    return df


@st.cache_data(show_spinner=True, ttl=600)
def _get_hourly_stats():
    """Compute hourly average volumes by borough."""
    data_path = _PROJECT_ROOT / "data" / "processed" / "merged_weather_traffic.csv"
    if not data_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(data_path, usecols=["Timestamp", "Boro", "Vol", "Street"],
                     parse_dates=["Timestamp"])
    df["Hour"] = df["Timestamp"].dt.hour
    df["DayOfWeek"] = df["Timestamp"].dt.day_name()
    stats = df.groupby(["Boro", "Hour"]).agg(
        avg_vol=("Vol", "mean"),
        max_vol=("Vol", "max"),
        count=("Vol", "count"),
    ).reset_index()
    return stats


def _congestion_level(vol: float) -> tuple:
    """Return (label, color_class, emoji)."""
    if vol < 50:
        return "Low", "congestion-low", "🟢"
    elif vol < 150:
        return "Moderate", "congestion-med", "🟡"
    else:
        return "Heavy", "congestion-high", "🔴"


def _is_segmented(m) -> bool:
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


# ══════════════════════════════════════════════════════════════════
#  HERO BANNER
# ══════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🚦 NYC Traffic Insight</div>
    <div class="hero-subtitle">
        Real-time traffic volume analysis and prediction for New York City
        &nbsp;·&nbsp; 1.1M+ records &nbsp;·&nbsp; 1,814 road segments &nbsp;·&nbsp; 5 boroughs
    </div>
</div>
""", unsafe_allow_html=True)


# ── Key metrics row ──────────────────────────────────────────────
sample_df = _load_sample_data()
if not sample_df.empty:
    total_records = 1_133_538
    unique_streets = sample_df["Street"].nunique() if "Street" in sample_df.columns else 0
    avg_vol = sample_df["Vol"].mean() if "Vol" in sample_df.columns else 0
    date_range_start = sample_df["Timestamp"].min().year if "Timestamp" in sample_df.columns else 2014
    date_range_end = sample_df["Timestamp"].max().year if "Timestamp" in sample_df.columns else 2025

    m1, m2, m3, m4 = st.columns(4, gap="medium")
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Records</div>
            <div class="metric-value" style="color: #FF6B6B;">1.13M</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Road Segments</div>
            <div class="metric-value" style="color: #4ECDC4;">1,814</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Volume / hr</div>
            <div class="metric-value" style="color: #FFE66D;">{avg_vol:.0f}</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Data Span</div>
            <div class="metric-value" style="color: #C3A6FF;">{date_range_start}–{date_range_end}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("")

# ══════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════

tab_map, tab_pred, tab_trends, tab_compare, tab_diag = st.tabs(
    ["🗺️ Traffic Heatmap", "📈 Predict", "📊 Borough Trends", "🔬 Compare Models", "⚙️ Settings"]
)

# ── Traffic Heatmap tab ──────────────────────────────────────────
with tab_map:
    st.markdown('<div class="section-header">🗺️ NYC Traffic Congestion Heatmap</div>', unsafe_allow_html=True)
    st.markdown("Explore traffic intensity across NYC. Brighter areas = more vehicles.")

    hm_col1, hm_col2, hm_col3 = st.columns([1, 1, 1])
    with hm_col1:
        hm_borough = st.selectbox("Borough", ["All"] + BOROUGHS, key="hm_boro")
    with hm_col2:
        hm_year = st.selectbox("Year", list(range(2024, 2013, -1)), key="hm_year")
    with hm_col3:
        hm_hour = st.slider("Hour filter", 0, 23, 8, key="hm_hour",
                            help="Filter data by hour of day")

    if st.button("🔥 Generate Heatmap", type="primary", key="hm_btn"):
        with st.spinner("Building heatmap from traffic data..."):
            try:
                import folium
                from folium.plugins import HeatMap
                from streamlit_folium import st_folium

                data_path = _PROJECT_ROOT / "data" / "processed" / "merged_weather_traffic.csv"
                if data_path.exists():
                    # Load with filters
                    df_map = pd.read_csv(data_path, parse_dates=["Timestamp"])
                    df_map = df_map[df_map["Timestamp"].dt.year == hm_year]
                    df_map = df_map[df_map["Timestamp"].dt.hour == hm_hour]
                    if hm_borough != "All":
                        boro_map = {"Manhattan": "Manhattan", "Brooklyn": "Brooklyn",
                                    "Queens": "Queens", "Bronx": "Bronx",
                                    "Staten Island": "Staten Island",
                                    "Mn": "Manhattan", "Bk": "Brooklyn",
                                    "Qn": "Queens", "Bx": "Bronx", "SI": "Staten Island"}
                        df_map["boro_norm"] = df_map["Boro"].map(boro_map)
                        df_map = df_map[df_map["boro_norm"] == hm_borough]

                    if len(df_map) > 0 and "latitude" in df_map.columns and "longitude" in df_map.columns:
                        # Aggregate by location
                        agg = df_map.groupby(["latitude", "longitude"]).agg(
                            avg_vol=("Vol", "mean")
                        ).reset_index()
                        agg = agg.dropna(subset=["latitude", "longitude", "avg_vol"])

                        # Create map
                        center = [40.739, -73.952]
                        m = folium.Map(
                            location=center,
                            zoom_start=11,
                            tiles="CartoDB dark_matter",
                            control_scale=True,
                        )

                        # Add heatmap layer
                        heat_data = agg[["latitude", "longitude", "avg_vol"]].values.tolist()
                        HeatMap(
                            heat_data,
                            min_opacity=0.3,
                            radius=18,
                            blur=15,
                            gradient={0.2: "#4ECDC4", 0.5: "#FFE66D", 0.8: "#FF6B6B", 1.0: "#FF0000"},
                        ).add_to(m)

                        # Stats
                        level, cls, emoji = _congestion_level(agg["avg_vol"].mean())
                        s1, s2, s3 = st.columns(3)
                        with s1:
                            st.metric("Data Points", f"{len(agg):,}")
                        with s2:
                            st.metric("Avg Volume", f"{agg['avg_vol'].mean():.0f}")
                        with s3:
                            st.metric("Congestion", f"{emoji} {level}")

                        st_folium(m, width=None, height=550, use_container_width=True)
                    else:
                        st.warning("No data found for the selected filters.")
                else:
                    st.error("Merged dataset not found. Run `python pipelines/download_data.py` first.")
            except ImportError as e:
                st.error(f"Missing library: {e}. Run `pip install folium streamlit-folium`.")

# ── Predict tab ──────────────────────────────────────────────────
with tab_pred:
    st.markdown('<div class="section-header">📈 Traffic Volume Prediction</div>', unsafe_allow_html=True)
    st.markdown("Predict hourly traffic volume using machine learning models trained on 1.1M+ NYC records.")

    models = _load_models()
    available = sorted(models.keys()) or ["HistGradientBoosting"]

    pred_col1, pred_col2 = st.columns([3, 1])
    with pred_col2:
        model_key = st.selectbox("🤖 Model", available, key="pred_model")
        is_holiday = st.checkbox("🎄 Holiday", value=False)
        heavy_snow = st.checkbox("❄️ Heavy snow", value=False)

    with pred_col1:
        c1, c2, c3 = st.columns(3)
        with c1:
            hour = st.slider("🕐 Hour of day", 0, 23, 8, help="0=midnight, 12=noon, 17=rush hour")
        with c2:
            weekday = st.selectbox("📅 Day of week", list(range(7)), index=0,
                                   format_func=lambda d: ["Monday", "Tuesday", "Wednesday",
                                                           "Thursday", "Friday", "Saturday", "Sunday"][d])
        with c3:
            month = st.slider("📆 Month", 1, 12, 6)

        with st.expander("🔧 Advanced: Historical volume inputs", expanded=False):
            st.caption("⚠️ These use past volume data. Set to 0 if unknown — predictions will be less accurate.")
            v1, v2, v3 = st.columns(3)
            with v1:
                vol_lag_1 = st.number_input("Previous hour", min_value=0.0, max_value=5000.0,
                                            value=100.0, step=10.0)
            with v2:
                vol_roll_3h = st.number_input("3hr rolling avg", min_value=0.0, max_value=5000.0,
                                               value=100.0, step=10.0)
            with v3:
                vol_roll_24h = st.number_input("24hr rolling avg", min_value=0.0, max_value=5000.0,
                                                value=100.0, step=10.0)

    if st.button("⚡ Predict Traffic Volume", type="primary", use_container_width=True):
        payload = _build_payload(hour, weekday, month, vol_lag_1, vol_roll_3h, vol_roll_24h,
                                 is_holiday, heavy_snow)
        try:
            y = _predict_volume(models, payload, model_key=model_key)
            level, cls, emoji = _congestion_level(y)
            day_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][weekday]

            st.markdown(f"""
            <div class="prediction-result">
                <div class="prediction-label">Predicted traffic volume for {day_name} at {hour}:00</div>
                <div class="prediction-value">{y:,.0f}</div>
                <div class="prediction-label">vehicles per hour</div>
                <div style="margin-top: 0.5rem;">
                    <span class="{cls}" style="font-size: 1.1rem; font-weight: 600;">
                        {emoji} {level} congestion
                    </span>
                    &nbsp;·&nbsp;
                    <span style="color: rgba(255,255,255,0.5);">Model: {model_key}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(str(e))

# ── Borough Trends tab ───────────────────────────────────────────
with tab_trends:
    st.markdown('<div class="section-header">📊 Traffic Patterns by Borough</div>', unsafe_allow_html=True)

    hourly_stats = _get_hourly_stats()
    if hourly_stats.empty:
        st.warning("No data available. Run the data download pipeline first.")
    else:
        import plotly.express as px
        import plotly.graph_objects as go

        # Borough name mapping
        boro_map = {"Mn": "Manhattan", "Bk": "Brooklyn", "Qn": "Queens",
                    "Bx": "Bronx", "SI": "Staten Island",
                    "Manhattan": "Manhattan", "Brooklyn": "Brooklyn",
                    "Queens": "Queens", "Bronx": "Bronx", "Staten Island": "Staten Island"}
        hourly_stats["Borough"] = hourly_stats["Boro"].map(boro_map)

        # Hourly pattern chart
        st.markdown("#### 🕐 Hourly Traffic Patterns")
        fig = px.line(
            hourly_stats,
            x="Hour",
            y="avg_vol",
            color="Borough",
            title=None,
            labels={"avg_vol": "Avg Vehicles/hr", "Hour": "Hour of Day"},
            color_discrete_map={
                "Manhattan": "#FF6B6B",
                "Brooklyn": "#4ECDC4",
                "Queens": "#FFE66D",
                "Bronx": "#C3A6FF",
                "Staten Island": "#FF9F43",
            },
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#FAFAFA",
            legend=dict(orientation="h", y=-0.15),
            xaxis=dict(dtick=2, gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            height=400,
            margin=dict(t=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Borough comparison bar chart
        st.markdown("#### 🏙️ Peak Hour Volume by Borough")
        peak = hourly_stats.loc[hourly_stats.groupby("Borough")["avg_vol"].idxmax()]
        fig2 = px.bar(
            peak.sort_values("avg_vol", ascending=True),
            y="Borough",
            x="avg_vol",
            orientation="h",
            color="Borough",
            color_discrete_map={
                "Manhattan": "#FF6B6B",
                "Brooklyn": "#4ECDC4",
                "Queens": "#FFE66D",
                "Bronx": "#C3A6FF",
                "Staten Island": "#FF9F43",
            },
            labels={"avg_vol": "Peak Avg Volume/hr", "Borough": ""},
        )
        fig2.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#FAFAFA",
            showlegend=False,
            height=300,
            margin=dict(t=10, l=0),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        )
        fig2.update_traces(
            text=[f"{v:.0f}" for v in peak.sort_values("avg_vol", ascending=True)["avg_vol"]],
            textposition="outside",
        )
        st.plotly_chart(fig2, use_container_width=True)

# ── Compare Models tab ───────────────────────────────────────────
with tab_compare:
    st.markdown('<div class="section-header">🔬 Model Comparison</div>', unsafe_allow_html=True)
    st.markdown("Compare predictions from all models side-by-side on the same inputs.")

    models = _load_models()
    if not models:
        st.warning("No models loaded.")
    else:
        cc1, cc2, cc3, cc4 = st.columns(4)
        with cc1:
            cmp_hour = st.slider("Hour", 0, 23, 8, key="cmp_hour")
        with cc2:
            cmp_wd = st.selectbox("Weekday", list(range(7)), key="cmp_wd",
                                   format_func=lambda d: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d])
        with cc3:
            cmp_month = st.slider("Month", 1, 12, 6, key="cmp_month")
        with cc4:
            cmp_lag = st.number_input("Vol lag 1h", 0.0, 5000.0, 100.0, key="cmp_lag")

        if st.button("⚡ Compare All Models", type="primary", use_container_width=True):
            payload = _build_payload(cmp_hour, cmp_wd, cmp_month, cmp_lag, cmp_lag, cmp_lag)
            results = []
            for name in sorted(models.keys()):
                try:
                    vol = _predict_volume(models, payload, model_key=name)
                    level, _, emoji = _congestion_level(vol)
                    results.append({"Model": name, "Volume": vol, "Congestion": f"{emoji} {level}"})
                except Exception as e:
                    results.append({"Model": name, "Volume": 0, "Congestion": f"❌ Error"})

            df_r = pd.DataFrame(results)

            # Display as styled cards
            cols = st.columns(len(results))
            for i, (col, r) in enumerate(zip(cols, results)):
                with col:
                    color = ["#FF6B6B", "#4ECDC4", "#FFE66D", "#C3A6FF"][i % 4]
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{r['Model']}</div>
                        <div class="metric-value" style="color: {color};">{r['Volume']:,.0f}</div>
                        <div style="font-size: 0.85rem;">{r['Congestion']}</div>
                    </div>
                    """, unsafe_allow_html=True)

# ── Settings / Diagnostics tab ───────────────────────────────────
with tab_diag:
    st.markdown('<div class="section-header">⚙️ Settings & Diagnostics</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Reload models"):
            model_loader.reset()
            st.cache_resource.clear()
            _load_models()
            st.success("Models reloaded.")
    with col2:
        if st.button("🗑️ Clear all caches"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Caches cleared.")

    with st.expander("📁 Model files"):
        paths = model_loader._model_paths(model_loader.MODELS_PATH)
        for name, path in paths.items():
            exists = path.exists()
            size = f"{path.stat().st_size / 1024 / 1024:.2f} MB" if exists else "—"
            icon = "✅" if exists else "❌"
            st.markdown(f"- {icon} **{name}**: `{path.name}` ({size})")

    with st.expander("🔧 Configuration"):
        has_secrets = False
        try:
            has_secrets = "gcp_service_account" in st.secrets
        except Exception:
            pass
        st.json({
            "MODEL_BUCKET": MODEL_BUCKET or "(not set)",
            "MODEL_PREFIX": MODEL_PREFIX or "(not set)",
            "MODEL_DIR": str(MODEL_DIR),
            "GCP secrets configured": has_secrets,
        })
