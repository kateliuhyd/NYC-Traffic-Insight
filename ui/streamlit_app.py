"""
NYC Traffic Insight — Streamlit UI.

Tabs:
    🗺️ Map       — filter-by-borough/year + Folium map
    📈 Predict   — ML volume prediction with human-friendly inputs
    🧪 Diagnostics — model file paths, GCS config, Drive access probe
"""

import logging
import math
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from streamlit.components.v1 import html

# ── Ensure project root is importable ────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (  # noqa: E402
    BOROUGHS,
    DATA_YEAR_RANGE,
    EXPECTED_MODELS,
    FEATURE_COLS,
    MODEL_BUCKET,
    MODEL_DIR,
    MODEL_PREFIX,
)
from src import model_loader  # noqa: E402
from src.geojson_utils import (  # noqa: E402
    build_traffic_map,
    filter_geojson_on_demand,
    probe_drive_file,
)
from src.config import GDRIVE_GEOJSON_FILE_ID  # noqa: E402

logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="NYC Traffic Insight", layout="wide")


# ── Streamlit-flavored secrets helper ─────────────────────────────
def _get_gcp_credentials():
    """Try to build GCP credentials from Streamlit secrets."""
    try:
        from google.oauth2.service_account import Credentials

        if "gcp_service_account" in st.secrets:
            return Credentials.from_service_account_info(st.secrets["gcp_service_account"])
    except Exception:
        pass
    return None


# ── Cached model loading ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_models():
    creds = _get_gcp_credentials()
    return model_loader.ensure_loaded(credentials=creds)


@st.cache_data(show_spinner=True)
def _cached_geojson(borough: str, year: int):
    return filter_geojson_on_demand(borough=borough, year=year)


# ── Prediction helpers ────────────────────────────────────────────
def _is_segmented(m) -> bool:
    try:
        return m.__class__.__name__ == "SegmentedModel"
    except Exception:
        return False


def _predict_volume(models: dict, payload: dict, model_key: str) -> float:
    import pandas as pd

    m = models.get(model_key)
    if m is None:
        raise RuntimeError(f"Model '{model_key}' not loaded.")

    df = pd.DataFrame([payload])

    if _is_segmented(m):
        return float(m.predict(df)[0])

    df_feat = df[FEATURE_COLS]
    yhat = m.predict(df_feat)[0]
    try:
        return float(np.expm1(yhat))
    except Exception:
        return float(yhat)


# ══════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════
st.title("🚗 NYC Traffic Volume Predictor")
tab_map, tab_pred, tab_diag = st.tabs(["🗺️ Map", "📈 Predict", "🧪 Diagnostics"])

# ── Map tab ──────────────────────────────────────────────────────
with tab_map:
    st.subheader("Filter by Borough & Year")
    left, right = st.columns(2)
    with left:
        borough = st.selectbox("Borough", BOROUGHS)
    with right:
        year = st.selectbox("Year", list(DATA_YEAR_RANGE))

    if st.button("Generate Map", type="primary"):
        try:
            data = _cached_geojson(borough=borough, year=year)
            if not data["features"]:
                st.warning("No features found for the selected filters.")
            else:
                m = build_traffic_map(data)
                html(m._repr_html_(), height=640, scrolling=True)
        except Exception as e:
            st.error(str(e))

# ── Predict tab ──────────────────────────────────────────────────
with tab_pred:
    st.subheader("Predict Hourly Traffic Volume")

    models = _load_models()
    available = sorted(models.keys()) or ["Random Forest"]
    model_key = st.selectbox("Model", available)

    # Event flags for segmented model
    is_holiday = False
    heavy_snow = False
    if model_key in models and _is_segmented(models[model_key]):
        st.markdown("**Event flags (for segmented model):**")
        c1, c2 = st.columns(2)
        with c1:
            is_holiday = st.checkbox("Holiday", value=False)
        with c2:
            heavy_snow = st.checkbox("Heavy snow", value=False)

    # ── Human-friendly inputs ────────────────────────────────────
    st.markdown("### Time & Calendar")
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        hour = st.slider("Hour of day", 0, 23, 12, help="0 = midnight, 12 = noon")
    with c2:
        weekday = st.selectbox(
            "Day of week",
            options=list(range(7)),
            format_func=lambda d: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d],
            index=0,
        )
    with c3:
        month = st.slider("Month", 1, 12, 6)

    # Auto-compute cyclic encodings
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)
    wd_sin = math.sin(2 * math.pi * weekday / 7)
    wd_cos = math.cos(2 * math.pi * weekday / 7)
    month_sin = math.sin(2 * math.pi * month / 12)
    month_cos = math.cos(2 * math.pi * month / 12)

    with st.expander("Computed cyclic features", expanded=False):
        st.json({
            "hour_sin": round(hour_sin, 4), "hour_cos": round(hour_cos, 4),
            "wd_sin": round(wd_sin, 4), "wd_cos": round(wd_cos, 4),
            "month_sin": round(month_sin, 4), "month_cos": round(month_cos, 4),
        })

    st.markdown("### Recent Volume History")
    v1, v2, v3 = st.columns(3, gap="large")
    vol_slider = dict(min_value=0.0, max_value=2000.0, step=1.0)
    with v1:
        vol_lag_1 = st.slider("Previous hour volume", value=100.0, **vol_slider)
    with v2:
        vol_roll_3h = st.slider("3-hour rolling avg", value=100.0, **vol_slider)
    with v3:
        vol_roll_24h = st.slider("24-hour rolling avg", value=100.0, **vol_slider)

    if st.button("Predict", type="primary"):
        payload = {
            "hour_sin": hour_sin, "hour_cos": hour_cos,
            "wd_sin": wd_sin, "wd_cos": wd_cos,
            "month_sin": month_sin, "month_cos": month_cos,
            "vol_lag_1": vol_lag_1, "vol_roll_3h": vol_roll_3h, "vol_roll_24h": vol_roll_24h,
        }
        if model_key in models and _is_segmented(models[model_key]):
            payload["is_holiday"] = int(is_holiday)
            payload["heavy_snow"] = int(heavy_snow)

        try:
            y = _predict_volume(models, payload, model_key=model_key)
            st.success(f"🚗 Predicted volume: **{y:,.0f}** vehicles/hour")
        except KeyError as e:
            st.error(f"Missing column for segmented model: {e}.")
        except Exception as e:
            st.error(str(e))

# ── Diagnostics tab ──────────────────────────────────────────────
with tab_diag:
    st.write("**Health**: OK ✅")

    cols = st.columns(3)
    with cols[0]:
        if st.button("Re-download models"):
            model_loader.reset()
            st.cache_resource.clear()
            _load_models()
            st.success("Models re-downloaded/reloaded.")
    with cols[1]:
        if st.button("Clear caches"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Streamlit caches cleared.")

    with st.expander("Model file paths"):
        paths = model_loader._model_paths(model_loader.MODELS_PATH)
        st.json({k: str(v) for k, v in paths.items()})

    with st.expander("Config & credentials"):
        st.json({
            "MODEL_BUCKET": MODEL_BUCKET,
            "MODEL_PREFIX": MODEL_PREFIX,
            "MODEL_DIR": str(MODEL_DIR),
            "Has gcp_service_account in secrets": "gcp_service_account" in st.secrets,
        })

    with st.expander("Probe Google Drive file access"):
        fid = st.text_input("Drive file_id", value=GDRIVE_GEOJSON_FILE_ID)
        if st.button("Check Drive access"):
            info = probe_drive_file(fid)
            st.json(info)
            if info.get("status_code") == 403:
                st.error("403: Set the Drive file to 'Anyone with the link (Viewer)'.")
            elif info.get("status_code") in (302, 303):
                st.success("Looks public (redirect detected).")
