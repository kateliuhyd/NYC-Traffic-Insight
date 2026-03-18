"""
NYC Traffic Insight — Streamlit UI.

Tabs:
    🗺️ Map         — filter-by-borough/year + Folium map
    📈 Predict      — ML volume prediction with human-friendly inputs
    📊 Compare      — side-by-side model comparison
    🧪 Diagnostics  — model file paths, GCS config, Drive access probe
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
    CYCLIC_FEATURES,
    DATA_YEAR_RANGE,
    EXPECTED_MODELS,
    FEATURE_COLS,
    LAG_FEATURES,
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

        if hasattr(st, "secrets") and "gcp_service_account" in st.secrets:
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
        return max(0, float(np.expm1(yhat)))
    except Exception:
        return max(0, float(yhat))


def _build_payload(
    hour: int, weekday: int, month: int,
    vol_lag_1: float, vol_roll_3h: float, vol_roll_24h: float,
    is_holiday: bool = False, heavy_snow: bool = False,
) -> dict:
    """Build a prediction payload from human-friendly inputs."""
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


# ══════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════
st.title("🚗 NYC Traffic Volume Predictor")
tab_map, tab_pred, tab_compare, tab_diag = st.tabs(
    ["🗺️ Map", "📈 Predict", "📊 Compare Models", "🧪 Diagnostics"]
)

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
    available = sorted(models.keys()) or ["HistGradientBoosting"]
    model_key = st.selectbox("Model", available, key="pred_model")

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
    st.markdown("### 🕐 Time & Calendar")
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

    with st.expander("ℹ️ Computed cyclic features", expanded=False):
        payload_preview = _build_payload(hour, weekday, month, 0, 0, 0)
        st.json({k: round(v, 4) for k, v in payload_preview.items() if k in CYCLIC_FEATURES})

    st.markdown("### 📊 Recent Volume History")

    st.info(
        "⚠️ **Data leakage note:** These lag features use actual past volumes. "
        "In a real deployment, you'd need a live data feed or use the previous day's "
        "values as proxies. Set to 0 if unknown.",
        icon="💡",
    )

    v1, v2, v3 = st.columns(3, gap="large")
    vol_slider = dict(min_value=0.0, max_value=2000.0, step=1.0)
    with v1:
        vol_lag_1 = st.slider("Previous hour volume", value=100.0, **vol_slider,
                              help="Set to 0 if not available. Uses vol_lag_1 feature.")
    with v2:
        vol_roll_3h = st.slider("3-hour rolling avg", value=100.0, **vol_slider)
    with v3:
        vol_roll_24h = st.slider("24-hour rolling avg", value=100.0, **vol_slider)

    if st.button("Predict", type="primary"):
        payload = _build_payload(
            hour, weekday, month,
            vol_lag_1, vol_roll_3h, vol_roll_24h,
            is_holiday, heavy_snow,
        )

        try:
            y = _predict_volume(models, payload, model_key=model_key)
            st.success(f"🚗 Predicted volume: **{y:,.0f}** vehicles/hour")
        except KeyError as e:
            st.error(f"Missing column for segmented model: {e}.")
        except Exception as e:
            st.error(str(e))

# ── Compare Models tab ───────────────────────────────────────────
with tab_compare:
    st.subheader("📊 Side-by-Side Model Comparison")
    st.markdown(
        "Compare predictions from all loaded models using the same inputs. "
        "This helps you understand each model's behavior."
    )

    models = _load_models()
    if not models:
        st.warning("No models loaded. Check the Diagnostics tab.")
    else:
        st.markdown("### Configure inputs")
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            cmp_hour = st.slider("Hour", 0, 23, 8, key="cmp_hour")
        with cc2:
            cmp_wd = st.selectbox(
                "Weekday", list(range(7)), key="cmp_wd",
                format_func=lambda d: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d],
            )
        with cc3:
            cmp_month = st.slider("Month", 1, 12, 3, key="cmp_month")

        cc4, cc5, cc6 = st.columns(3)
        with cc4:
            cmp_lag = st.slider("Vol lag 1h", 0.0, 2000.0, 100.0, key="cmp_lag")
        with cc5:
            cmp_roll3 = st.slider("Vol roll 3h", 0.0, 2000.0, 100.0, key="cmp_r3")
        with cc6:
            cmp_roll24 = st.slider("Vol roll 24h", 0.0, 2000.0, 100.0, key="cmp_r24")

        cmp_hol = st.checkbox("Holiday?", key="cmp_hol")
        cmp_snow = st.checkbox("Heavy snow?", key="cmp_snow")

        if st.button("🔍 Compare All Models", type="primary"):
            payload = _build_payload(
                cmp_hour, cmp_wd, cmp_month,
                cmp_lag, cmp_roll3, cmp_roll24,
                cmp_hol, cmp_snow,
            )

            results = []
            for name in sorted(models.keys()):
                try:
                    vol = _predict_volume(models, payload, model_key=name)
                    results.append({"Model": name, "Predicted Volume": f"{vol:,.0f}", "Status": "✅"})
                except Exception as e:
                    results.append({"Model": name, "Predicted Volume": "—", "Status": f"❌ {e}"})

            import pandas as pd
            df_results = pd.DataFrame(results)
            st.dataframe(df_results, use_container_width=True, hide_index=True)

            # Bar chart of predictions
            successful = [r for r in results if r["Status"] == "✅"]
            if successful:
                chart_df = pd.DataFrame({
                    "Model": [r["Model"] for r in successful],
                    "Volume": [float(r["Predicted Volume"].replace(",", "")) for r in successful],
                })
                st.bar_chart(chart_df.set_index("Model"), height=300)

        # ── Model info cards ─────────────────────────────────────
        st.markdown("### 📋 Model Reference")
        info_data = {
            "HistGradientBoosting": {
                "type": "HistGradientBoostingRegressor",
                "features": "9 cyclic + lag features",
                "target": "log1p(Vol)",
                "file_size": "~734 KB",
                "expected_r2": "~0.889",
            },
            "Segmented HGB": {
                "type": "SegmentedModel (dual HGB)",
                "features": "9 features + is_holiday + heavy_snow",
                "target": "log1p(Vol) → raw Vol",
                "file_size": "~1 MB",
                "expected_r2": "~0.891",
            },
            "Random Forest": {
                "type": "RandomForestRegressor (100 trees)",
                "features": "9 cyclic + lag features",
                "target": "log1p(Vol)",
                "file_size": "~319 MB ⚠️",
                "expected_r2": "~0.87",
            },
            "LightGBM": {
                "type": "LGBMRegressor",
                "features": "cyclic + weather + lag",
                "target": "log1p(Vol)",
                "file_size": "~1 MB (estimated)",
                "expected_r2": "TBD (not yet trained)",
            },
        }
        for name in sorted(models.keys()):
            info = info_data.get(name, {})
            with st.expander(f"**{name}**"):
                if info:
                    for k, v in info.items():
                        st.markdown(f"- **{k}**: {v}")
                else:
                    st.write("No additional info available.")

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
        has_secrets = False
        try:
            has_secrets = "gcp_service_account" in st.secrets
        except Exception:
            pass
        st.json({
            "MODEL_BUCKET": MODEL_BUCKET,
            "MODEL_PREFIX": MODEL_PREFIX,
            "MODEL_DIR": str(MODEL_DIR),
            "Has gcp_service_account in secrets": has_secrets,
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
