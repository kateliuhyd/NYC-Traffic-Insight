# streamlit_app.py — Pure Streamlit UI + lazy model loading + optional GCS fetch
import os, json, logging, threading
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit.components.v1 import html

logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="NYC Traffic", layout="wide")

# ---- Bigger typography (tweak SCALE as you like) ----
def inject_typography_css(scale: float = 1.2):
    css = f"""
    <style>
      /* Base text size for the whole app */
      [data-testid="stAppViewContainer"] * {{
        font-size: calc(16px * {scale});
      }}

      /* Headings */
      h1 {{ font-size: calc(2.0rem * {scale}) !important; }}
      h2 {{ font-size: calc(1.6rem * {scale}) !important; }}
      h3 {{ font-size: calc(1.3rem * {scale}) !important; }}

      /* Buttons */
      .stButton > button {{
        font-size: calc(1.0rem * {scale}) !important;
        padding: 0.45rem 1rem !important;
      }}

      /* Common input labels */
      [data-testid="stTextInput"] label,
      [data-testid="stNumberInput"] label,
      [data-testid="stSelectbox"] label,
      [data-testid="stSlider"] label,
      [data-testid="stMultiSelect"] label {{
        font-size: calc(1.0rem * {scale}) !important;
      }}

      /* Slider readouts/ticks (尽量兼容不同版本) */
      [data-testid="stSlider"] span, 
      [data-testid="stSlider"] div[role="slider"] {{
        font-size: calc(0.95rem * {scale}) !important;
      }}

      /* Tabs titles */
      [data-baseweb="tab"] p {{
        font-size: calc(0.95rem * {scale}) !important;
      }}

      /* Sidebar */
      section[data-testid="stSidebar"] * {{
        font-size: calc(15px * {scale}) !important;
      }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

inject_typography_css(scale=1.2)  # 想再大一些就调到 1.3 / 1.4


BASE_DIR = Path(__file__).resolve().parent

# Ensure the project root is importable so pickle can resolve custom classes,
# e.g., SegmentedModeling.SegmentedModeling.SegmentedModel
import sys
sys.path.insert(0, str(BASE_DIR))
try:
    import SegmentedModeling  # noqa: F401  # best-effort pre-import; ignore if missing
except Exception:
    pass


def _get_secret(name: str, default: str = "") -> str:
    """Fetch a config value from env first, then Streamlit secrets.
    Falls back to `default` if neither is available.
    """
    # Read from environment first
    v = os.getenv(name, "").strip()
    if v:
        return v
    # Then try from Streamlit secrets (secrets.toml / Cloud Secrets)
    try:
        return str(st.secrets.get(name, default)).strip()
    except Exception:
        return default


# Config (env/secrets overridable)
MODEL_BUCKET = _get_secret("MODEL_BUCKET", "")
MODEL_PREFIX = _get_secret("MODEL_PREFIX", "")
MODEL_DIR = Path(_get_secret("MODEL_DIR", "/tmp/models")).resolve()

# Map "display name" -> filename. The dict key will also appear in the UI model list.
EXPECTED_FILES = {
    "HistGradientBoosting": "hgb_model.joblib",
    "Random Forest":        "rf_model.joblib",
    "Segment HGB":          "segmented_model.joblib",
}

def _model_paths(root: Path):
    """Build absolute paths for expected model files under `root`."""
    return {k: root / v for k, v in EXPECTED_FILES.items()}

# Prefer baked-in files first; will be overridden by GCS fetch if enabled.
MODEL_FILES = _model_paths(BASE_DIR)

# In-memory model registry and concurrency primitives
MODELS = {}
_LOAD_LOCK = threading.Lock()
_FETCH_LOCK = threading.Lock()
_FETCHED = False


def fetch_models_from_gcs():
    """Optionally fetch model files from GCS into MODEL_DIR.
    Skips when MODEL_BUCKET/MODEL_PREFIX are not set.
    """
    global _FETCHED, MODEL_FILES
    if not MODEL_BUCKET or not MODEL_PREFIX:
        logging.info("GCS download skipped: MODEL_BUCKET/MODEL_PREFIX not set.")
        return

    with _FETCH_LOCK:
        if _FETCHED:
            return

        try:
            from google.cloud import storage  # lazy import to avoid hard dep locally
        except Exception as e:
            logging.exception("google-cloud-storage not available: %s", e)
            return

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        client = storage.Client()
        prefix = f"{MODEL_PREFIX}/" if MODEL_PREFIX else ""
        wanted = set(EXPECTED_FILES.values())
        found_any = False

        logging.info("Fetching models from gs://%s/%s", MODEL_BUCKET, prefix)
        for blob in client.list_blobs(MODEL_BUCKET, prefix=prefix):
            name = os.path.basename(blob.name)
            if name in wanted:
                dest = MODEL_DIR / name
                try:
                    blob.download_to_filename(str(dest))
                    logging.info("Downloaded %s -> %s", blob.name, dest)
                    found_any = True
                except Exception as e:
                    logging.exception("Failed to download %s: %s", blob.name, e)

        # After a successful fetch attempt, prefer MODEL_DIR over baked-in paths.
        MODEL_FILES = _model_paths(MODEL_DIR)

        # Log any missing files for easier debugging.
        for k, p in MODEL_FILES.items():
            if not p.exists():
                logging.error("Missing model after GCS fetch: %s at %s", k, p)

        _FETCHED = True
        if not found_any:
            logging.error("No expected model files found on GCS.")


def _load_models_once():
    """Load models at most once into the global MODELS dict.
    - Tries GCS (if configured) then falls back to baked-in locations.
    - Loads only the files that actually exist.
    """
    if not _FETCHED:
        fetch_models_from_gcs()

    with _LOAD_LOCK:
        if MODELS:
            return

        import joblib  # heavy-ish import; keep local

        # Candidate list: prefer fetched files, then baked-in files as fallback.
        candidates = list(MODEL_FILES.items())
        baked_files = _model_paths(BASE_DIR)
        for k, baked in baked_files.items():
            if k not in dict(candidates) or not dict(candidates)[k].exists():
                candidates.append((k, baked))

        # Deduplicate by key while preserving first occurrence (downloaded > baked-in)
        seen, ordered = set(), []
        for k, p in candidates:
            if k not in seen:
                seen.add(k)
                ordered.append((k, p))

        # Warn for missing files, then attempt to load existing ones.
        for name, path in ordered:
            if not path.exists():
                logging.error("Missing model file for %s at %s", name, path)
        for name, path in ordered:
            if not path.exists():
                continue
            try:
                MODELS[name] = joblib.load(path)
                logging.info("Loaded %s from %s", name, path)
            except Exception as e:
                logging.exception("Failed to load %s: %s", name, e)


@st.cache_resource(show_spinner=False)
def ensure_models_loaded():
    """Streamlit resource cache: load once per process/session."""
    _load_models_once()
    return MODELS


@st.cache_data(show_spinner=True)
def filter_geojson_on_demand(file_id: str, borough: str, year: int):
    """Download a public GeoJSON from Google Drive by file_id, then filter by borough/year.
    Returns a minimal FeatureCollection for Folium to render.
    """
    import gdown
    temp_path = "/tmp/traffic_temp.geojson"
    gdown.download(id=file_id, output=temp_path, quiet=True)

    try:
        with open(temp_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        logging.exception("Error loading GeoJSON: %s", e)
        return {"type": "FeatureCollection", "features": []}

    filtered = []
    for feature in raw.get("features", []):
        props = feature.get("properties", {})
        b = props.get("Borough", "").lower()
        ts = props.get("Timestamp", "")
        try:
            dt = datetime.fromisoformat(ts)
            if b == borough.lower() and dt.year == year:
                filtered.append(feature)
        except ValueError:
            # Skip malformed timestamps
            continue

    return {"type": "FeatureCollection", "features": filtered}


def render_map(geojson):
    """Render the filtered GeoJSON using Folium and embed it in Streamlit."""
    if not geojson["features"]:
        st.warning("No features found for the selected filters.")
        return

    m = folium.Map(location=[40.739, -73.952], zoom_start=12)

    # Simple style function based on traffic volume buckets
    def style_function(feature):
        volume = feature["properties"].get("Volume", 0)
        if volume > 20:
            color = "red"
        elif volume > 10:
            color = "orange"
        elif volume > 5:
            color = "yellow"
        else:
            color = "green"
        return {"color": color, "weight": 5, "opacity": 0.8}

    folium.GeoJson(
        geojson,
        name="Traffic Data",
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["Street", "From", "To", "Volume", "Timestamp", "Direction", "Borough"],
            aliases=["Street:", "From:", "To:", "Volume:", "Timestamp:", "Direction:", "Borough:"],
            localize=True,
        ),
    ).add_to(m)

    # Embed generated HTML in the Streamlit app
    html(m._repr_html_(), height=640, scrolling=True)


def _is_segmented_model(m) -> bool:
    """Return True if `m` looks like the custom SegmentedModel (used to switch logic)."""
    try:
        return (m.__class__.__name__ == "SegmentedModel") or m.__module__.startswith("SegmentedModeling")
    except Exception:
        return False


def predict_volume(models, payload: dict, model_key: str = "rf") -> float:
    """Make a single prediction given a payload dict and the selected model key.
    - For plain sklearn regressors trained on log1p(Vol): apply expm1 to invert.
    - For the custom SegmentedModel: it already returns raw volume, so use as-is.
    """
    m = models.get(model_key)
    if m is None:
        raise RuntimeError(f"Model '{model_key}' not loaded.")

    import pandas as pd
    import numpy as np

    df = pd.DataFrame([payload])

    if _is_segmented_model(m):
        # Segmented model expects extra flags: is_holiday/heavy_snow in the df.
        yhat = m.predict(df)[0]
        return float(yhat)
    else:
        # Plain sklearn estimator: usually trained on log1p(Vol)
        yhat = m.predict(df)[0]
        try:
            return float(np.expm1(yhat))
        except Exception:
            return float(yhat)


# UI 
st.title("🚗 Traffic Volume Predictor")
tab_map, tab_pred, tab_diag = st.tabs(["🗺️ Map", "📈 Predict", "🧪 Diagnostics"])

# Map tab
with tab_map:
    st.subheader("Filter by Borough & Year")
    left, right = st.columns(2)
    with left:
        borough = st.selectbox("Borough", ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"])
    with right:
        year = st.selectbox("Year", list(range(2014, 2024)))

    # On demand: fetch + filter + render
    if st.button("Generate Map", type="primary"):
        data = filter_geojson_on_demand(
            file_id="1wO3NjqVdg_GUpoEv1JpJHxZoV20Zz-Uq",  # Public Google Drive file id
            borough=borough,
            year=year,
        )
        render_map(data)

# Predict tab
with tab_pred:
    st.subheader("Predict hourly volume")

    # Load models and show only the ones that actually loaded
    models = ensure_models_loaded()
    available_models = sorted(list(models.keys())) or ["rf"]  # defensive fallback
    model_key = st.selectbox("Model", available_models)

    # Extra flags required by the segmented model to route between sub-models
    is_holiday = False
    heavy_snow = False
    if model_key in models and _is_segmented_model(models[model_key]):
        st.markdown("**Event flags (for segmented model):**")
        c1, c2 = st.columns(2)
        with c1:
            is_holiday = st.checkbox("is_holiday", value=False)
        with c2:
            heavy_snow = st.checkbox("heavy_snow", value=False)

    # Features (use sliders; only shown on Predict tab)
    st.markdown("### Features")

    # Presets for sliders
    unit_slider = dict(min_value=-1.0, max_value=1.0, step=0.01)
    vol_slider  = dict(min_value=0.0,  max_value=500.0, step=1.0)  # adjust max to your data scale

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        hour_sin  = st.slider("hour_sin",  value=0.00, **unit_slider, help="sin(2π * hour/24)")
        wd_sin    = st.slider("wd_sin",    value=0.00, **unit_slider, help="sin(2π * weekday/7)")
        month_sin = st.slider("month_sin", value=0.00, **unit_slider, help="sin(2π * month/12)")

    with col2:
        hour_cos  = st.slider("hour_cos",  value=1.00, **unit_slider, help="cos(2π * hour/24)")
        wd_cos    = st.slider("wd_cos",    value=1.00, **unit_slider, help="cos(2π * weekday/7)")
        month_cos = st.slider("month_cos", value=1.00, **unit_slider, help="cos(2π * month/12)")

    with col3:
        vol_lag_1    = st.slider("vol_lag_1",    value=10.0, **vol_slider)
        vol_roll_3h  = st.slider("vol_roll_3h",  value=10.0, **vol_slider)
        vol_roll_24h = st.slider("vol_roll_24h", value=10.0, **vol_slider)

    if st.button("Predict", type="primary"):
        # Base payload for sklearn models
        payload = {
            "hour_sin": hour_sin, "hour_cos": hour_cos,
            "wd_sin": wd_sin, "wd_cos": wd_cos,
            "month_sin": month_sin, "month_cos": month_cos,
            "vol_lag_1": vol_lag_1, "vol_roll_3h": vol_roll_3h, "vol_roll_24h": vol_roll_24h,
        }

        # Segmented model needs the two event flags for routing (not as features)
        if model_key in models and _is_segmented_model(models[model_key]):
            payload = {**payload, "is_holiday": int(is_holiday), "heavy_snow": int(heavy_snow)}

        try:
            y = predict_volume(models, payload, model_key=model_key)
            st.success(f"Predicted volume: {y:.2f}")
        except KeyError as e:
            # Most likely missing event flags when using segmented model
            st.error(f"Missing column for segmented model: {e}. Please set event flags above.")
        except Exception as e:
            st.error(str(e))

# Diagnostics tab
with tab_diag:
    st.write("**Health**: OK ✅")
    # Show the resolved file paths we attempted to load (for debugging)
    if st.button("List expected model files"):
        st.json({k: str(MODEL_FILES.get(k)) for k in EXPECTED_FILES.keys()})