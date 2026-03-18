"""
Centralized configuration for the NYC Traffic Insight project.

All magic constants, file IDs, model names, and environment-based
config live here so that api/ and ui/ never hard-code them.
"""

import os
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Google Drive ──────────────────────────────────────────────────
GDRIVE_GEOJSON_FILE_ID = os.getenv(
    "GDRIVE_GEOJSON_FILE_ID",
    "1wO3NjqVdg_GUpoEv1JpJHxZoV20Zz-Uq",
)

# ── GCS model storage ────────────────────────────────────────────
MODEL_BUCKET = os.getenv("MODEL_BUCKET", "").strip()
MODEL_PREFIX = os.getenv("MODEL_PREFIX", "").strip().strip("/")
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/tmp/models")).resolve()

# ── Model registry ───────────────────────────────────────────────
# key = display name (used in UI & API), value = filename on disk
# Ordered by recommendation: HGB is default, RF deprioritized (huge file, lower accuracy)
EXPECTED_MODELS: dict[str, str] = {
    "HistGradientBoosting": "hgb_model.joblib",
    "Segmented HGB":        "segmented_model.joblib",
    "LightGBM":             "lgbm_model.joblib",
    "Random Forest":        "rf_model.joblib",      # 319MB — consider removing
}

# Short aliases accepted by the FastAPI /predict endpoint
MODEL_ALIASES: dict[str, str] = {
    "hgb":  "HistGradientBoosting",
    "seg":  "Segmented HGB",
    "lgbm": "LightGBM",
    "rf":   "Random Forest",
}

# Default model for the API /predict endpoint
DEFAULT_MODEL = "hgb"

# Directory where .joblib files are baked into the image
MODELS_PATH = PROJECT_ROOT / "models"

# ── NYC constants ─────────────────────────────────────────────────
NYC_CENTER = (40.739, -73.952)
BOROUGHS = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
DATA_YEAR_RANGE = range(2014, 2025)

# ── CORS (production should override via env) ────────────────────
ALLOWED_ORIGINS = os.getenv(
    "CORS_ORIGINS", "*"
).split(",")

# ── Feature columns ──────────────────────────────────────────────
# These are split into categories to handle the data leakage issue:
#   - ALWAYS_AVAILABLE: can be computed from timestamp alone
#   - LAG_FEATURES: require historical volume data (may not be available at inference)
#   - EVENT_FEATURES: binary flags for special conditions

CYCLIC_FEATURES = [
    "hour_sin", "hour_cos",
    "wd_sin",   "wd_cos",
    "month_sin", "month_cos",
]

WEATHER_FEATURES = [
    "temperature_2m",
    "precipitation",
    "cloud_cover_low",
    "snow_depth",
    "visibility",
    "rain",
    "snowfall",
]

LAG_FEATURES = [
    "vol_lag_1",       # ⚠️ requires previous hour actual — potential data leakage
    "vol_roll_3h",     # ⚠️ requires 3-hour history
    "vol_roll_24h",    # ⚠️ requires 24-hour history
]

EVENT_FEATURES = [
    "is_holiday",
    "heavy_snow",
]

BOROUGH_FEATURE = "borough_encoded"

# Feature sets for different model types
FEATURE_COLS_V1 = CYCLIC_FEATURES + LAG_FEATURES  # Original (backward compat)

FEATURE_COLS_V2 = (
    CYCLIC_FEATURES
    + WEATHER_FEATURES
    + LAG_FEATURES
)  # Enhanced with weather

FEATURE_COLS = FEATURE_COLS_V1  # Default for existing models

# ── Training configuration ────────────────────────────────────────
TRAIN_TEST_SPLIT = 0.8  # time-based split (NOT random!)
RANDOM_STATE = 42
