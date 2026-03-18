"""
Shared model fetching and loading logic.

Used by both the FastAPI backend (api/main.py) and the Streamlit UI
(ui/streamlit_app.py) to avoid duplicating ~120 lines.
"""

import logging
import os
import threading
from pathlib import Path

from src.config import (
    EXPECTED_MODELS,
    MODEL_BUCKET,
    MODEL_DIR,
    MODEL_PREFIX,
    MODELS_PATH,
)

logger = logging.getLogger(__name__)

# ── Internal state ────────────────────────────────────────────────
_MODELS: dict = {}
_LOAD_LOCK = threading.Lock()
_FETCH_LOCK = threading.Lock()
_FETCHED = False


def _model_paths(root: Path) -> dict[str, Path]:
    """Build {display_name: absolute_path} for every expected model."""
    return {k: root / v for k, v in EXPECTED_MODELS.items()}


# Start with baked-in files; may switch to MODEL_DIR after GCS fetch.
_MODEL_FILES = _model_paths(MODELS_PATH)


def fetch_models_from_gcs(
    *,
    credentials=None,
) -> None:
    """Download expected model files from GCS into MODEL_DIR if missing.

    Args:
        credentials: Optional google.oauth2 credentials object.
                     When *None*, the default application credentials are used.
    """
    global _FETCHED, _MODEL_FILES

    if not MODEL_BUCKET or not MODEL_PREFIX:
        logger.info("GCS download skipped: MODEL_BUCKET/MODEL_PREFIX not set.")
        return

    with _FETCH_LOCK:
        if _FETCHED:
            return

        # All files already on disk? Skip the download.
        candidate = _model_paths(MODEL_DIR)
        already_here = all(
            (candidate[k].exists() or _MODEL_FILES[k].exists())
            for k in EXPECTED_MODELS
        )
        if already_here:
            logger.info("All model files already present; skipping GCS fetch.")
            _FETCHED = True
            return

        try:
            from google.cloud import storage
        except ImportError:
            logger.exception("google-cloud-storage not installed.")
            return

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        client = storage.Client(
            credentials=credentials,
            project=(getattr(credentials, "project_id", None) if credentials else None),
        )
        prefix = f"{MODEL_PREFIX}/" if MODEL_PREFIX else ""
        wanted = set(EXPECTED_MODELS.values())
        found_any = False

        logger.info("Fetching models from gs://%s/%s", MODEL_BUCKET, prefix)
        for blob in client.list_blobs(MODEL_BUCKET, prefix=prefix):
            name = os.path.basename(blob.name)
            if name in wanted:
                dest = MODEL_DIR / name
                try:
                    blob.download_to_filename(str(dest))
                    logger.info("Downloaded %s -> %s", blob.name, dest)
                    found_any = True
                except Exception:
                    logger.exception("Failed to download %s", blob.name)

        # Prefer downloaded files from now on.
        _MODEL_FILES = _model_paths(MODEL_DIR)
        for k, p in _MODEL_FILES.items():
            if not p.exists():
                logger.error("Missing model after GCS fetch: %s at %s", k, p)

        _FETCHED = True
        if not found_any:
            logger.error("No expected model files found on GCS.")


def load_models(*, credentials=None) -> dict:
    """Load all model files into memory (lazy, thread-safe).

    Returns the global model dict: {display_name: sklearn_estimator}.
    """
    if not _FETCHED:
        fetch_models_from_gcs(credentials=credentials)

    with _LOAD_LOCK:
        if _MODELS:
            return _MODELS

        import joblib

        # Build candidate list: downloaded > baked-in
        candidates: list[tuple[str, Path]] = list(_MODEL_FILES.items())
        baked = _model_paths(MODELS_PATH)
        for k, p in baked.items():
            if not dict(candidates).get(k, p).exists():
                candidates.append((k, p))

        # Deduplicate, keeping first occurrence
        seen: set[str] = set()
        ordered: list[tuple[str, Path]] = []
        for k, p in candidates:
            if k not in seen:
                seen.add(k)
                ordered.append((k, p))

        for name, path in ordered:
            if not path.exists():
                logger.error("Missing model file for %s at %s", name, path)
                continue
            try:
                _MODELS[name] = joblib.load(path)
                logger.info("Loaded %s from %s", name, path)
            except Exception:
                logger.exception("Failed to load %s", name)

    return _MODELS


def ensure_loaded(*, credentials=None) -> dict:
    """Convenience wrapper: load if not yet loaded, return dict."""
    if not _MODELS:
        load_models(credentials=credentials)
    return _MODELS


def reset() -> None:
    """Force re-download and reload (useful in diagnostic UIs)."""
    global _FETCHED, _MODEL_FILES
    _MODELS.clear()
    _FETCHED = False
    _MODEL_FILES = _model_paths(MODELS_PATH)
