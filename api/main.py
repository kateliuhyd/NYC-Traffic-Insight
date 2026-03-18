"""
NYC Traffic Insight — FastAPI backend.

Provides:
    GET  /           → filter form (borough + year selectors)
    GET  /map        → Folium traffic map filtered by borough & year
    POST /predict    → ML prediction (RF / HGB / Segmented)
    GET  /healthz    → liveness probe
    GET  /ping       → readiness probe
"""

import logging
import os
import threading

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel

from src.config import (
    ALLOWED_ORIGINS,
    BOROUGHS,
    DATA_YEAR_RANGE,
    FEATURE_COLS,
    MODEL_ALIASES,
)
from src import model_loader
from src.geojson_utils import build_traffic_map, filter_geojson_on_demand

logging.basicConfig(level=logging.INFO)

# ── App setup ─────────────────────────────────────────────────────
app = FastAPI(
    title="NYC Traffic Insight",
    description="Traffic volume prediction and map visualization for NYC.",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup ───────────────────────────────────────────────────────
@app.on_event("startup")
def startup() -> None:
    threading.Thread(target=model_loader.fetch_models_from_gcs, daemon=True).start()
    if os.getenv("PRELOAD_MODELS", "false").lower() == "true":
        threading.Thread(target=model_loader.load_models, daemon=True).start()


# ── Health / readiness ────────────────────────────────────────────
@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"


@app.get("/ping")
def ping():
    return {"status": "ok"}


# ── Landing page (filter form) ───────────────────────────────────
@app.get("/", response_class=HTMLResponse)
@app.get("/filter", response_class=HTMLResponse)
def filter_form():
    def opts(items):
        return "\n".join(f'<option value="{it}">{it}</option>' for it in items)

    return f"""
    <html>
    <head><title>NYC Traffic Map</title></head>
    <body>
        <h2>Select Borough and Year</h2>
        <form action="/map" method="get">
            <label for="borough">Borough:</label>
            <select name="borough" required>{opts(BOROUGHS)}</select><br><br>
            <label for="year">Year:</label>
            <select name="year" required>{opts(DATA_YEAR_RANGE)}</select><br><br>
            <button type="submit">Generate Map</button>
        </form>
    </body>
    </html>
    """


# ── Map endpoint ─────────────────────────────────────────────────
@app.get("/map")
def get_map(
    borough: str = Query(...),
    year: int = Query(...),
    background_tasks: BackgroundTasks = None,
):
    logging.info("/map hit: borough=%s, year=%s", borough, year)

    geojson = filter_geojson_on_demand(borough=borough, year=year)
    if not geojson["features"]:
        return JSONResponse(
            status_code=404,
            content={"error": f"No features found for {borough} in {year}"},
        )

    m = build_traffic_map(geojson)
    map_file = "/tmp/traffic_map.html"

    # Remove stale file if present
    try:
        if os.path.exists(map_file):
            os.remove(map_file)
    except PermissionError:
        return JSONResponse(content={"error": "Map file in use. Try again."}, status_code=500)

    m.save(map_file)
    if background_tasks:
        background_tasks.add_task(os.remove, map_file)
    return FileResponse(map_file, background=background_tasks)


# ── Prediction API ───────────────────────────────────────────────
class PredictRequest(BaseModel):
    hour_sin: float
    hour_cos: float
    wd_sin: float
    wd_cos: float
    month_sin: float
    month_cos: float
    vol_lag_1: float
    vol_roll_3h: float
    vol_roll_24h: float
    # Optional event flags (only needed for Segmented model)
    is_holiday: int = 0
    heavy_snow: int = 0


class PredictResponse(BaseModel):
    volume: float
    model: str


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, model: str = "rf"):
    models = model_loader.ensure_loaded()

    # Resolve short alias ("rf") → display name ("Random Forest")
    display_name = MODEL_ALIASES.get(model, model)
    estimator = models.get(display_name)
    if estimator is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model}'. Available: {list(models.keys())}",
        )

    df = pd.DataFrame([req.dict()])

    # Segmented model returns raw volume; sklearn estimators are trained on log1p
    is_segmented = getattr(estimator, "__class__", None).__name__ == "SegmentedModel"
    yhat = estimator.predict(df)[0]

    if is_segmented:
        volume = float(yhat)
    else:
        # Predict on feature subset only
        df_feat = df[FEATURE_COLS]
        yhat = estimator.predict(df_feat)[0]
        try:
            volume = float(np.expm1(yhat))
        except Exception:
            volume = float(yhat)

    return PredictResponse(volume=volume, model=display_name)


# ── Entry point ──────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port)
