# NYC Traffic Insight

**Data-driven analysis of NYC traffic and weather patterns using machine learning, data engineering, and geospatial visualization.**

---

## 📁 Project Structure

```text
NYC-Traffic-Insight/
├── src/                    # Shared core modules
│   ├── config.py           # Centralized configuration
│   ├── model_loader.py     # GCS fetch + joblib loading
│   ├── geojson_utils.py    # GeoJSON download/filter + Folium map
│   └── models/             # ML model definitions
│       └── segmented_model.py
├── api/                    # FastAPI backend
│   └── main.py             # REST API (/predict, /map, /filter)
├── ui/                     # Streamlit frontend
│   └── streamlit_app.py    # Interactive UI with map & prediction
├── training/               # Model training scripts
│   ├── train_hgb.py        # HistGradientBoosting
│   ├── train_rf.py         # Random Forest
│   ├── train_segmented.py  # Segmented model (holiday/snow aware)
│   ├── save_models.py      # Serialize all models to .joblib
│   └── test_inference.py   # Sanity check for trained models
├── pipelines/              # Data processing pipelines
│   ├── features.py         # Feature engineering
│   ├── weather_merge.py    # Weather data cleaning & merge
│   ├── raw_merge.py        # Traffic + weather merge
│   ├── enrich_weather.py   # Open-Meteo API data collection
│   ├── convert_csv_to_geojson.py
│   ├── point_to_linestring.py
│   └── downsize.py
├── notebooks/              # Experiments & exploration
│   ├── models.py           # Base model classes
│   ├── train_model.py      # Early training experiments
│   ├── LinearRegression.py
│   ├── NYC_Traffic_Congestion.py
│   └── LR_withCorrelationMatrix.ipynb
├── models/                 # Serialized model files (.joblib)
├── data/                   # Processed data
├── RawDataFiles/           # Raw data (weather CSVs, traffic CSV)
├── requirements.txt
├── Procfile                # Cloud Run entry point
├── cloudbuild.yaml         # CI/CD pipeline
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Streamlit UI

```bash
streamlit run ui/streamlit_app.py
```

### 3. Run the FastAPI backend

```bash
uvicorn api.main:app --reload
```

---

## 🔬 Key Components

### Shared Core (`src/`)
- **`config.py`** — All configuration in one place (Drive IDs, model names, GCS settings)
- **`model_loader.py`** — Thread-safe lazy model loading with optional GCS fetch
- **`geojson_utils.py`** — GeoJSON download from Google Drive, filtering, and Folium map building
- **`models/segmented_model.py`** — SegmentedModel class (holiday/snow-aware dual estimator)

### API (`api/`)
- FastAPI app with `/predict` (POST), `/map` (GET), `/filter` (GET), `/healthz`, `/ping`
- Supports model selection via short aliases (`rf`, `hgb`, `seg`)

### UI (`ui/`)
- Streamlit app with Map, Predict, and Diagnostics tabs
- Human-friendly prediction inputs (hour slider, weekday dropdown, month selector)

### Training (`training/`)
- Separate scripts for each model type (HGB, RF, Segmented)
- `save_models.py` for batch training and serialization

---

## 📊 Data Sources

- **Traffic Data:** NYC DOT [Automated Traffic Volume Counts](https://data.cityofnewyork.us/Transportation/Automated-Traffic-Volume-Counts/btm5-4yqh)
- **Weather Data:** [Open-Meteo API](https://open-meteo.com/) (historical hourly, 2014–2024)

---

## 🛠️ Technologies

| Category      | Tools                                           |
|---------------|-------------------------------------------------|
| Language      | Python 3.11+                                    |
| ML            | scikit-learn, pandas, numpy, matplotlib         |
| Geo/Mapping   | folium, geopandas, osmnx, shapely               |
| API           | FastAPI, uvicorn, pydantic                      |
| UI            | Streamlit                                       |
| Cloud         | Google Cloud Run, Cloud Build, GCS              |
| DevTools      | Git, Git LFS, GitHub                            |

---

## 👥 Contributors

- **Kate Liu**
- **Sania Shree**
- **Justin Forbes**
- **Nnaemeka Okonkwo**
---

## License

MIT
