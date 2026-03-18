"""
GeoJSON download, filtering, and Folium map building.

Shared between the FastAPI map endpoint and the Streamlit Map tab.
"""

import json
import logging
import os
from datetime import datetime

from src.config import GDRIVE_GEOJSON_FILE_ID, NYC_CENTER

logger = logging.getLogger(__name__)


# ── Google Drive helpers ──────────────────────────────────────────

def probe_drive_file(file_id: str) -> dict:
    """Probe whether a Google Drive *file_id* is publicly accessible."""
    import requests

    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        r = requests.get(url, allow_redirects=False, timeout=10)
        return {
            "url": url,
            "status_code": r.status_code,
            "location": r.headers.get("Location", ""),
            "is_public_like": r.status_code in (302, 303),
        }
    except Exception as e:
        return {"url": url, "error": str(e), "status_code": None}


# ── GeoJSON download + filter ────────────────────────────────────

def filter_geojson_on_demand(
    borough: str,
    year: int,
    *,
    file_id: str = "",
) -> dict:
    """Download a public GeoJSON from Google Drive and filter features.

    Args:
        borough: NYC borough name (case-insensitive).
        year: Calendar year to keep.
        file_id: Google Drive file id.  Falls back to the default in config.

    Returns:
        A minimal GeoJSON *FeatureCollection*.
    """
    import gdown

    fid = file_id or GDRIVE_GEOJSON_FILE_ID

    # Probe for access errors
    probe = probe_drive_file(fid)
    if probe.get("status_code") == 403:
        raise RuntimeError(
            "Google Drive returned 403. "
            "Please set the file sharing to 'Anyone with the link (Viewer)'."
        )
    if probe.get("status_code") and probe["status_code"] >= 400:
        raise RuntimeError(
            f"Google Drive returned HTTP {probe['status_code']} for file id {fid}"
        )

    temp_path = "/tmp/traffic_temp.geojson"
    gdown.download(id=fid, output=temp_path, quiet=True, use_cookies=False)

    try:
        with open(temp_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        logger.exception("Error loading GeoJSON")
        return {"type": "FeatureCollection", "features": []}
    finally:
        # Always clean up the temp file
        try:
            os.remove(temp_path)
        except OSError:
            pass

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
            continue

    return {"type": "FeatureCollection", "features": filtered}


# ── Folium map builder ────────────────────────────────────────────

def _volume_color(volume: float) -> str:
    if volume > 20:
        return "red"
    if volume > 10:
        return "orange"
    if volume > 5:
        return "yellow"
    return "green"


def build_traffic_map(geojson: dict):
    """Build a Folium ``Map`` from a filtered GeoJSON FeatureCollection.

    Returns the map object (caller decides how to serve/embed it).
    """
    import folium
    m = folium.Map(
        location=list(NYC_CENTER),
        zoom_start=12,
        tiles="CartoDB positron",
        attr="&copy; OpenStreetMap contributors & CARTO",
    )

    def style_function(feature):
        volume = feature["properties"].get("Volume", 0)
        return {"color": _volume_color(volume), "weight": 5, "opacity": 0.8}

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

    return m
