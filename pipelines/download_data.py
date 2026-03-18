"""
Download fresh NYC traffic data from NYC Open Data and
weather data from Open-Meteo API.

Data Sources:
    Traffic: NYC DOT Automated Traffic Volume Counts
             https://data.cityofnewyork.us/Transportation/Automated-Traffic-Volume-Counts/7ym2-wayt
             API: Socrata Open Data API (SODA)

    Weather: Open-Meteo Historical Weather API
             https://archive-api.open-meteo.com/v1/archive

Usage:
    python pipelines/download_data.py                    # download both
    python pipelines/download_data.py --traffic-only     # just traffic
    python pipelines/download_data.py --weather-only     # just weather
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

# ── Configuration ─────────────────────────────────────────────────

# NYC Open Data - Automated Traffic Volume Counts (SODA API)
TRAFFIC_API = "https://data.cityofnewyork.us/resource/7ym2-wayt.csv"
TRAFFIC_APP_TOKEN = os.getenv("NYC_OPEN_DATA_TOKEN", "")  # optional, increases rate limit

# Open-Meteo Archive API (free, no key needed)
WEATHER_API = "https://archive-api.open-meteo.com/v1/archive"

# Borough representative coordinates for weather queries
BOROUGH_COORDS = {
    "Manhattan":      (40.7831, -73.9712),
    "Brooklyn":       (40.6782, -73.9442),
    "Queens":         (40.7282, -73.7949),
    "Bronx":          (40.8448, -73.8648),
    "Staten Island":  (40.5795, -74.1502),
}

# Weather variables to fetch
WEATHER_VARS = [
    "temperature_2m",
    "precipitation",
    "cloud_cover_low",
    "snow_depth",
    "visibility",
    "weather_code",
    "rain",
    "showers",
    "snowfall",
    "uv_index",
]

OUTPUT_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


# ── Traffic data download ────────────────────────────────────────

def download_traffic_data(
    start_year: int = 2014,
    end_year: int = 2025,
    output_path: str = None,
) -> pd.DataFrame:
    """Download all traffic volume records from NYC Open Data via SODA API."""

    out = Path(output_path) if output_path else OUTPUT_DIR / "traffic_volumes.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"📥 Downloading NYC traffic data ({start_year}–{end_year})...")

    all_rows = []
    offset = 0
    batch_size = 50000  # SODA max per request
    headers = {}
    if TRAFFIC_APP_TOKEN:
        headers["X-App-Token"] = TRAFFIC_APP_TOKEN

    while True:
        params = {
            "$limit": batch_size,
            "$offset": offset,
            "$order": "RequestID",
            "$where": f"Yr >= {start_year} AND Yr <= {end_year}",
        }
        print(f"  Fetching rows {offset:,}–{offset + batch_size:,}...", end=" ", flush=True)

        try:
            resp = requests.get(TRAFFIC_API, params=params, headers=headers, timeout=120)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"\n⚠️  Request failed at offset {offset}: {e}")
            if offset == 0:
                print("❌ Cannot reach NYC Open Data API. Check your network connection.")
                return pd.DataFrame()
            print("   Continuing with data downloaded so far.")
            break

        chunk = pd.read_csv(pd.io.common.StringIO(resp.text))
        n = len(chunk)
        print(f"got {n:,} rows")

        if n == 0:
            break

        all_rows.append(chunk)
        offset += batch_size

        if n < batch_size:
            break

        time.sleep(0.5)  # be polite to the API

    if not all_rows:
        print("❌ No data downloaded.")
        return pd.DataFrame()

    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(out, index=False)
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"✅ Saved {len(df):,} rows to {out} ({size_mb:.1f} MB)")

    return df


# ── Weather data download ────────────────────────────────────────

def download_weather_for_borough(
    borough: str,
    lat: float,
    lon: float,
    start_date: str = "2014-01-01",
    end_date: str = "2025-12-31",
) -> pd.DataFrame:
    """Download hourly weather data for a single borough from Open-Meteo."""

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(WEATHER_VARS),
        "timezone": "America/New_York",
        "temperature_unit": "fahrenheit",
    }

    print(f"  📥 {borough} ({lat}, {lon})...", end=" ", flush=True)

    try:
        resp = requests.get(WEATHER_API, params=params, timeout=120)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        print(f"❌ Failed: {e}")
        return pd.DataFrame()

    hourly = data.get("hourly", {})
    if "time" not in hourly:
        print("❌ No hourly data in response")
        return pd.DataFrame()

    df = pd.DataFrame(hourly)
    df.rename(columns={"time": "date"}, inplace=True)
    df["latitude"] = lat
    df["longitude"] = lon
    df["borough"] = borough

    print(f"got {len(df):,} hours")
    return df


def download_all_weather(
    start_date: str = "2014-01-01",
    end_date: str = "2025-12-31",
    output_path: str = None,
) -> pd.DataFrame:
    """Download weather data for all 5 boroughs and combine."""

    out = Path(output_path) if output_path else OUTPUT_DIR / "weather_all_boroughs.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"📥 Downloading weather data ({start_date} to {end_date})...")

    dfs = []
    for borough, (lat, lon) in BOROUGH_COORDS.items():
        df = download_weather_for_borough(borough, lat, lon, start_date, end_date)
        if not df.empty:
            dfs.append(df)
        time.sleep(1)  # rate limit

    if not dfs:
        print("❌ No weather data downloaded.")
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(out, index=False)
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"✅ Saved {len(combined):,} rows to {out} ({size_mb:.1f} MB)")

    return combined


# ── Merge traffic + weather ──────────────────────────────────────

def merge_traffic_weather(
    traffic_path: str = None,
    weather_path: str = None,
    output_path: str = None,
) -> pd.DataFrame:
    """Merge traffic and weather data on timestamp + borough."""

    t_path = traffic_path or str(OUTPUT_DIR / "traffic_volumes.csv")
    w_path = weather_path or str(OUTPUT_DIR / "weather_all_boroughs.csv")
    out = Path(output_path) if output_path else PROCESSED_DIR / "merged_weather_traffic.csv"

    print("🔗 Merging traffic + weather data...")

    traffic = pd.read_csv(t_path)
    weather = pd.read_csv(w_path)

    # Normalize column names: SODA API returns lowercase (yr, m, d, hh, vol, boro)
    # but some raw files use capitalized (Yr, M, D, HH, Vol, Boro)
    col_rename = {}
    for c in traffic.columns:
        cl = c.lower()
        if cl in ("yr",):
            col_rename[c] = "Yr"
        elif cl == "m":
            col_rename[c] = "M"
        elif cl == "d":
            col_rename[c] = "D"
        elif cl == "hh":
            col_rename[c] = "HH"
        elif cl == "vol":
            col_rename[c] = "Vol"
        elif cl == "boro":
            col_rename[c] = "Boro"
        elif cl == "street":
            col_rename[c] = "Street"
        elif cl == "fromst":
            col_rename[c] = "From"
        elif cl == "tost":
            col_rename[c] = "To"
        elif cl == "direction":
            col_rename[c] = "Direction"
    if col_rename:
        traffic.rename(columns=col_rename, inplace=True)
        print(f"  Normalized {len(col_rename)} column names")

    # Build Timestamp from Yr, M, D, HH fields
    if "Timestamp" not in traffic.columns:
        if all(c in traffic.columns for c in ["Yr", "M", "D", "HH"]):
            traffic["Timestamp"] = pd.to_datetime(
                traffic[["Yr", "M", "D"]].assign(H=traffic["HH"]).rename(
                    columns={"Yr": "year", "M": "month", "D": "day", "H": "hour"}
                )
            )
        else:
            print("❌ Cannot construct timestamp from traffic data columns")
            print(f"   Available columns: {list(traffic.columns)}")
            return pd.DataFrame()

    traffic["Timestamp"] = pd.to_datetime(traffic["Timestamp"])

    # Standardize weather timestamps
    weather["date"] = pd.to_datetime(weather["date"])

    # Create merge keys
    traffic["merge_hour"] = traffic["Timestamp"].dt.floor("h")

    # Map traffic "Boro" to borough name
    boro_map = {
        "Manhattan": "Manhattan", "Mn": "Manhattan",
        "Brooklyn": "Brooklyn", "Bk": "Brooklyn",
        "Queens": "Queens", "Qn": "Queens",
        "Bronx": "Bronx", "Bx": "Bronx",
        "Staten Island": "Staten Island", "SI": "Staten Island",
    }
    if "Boro" in traffic.columns:
        traffic["borough"] = traffic["Boro"].map(boro_map).fillna(traffic["Boro"])
    elif "Borough" in traffic.columns:
        traffic["borough"] = traffic["Borough"]

    weather["merge_hour"] = weather["date"].dt.floor("h")

    merged = traffic.merge(
        weather,
        on=["merge_hour", "borough"],
        how="inner",
        suffixes=("", "_weather"),
    )
    merged.drop(columns=["merge_hour"], inplace=True)

    # Ensure Vol column exists
    if "Vol" not in merged.columns:
        for candidate in ["vol", "Volume", "volume"]:
            if candidate in merged.columns:
                merged.rename(columns={candidate: "Vol"}, inplace=True)
                break

    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"✅ Merged: {len(merged):,} rows → {out} ({size_mb:.1f} MB)")

    return merged


# ── Main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and merge NYC traffic + weather data")
    parser.add_argument("--traffic-only", action="store_true", help="Only download traffic data")
    parser.add_argument("--weather-only", action="store_true", help="Only download weather data")
    parser.add_argument("--merge-only", action="store_true", help="Only merge existing files")
    parser.add_argument("--start-year", type=int, default=2014, help="Start year (default: 2014)")
    parser.add_argument("--end-year", type=int, default=2025, help="End year (default: 2025)")
    parser.add_argument("--weather-end", default="2025-03-18", help="Weather end date")
    args = parser.parse_args()

    start_date = f"{args.start_year}-01-01"

    if args.merge_only:
        merge_traffic_weather()
    elif args.traffic_only:
        download_traffic_data(args.start_year, args.end_year)
    elif args.weather_only:
        download_all_weather(start_date, args.weather_end)
    else:
        # Download both and merge
        download_traffic_data(args.start_year, args.end_year)
        download_all_weather(start_date, args.weather_end)
        merge_traffic_weather()

    print("\n🎉 Done!")
