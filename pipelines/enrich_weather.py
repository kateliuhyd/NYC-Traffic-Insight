from decimal import Decimal, ROUND_05UP
import openmeteo_requests  # Open-Meteo's Python client for weather API
import pandas as pd
import requests_cache  # Used to cache API requests and avoid duplicate calls
from retry_requests import retry  # Automatically retries failed HTTP requests
import os
import time
from openmeteo_requests.Client import OpenMeteoRequestsError


# === Setup input/output paths ===
# Get the base directory of the project (go up one level from /scripts)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
output_dir = os.path.join(BASE_DIR, "data", "processed", "output")
os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
cache_path = os.path.join(BASE_DIR, "data", "cache", "weather_cache")
os.makedirs(os.path.dirname(cache_path), exist_ok=True)
progress_log_path = os.path.join(output_dir, "weather_processing_progress.txt")

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession(cache_path, expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

request_count_hour = 0
request_count_day = 0
hour_start = time.time()
day_start = time.time()
processed_count = 0
processed_features = 0

url = "https://archive-api.open-meteo.com/v1/archive"

all_weather_data = []

# these we're picked by a script that draws a bounding box around a borough and picks the
# furthest points from one another
borough_points = [
    ("Brooklyn", -73.9527531905247, 40.73579422088676),
    ("Brooklyn", -73.88919401240395, 40.57963421654645),
    ("Queens", -73.78224969710911, 40.78902023475063),
    ("Queens", -73.88377292466673, 40.5672536844456),
    ("Manhattan", -74.01176217456705, 40.70138635779195),
    ("Manhattan", -73.90948844071255, 40.87445551939913),
    ("Bronx", -73.80471827028069, 40.886171212869385),
    ("Bronx", -73.93103571415577, 40.80824652852295),
    ("Staten Island", -74.07266409737403, 40.64360714096245),
    ("Staten Island", -74.24994651503094, 40.498253614198454),
]

for borough, lon, lat in borough_points:
    # Arounded to 2 decimal places for better precision in API calls
    long = Decimal(lon).quantize(Decimal('0.01'), rounding=ROUND_05UP)
    lati = Decimal(lat).quantize(Decimal('0.01'), rounding=ROUND_05UP)

    # === Construct API URL and parameters ===
    params = {
        "latitude": lati,
        "longitude": long,
        "start_date": "2022-12-08",     # Static start for now — can be changed to dynamic if needed # 2016-01-01
        "end_date": "2024-06-10",       # Static end date
        "hourly": ["temperature_2m", "precipitation", "cloud_cover_low", "snow_depth", "visibility", "weather_code", "freezing_level_height", "rain", "showers", "snowfall", "uv_index"],
        "wind_speed_unit": "mph",
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "timezone": "America/New_York"
    }

    while True:
        # Check elapsed time since hour/day start
        now = time.time()
        elapsed_hour = now - hour_start
        elapsed_day = now - day_start

        # Reset counters if needed
        if elapsed_hour >= 3600:
            request_count_hour = 0
            hour_start = now
        if elapsed_day >= 86400:
            request_count_day = 0
            day_start = now

        # Throttle proactively
        if request_count_hour >= 4500:
            sleep_time = 3600 - elapsed_hour
            print(f"[Rate limit] Hourly threshold near. Sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
            continue
        if request_count_day >= 9500:
            sleep_time = 86400 - elapsed_day
            print(f"[Rate limit] Daily threshold near. Sleeping {sleep_time / 3600:.2f} hours")
            time.sleep(sleep_time)
            continue

        try:
            responses = openmeteo.weather_api(url, params=params)
            request_count_hour += 1
            request_count_day += 1
            time.sleep(0.8)
            break
        except OpenMeteoRequestsError as e:
            error_str = str(e)
            with open(progress_log_path, "a") as log:
                if "Minutely" in error_str:
                    log.write("Minutely rate limit hit. Sleeping 10 seconds...\n")
                    print("Minutely rate limit hit. Sleeping 10 seconds...")
                    time.sleep(10)
                elif "Hourly" in error_str:
                    log.write("Hourly rate limit hit. Sleeping 1 hour...\n")
                    print("Hourly rate limit hit. Sleeping 1 hour...")
                    time.sleep(3600)
                elif "Daily" in error_str:
                    log.write("Daily rate limit hit. Sleeping 24 hours...\n")
                    print("Daily rate limit hit. Sleeping 24 hours...")
                    time.sleep(86400)
                elif "Monthly" in error_str:
                    log.write("Monthly rate limit hit. Exiting script...\n")
                    print("Monthly rate limit hit. Exiting script.")
                    raise SystemExit(1)
                else:
                    raise e

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()}{response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process hourly data. The order of variables needs to be the same as requested.
    # The variables are accessed based on their positions on the list
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_precipitation = hourly.Variables(1).ValuesAsNumpy()
    hourly_cloud_cover_low = hourly.Variables(2).ValuesAsNumpy()
    hourly_snow_depth = hourly.Variables(3).ValuesAsNumpy()
    hourly_visibility = hourly.Variables(4).ValuesAsNumpy()
    hourly_weather_code = hourly.Variables(5).ValuesAsNumpy()
    hourly_freezing_level_height = hourly.Variables(6).ValuesAsNumpy()
    hourly_rain = hourly.Variables(7).ValuesAsNumpy()
    hourly_showers = hourly.Variables(8).ValuesAsNumpy()
    hourly_snowfall = hourly.Variables(9).ValuesAsNumpy()
    hourly_uv_index = hourly.Variables(10).ValuesAsNumpy()

    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "latitude": float(lat),
        "longitude": float(lon),
        "borough": [borough] * hourly.Variables(0).ValuesAsNumpy().size,
        "temperature_2m": hourly_temperature_2m, # Extreme heat or cold can affect vehicle performance and road safety.
        "precipitation": hourly_precipitation,
        "cloud_cover_low": hourly_cloud_cover_low,
        "snow_depth": hourly_snow_depth, #  Impacts vehicle traction and congestion
        "visibility": hourly_visibility, # Critical for driving safety, especially in fog, rain, or snow
        "weather_code": hourly_weather_code, # Categorical description of weather (e.g., clear, fog, rain, storm, etc.)
        "freezing_level_height": hourly_freezing_level_height, # Can help model icy road conditions in winter
        "rain": hourly_rain,
        "showers": hourly_showers,
        "snowfall": hourly_snowfall,
        "uv_index": hourly_uv_index
    }

    hourly_dataframe = pd.DataFrame(data=hourly_data)
    print(hourly_dataframe)
    all_weather_data.append(hourly_dataframe)

    # Save cumulative progress after each successful response
    partial_csv_path = os.path.join(output_dir, "weather_data_partial_vm7.csv")
    pd.concat(all_weather_data, ignore_index=True).to_csv(partial_csv_path, index=False)

    processed_count += 1
    with open(progress_log_path, "a") as log:
        log.write(f"Processed coords/times: {processed_count}/{len(borough_points)}\n")

# Combine all location DataFrames into one
weather_df = pd.concat(all_weather_data, ignore_index=True)
# Save to CSV
weather_df.to_csv(os.path.join(output_dir, "weather_data_vm7.csv"), index=False)
