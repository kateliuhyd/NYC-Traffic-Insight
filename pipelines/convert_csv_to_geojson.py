import pandas as pd
import json
from pyproj import Transformer
# this is a script to convert the traffic csv file to geojson point format for use in the map.
# will prolly get rid of this since it's replaced by convert_point_to_line_geojson.py
# Load CSV
csv_path = r'C:\Users\nokon\Downloads\Automated_Traffic_Volume_Counts_20250607.csv'
OUTPUT_PATH = r'C:\Users\nokon\Downloads\traffic_data.geojson'
df = pd.read_csv(csv_path, low_memory=False)

# Convert Vol column to numeric (handle mixed types)
df["Vol"] = pd.to_numeric(df["Vol"], errors="coerce")

# Initialize transformer
transformer = Transformer.from_crs("EPSG:2263", "EPSG:4326", always_xy=True)

# Convert WKT to lat/lon
def extract_latlon(wkt):
    coords = wkt.replace("POINT (", "").replace(")", "").split()
    x, y = float(coords[0]), float(coords[1])
    lon, lat = transformer.transform(x, y)
    return lat, lon

# Generate GeoJSON features
features = []
for _, row in df.iterrows():
    try:
        lat, lon = extract_latlon(row["WktGeom"])
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat]
            },
            "properties": {
                "RequestID": row["RequestID"],
                "Volume": row["Vol"],
                "Timestamp": f"{row['Yr']}-{int(row['M']):02d}-{int(row['D']):02d}T{int(row['HH']):02d}:{int(row['MM']):02d}:00",
                "Street": row["street"],
                "From": row["fromSt"],
                "To": row["toSt"],
                "Direction": row["Direction"],
                "Borough": row["Boro"]
            }
        }
        features.append(feature)
        print(f"Processed RequestID {row['RequestID']}")
    except Exception as e:
        print(f"Error processing row {row['RequestID']}: {e}")

# Create GeoJSON structure
geojson_data = {
    "type": "FeatureCollection",
    "features": features
}

# Save to file
with open(OUTPUT_PATH, "w") as f:
    json.dump(geojson_data, f)

print("GeoJSON file saved as traffic_data.geojson")
