import json
import os
import osmnx as ox
from pathlib import Path

# Config
# I should/will later change this so that if the path doesnt exist, create it
# If you're looking at this, you have to create these dirs or it wont work lol
# Fixed
BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "data" / "processed" / "input"
INPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = INPUT_DIR / "traffic_data_small.geojson"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "output" / "traffic_data_osm_lines_clean.geojson"
COORDINATE_MAP_FILE = BASE_DIR / "data" / "processed" / "coordinate_map.json"

# maksure output directory exists
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# load the road network
print("Downloading NYC road network")
G = ox.graph_from_place("New York City, USA", network_type='drive')
print("NYC road network loaded.")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)
features = data["features"]
print(f"Loaded {len(features)} features.")

with open(COORDINATE_MAP_FILE, "r", encoding="utf-8") as f:
    raw_coordinate_map = json.load(f)
coordinate_map = {eval(k): v for k, v in raw_coordinate_map.items()}

output_features = []

for idx, feature in enumerate(features, 1):
    props = feature["properties"]
    street = props.get("Street", "")
    from_addr = props.get("From", "")
    to_addr = props.get("To", "")
    key = (street, from_addr, to_addr)

    # Get from/to points
    if key in coordinate_map:
        from_point = tuple(coordinate_map[key][0])
        to_point = tuple(coordinate_map[key][1])
    else:
        from_point = tuple(feature["geometry"]["coordinates"])
        to_point = from_point

    try:
        u, v, key_edge = ox.distance.nearest_edges(G, from_point[0], from_point[1])
        edge_data = G.get_edge_data(u, v, key_edge)
        if "geometry" in edge_data:
            edge_coords = [(pt[0], pt[1]) for pt in edge_data["geometry"].coords]
            coords_to_use = edge_coords
            print(f"{idx}/{len(features)}: Using edge geometry for {key}")
        else:
            coords_to_use = [from_point, to_point]
            print(f"{idx}/{len(features)}: No geometry on edge, using direct line for {key}")
    except Exception as e:
        coords_to_use = [from_point, to_point]
        print(f"{idx}/{len(features)}: Failed to get edge geometry for {key}. Using direct line. Error: {e}")

    new_feature = {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": coords_to_use
        },
        "properties": props
    }
    output_features.append(new_feature)

output_geojson = {
    "type": "FeatureCollection",
    "features": output_features
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output_geojson, f, indent=2)

print(f"Saved {len(output_features)} clean LineStrings to {OUTPUT_FILE}")
