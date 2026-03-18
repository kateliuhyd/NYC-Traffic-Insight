import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


def load_raw_data():
    traffic = pd.read_csv("RawDataFiles/TrafficData.csv")
    weather = pd.read_csv("RawDataFiles/finalWeather.csv")
    return traffic, weather

def clean_traffic(traffic):
    traffic.drop_duplicates(inplace=True)
    traffic.drop(columns= ['MM'], inplace=True)
    traffic["WktGeom"] = traffic["WktGeom"].str.strip("POINT ()")
    traffic.drop(columns= ['SegmentID'], inplace=True)
    traffic[["Latitude", "Longitude"]] = traffic['WktGeom'].str.split(' ', n=2, expand=True)
    traffic = traffic.drop(['WktGeom'], axis=1, inplace=True)
    traffic.fillna('', inplace=True)
    return traffic

def merging(traffic, weather):
#Prep data for merge
    weather = weather.rename(columns={'borough':'Boro'})
    traffic['Boro'] = traffic['Boro'].astype('category')
    weather['Boro'] = weather['Boro'].astype('category')

    weather['date'] = pd.to_datetime(weather['date'])
    weather['Yr'] = weather['date'].dt.year
    weather['M'] = weather['date'].dt.month
    weather['D'] = weather['date'].dt.day
    weather['HH'] = weather['date'].dt.hour
    merged_traffic_data = pd.merge(traffic, weather, on=['Yr', 'M', 'D', 'HH', 'Boro'])
    return merged_traffic_data

def main():
    print("Loading raw traffic data")
    traffic, weather = load_raw_data()
    print("Data loaded")
    print("Cleaning traffic data")
    traffic = clean_traffic(traffic)
    print("Data cleaned")
    print("Merging traffic data")
    merged_traffic_data = merging(traffic, weather)
    print("Data merged")

    if os.path.exists('merged_traffic_data.csv'):
        print('file already exists')
        return
    else:
        merged_traffic_data.to_csv("RawDataFiles/merged_traffic_data.csv", index=False)
        print("Merged traffic + Weather data saved")

if __name__ == "__main__":
    main()