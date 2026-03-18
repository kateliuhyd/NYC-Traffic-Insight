import os
import pandas as pd
import numpy as np
import holidays

def load_traffic():
    return pd.read_csv("RawDataFiles/merged_traffic_data.csv")

def make_datetime(df):
    df['date'] = pd.to_datetime(df['date'])
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["dayofweek"] = df["date"].dt.dayofweek
    return df

def seasons(df):
    #Weekend and Season Classification
    df["is_winter"] = df["month"].isin([12,1,2,3]).astype(int)
    df["is_summer"] = df["month"].isin([6,7,8,9]).astype(int)
    df['is_spring'] = df["month"].isin([4, 5]).astype(int)
    df['is_autumn'] = df["month"].isin([10, 11]).astype(int)
    df['isWeekend'] = np.where(df['date'].dt.weekday >= 5, True, False)
    return df

#Holiday Classification
def ny_holidays(df):
    us_ny_holidays = holidays.country_holidays('US', subdiv='NY')
    df["is_holiday"] = df['date'].apply(lambda x: x in us_ny_holidays)
    return df

def make_interactions(df):
    # Interaction Features
    df["precip_x_isWinter"] = df["precipitation"] * df["is_winter"]
    df["temp_x_isSummer"] = df["temperature_2m"] * df["is_summer"]
    df["rain_is_winter"] = df["snowfall"] * df["is_winter"]
    df["cloud_x_hour"] = df["cloud_cover_low"] * df["HH"]
    df["rain_x_is_winter"] = df["rain"] * df["is_winter"]
    df["precip_x_is_holiday"] = df["precipitation"] * df["is_holiday"]
    df["is_holiday_x_HH"] = df["is_holiday"] * df["HH"]
    return df

def main():
    print('loading data')
    df = load_traffic()

    print("feature creation")
    df = make_datetime(df)
    df = seasons(df)
    df = ny_holidays(df)

    df_encoded = pd.get_dummies(df[["Boro", "Direction"]], prefix=["borough", "direction"], drop_first=True)
    df = pd.concat([df, df_encoded], axis=1)
    df = make_interactions(df)

    #Drop useless rows
    df.drop(columns=['Latitude', 'Longitude', 'Boro', 'Yr', 'M', 'D', 'HH'], inplace=True, errors='ignore')
    df.rename(columns={'Unnamed: 0': 'ID'}, inplace=True, errors='ignore')

    if os.path.exists('RawDataFiles/engineered_traffic_data.csv'):
        print('file already exists')
        return
    else:
        df.to_csv("RawDataFiles/engineered_traffic_data.csv", index=False)
        print(df.shape)
        print('Data Saved')

if __name__ == '__main__':
    main()