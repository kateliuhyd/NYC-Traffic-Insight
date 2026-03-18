import os
import pandas as pd

def load_weather():
    weather_files = ['RawDataFiles/weather_data_vm1.csv', 'RawDataFiles/weather_data_vm2.csv',
'RawDataFiles/weather_data_vm3.csv', 'RawDataFiles/weather_data_vm4.csv',
'RawDataFiles/weather_data_vm5.csv','RawDataFiles/weather_data_vm6.csv', 'RawDataFiles/weather_data_vm7.csv']
    return [pd.read_csv(file) for file in weather_files] #reads each file in weather files

def clean_weather(finalWeatherframe):
    #found null @ columns uv_index(915360), freezing_level_height(915360), visibility(915360), snow_depth(91536)
    finalWeatherframe = finalWeatherframe.drop(['uv_index', 'freezing_level_height', 'visibility'], axis='columns')
#cleaning formatting of the date-time
    finalWeatherframe['date'] = pd.to_datetime(finalWeatherframe['date'])
    finalWeatherframe['date'] = finalWeatherframe['date'].dt.strftime('%Y-%m-%d %H:%M')
#Now in format (year-month-day hour:minute)
    finalWeatherframe['borough'] = finalWeatherframe['borough'].astype('category')
    return finalWeatherframe

def main():
    print('Loading weather data...')
    wFrames = load_weather()
    finalWeatherframe = pd.concat(wFrames, ignore_index=True)
    print("Weather loaded")
    print('Cleaning weather data...')
    finalWeatherframe = clean_weather(finalWeatherframe)
    print("Weather cleaned")
    if os.path.exists('RawDataFiles/finalWeather.csv'):
        print('file already exists')
        return
    else:
        print('Saving weather data...')
        finalWeatherframe.to_csv('RawDataFiles/finalWeather.csv', index=False)
        print("Weather data saved")
if __name__ == "__main__":
    main()