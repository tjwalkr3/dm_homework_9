import pandas as pd

class Weather:
    def __init__(self, filepath):
        df = pd.read_csv(filepath, skiprows=[0, 2, 3])
        
        columns_to_keep = [
            'TIMESTAMP',
            'AirTF_Avg',
            'AirTF_Max',
            'AirTF_Min',
            'RH_Avg',
            'WindGust',
            'AveWindSp',
            'WindDir',
            'BP_inHg_Avg',
            'Rain_Tot',
            'TdC_Min',
            'TdC_Max'
        ]
        
        self.weather_df = df[columns_to_keep]

if __name__ == "__main__":
    weather = Weather("Snow Weather_Daily.csv")
    print(weather.weather_df.head())
    print(f"\nShape: {weather.weather_df.shape}")
