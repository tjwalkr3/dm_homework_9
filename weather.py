import pandas as pd

def load_weather_data(filepath):
    weather_df = pd.read_csv(filepath, skiprows=[0, 2, 3])
    
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
    
    weather_df = weather_df[columns_to_keep]
    
    return weather_df

if __name__ == "__main__":
    weather_df = load_weather_data("Snow Weather_Daily.csv")
    print(weather_df.head())
    print(f"\nShape: {weather_df.shape}")
