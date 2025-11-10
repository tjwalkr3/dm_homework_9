import pandas as pd
import numpy as np
from scipy import stats

class Weather:
    def __init__(self, filepath):
        df = pd.read_csv(filepath, skiprows=[0, 2, 3])
        
        columns_to_keep = [
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
        
        self.weather_df = df[columns_to_keep].copy()
        self.clean_data()
    
    def clean_data(self):
        # Drop rows with missing values
        self.weather_df = self.weather_df.dropna()
        
        # Filter temperature range
        self.weather_df = self.weather_df[(self.weather_df['AirTF_Avg'] <= 200) & (self.weather_df['AirTF_Avg'] >= -20)]
        
        # Remove outliers using Z-score (threshold=3)
        numerical_cols = self.weather_df.select_dtypes(include=np.number).columns
        
        if not numerical_cols.empty:
            z_scores = np.abs(stats.zscore(self.weather_df[numerical_cols]))
            self.weather_df = self.weather_df[(z_scores < 3).all(axis=1)]
        
        self.weather_df.reset_index(drop=True, inplace=True)
    
    def linear_regression(self, x_cols, y_col):
        if isinstance(x_cols, str):
            x_cols = [x_cols]
        
        X = np.column_stack([self.weather_df[col].values for col in x_cols])
        y = self.weather_df[y_col].values
        A = np.column_stack([X, np.ones(len(y))])
        
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        S_inv = np.diag(1.0 / S)
        A_dagger = Vt.T @ S_inv @ U.T
        coeffs = A_dagger @ y
        
        return coeffs[:-1], coeffs[-1]
    
    def correlation(self, x_col, y_col):
        x = self.weather_df[x_col].values
        y = self.weather_df[y_col].values
        
        n = len(x)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        s_x = np.std(x, ddof=1)
        s_y = np.std(y, ddof=1)
        
        r = (1 / (n - 1)) * np.sum(((x - x_mean) / s_x) * ((y - y_mean) / s_y))
        
        return r

if __name__ == "__main__":
    weather = Weather("Snow Weather_Daily.csv")

    print("Linear Regression:")
    params, intercept = weather.linear_regression("AirTF_Avg", "RH_Avg")
    
    print(f"Bias: {intercept}, Parameter: {params[0]}")
    
    x = weather.weather_df["AirTF_Avg"].values
    y_actual = weather.weather_df["RH_Avg"].values
    y_predicted = params[0] * x + intercept
    print(f"RMSE: {np.sqrt(np.mean((y_actual - y_predicted) ** 2))}")
    print(f"Correlation: {weather.correlation("AirTF_Avg", "RH_Avg")}")

    all_predictors = ["AirTF_Avg", "AirTF_Max", "AirTF_Min", "WindGust", "AveWindSp", "WindDir", "BP_inHg_Avg", "Rain_Tot", "TdC_Min", "TdC_Max"]
    print("\nMultilinear Regression:")
    params_all, intercept_all = weather.linear_regression(all_predictors, "RH_Avg")
    
    print(f"Bias: {intercept_all}")
    for i, col in enumerate(all_predictors):
        print(f"{col}: {params_all[i]}")
    
    X_all = np.column_stack([weather.weather_df[col].values for col in all_predictors])
    y_actual_multi = weather.weather_df["RH_Avg"].values
    y_predicted_all = X_all @ params_all + intercept_all
    print(f"RMSE: {np.sqrt(np.mean((y_actual_multi - y_predicted_all) ** 2))}")
    
    print("\nCorrelations:")
    print(f"AirTF_Avg & RH_Avg -> {weather.correlation("AirTF_Avg", "RH_Avg")}")
    print(f"AirTF_Max & RH_Avg -> {weather.correlation("AirTF_Max", "RH_Avg")}")
    print(f"AirTF_Min & RH_Avg -> {weather.correlation("AirTF_Min", "RH_Avg")}")
    print(f"WindGust & RH_Avg -> {weather.correlation("WindGust", "RH_Avg")}")
    print(f"AveWindSp & RH_Avg -> {weather.correlation("AveWindSp", "RH_Avg")}")
    print(f"WindDir & RH_Avg -> {weather.correlation("WindDir", "RH_Avg")}")
    print(f"BP_inHg_Avg & RH_Avg -> {weather.correlation("BP_inHg_Avg", "RH_Avg")}")
    print(f"Rain_Tot & RH_Avg -> {weather.correlation("Rain_Tot", "RH_Avg")}")
    print(f"TdC_Min & RH_Avg -> {weather.correlation("TdC_Min", "RH_Avg")}")
    print(f"TdC_Max & RH_Avg -> {weather.correlation("TdC_Max", "RH_Avg")}")

    print("\nPrediction:")
    new_data = np.array([70, 77, 58, 21.5, 9, 180, 29.84, 0, 38, 46])
    prediction = new_data @ params_all + intercept_all
    print(f"Predicted Relative Humidity: {prediction}")

