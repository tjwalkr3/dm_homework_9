import pandas as pd
import numpy as np

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
        
        self.weather_df = df[columns_to_keep].copy()
        self.clean_data()
    
    def clean_data(self):
        numeric_cols = self.weather_df.select_dtypes(include=[np.number]).columns
        self.weather_df[numeric_cols] = self.weather_df[numeric_cols].replace(['NAN', 'nan', 'NaN', ''], np.nan)
        self.weather_df.dropna(inplace=True)
        
        for col in numeric_cols:
            Q1 = self.weather_df[col].quantile(0.25)
            Q3 = self.weather_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.weather_df = self.weather_df[(self.weather_df[col] >= lower_bound) & (self.weather_df[col] <= upper_bound)]
        
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
        s_x = np.std(x)
        s_y = np.std(y)
        
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
    
    rmse = np.sqrt(np.mean((y_actual - y_predicted) ** 2))
    print(f"RMSE: {rmse}")

    r = weather.correlation("AirTF_Avg", "RH_Avg")
    print(f"Correlation coefficient: {r}")

    all_predictors = ["AirTF_Avg", "AirTF_Max", "AirTF_Min", "WindGust", "AveWindSp", "WindDir", "BP_inHg_Avg", "Rain_Tot", "TdC_Min", "TdC_Max"]
    print("\nMultilinear Regression:")
    params_all, intercept_all = weather.linear_regression(all_predictors, "RH_Avg")
    print(f"Bias: {intercept_all}")
    for i, col in enumerate(all_predictors):
        print(f"{col}: {params_all[i]}")
    
    X_all = np.column_stack([weather.weather_df[col].values for col in all_predictors])
    y_predicted_all = X_all @ params_all + intercept_all
    rmse_all = np.sqrt(np.mean((y_actual - y_predicted_all) ** 2))
    print(f"RMSE (Multilinear): {rmse_all}")

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

