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
        self.weather_df = self.weather_df.dropna()
        self.weather_df = self.weather_df[
            (self.weather_df['AirTF_Avg'] <= 200) & 
            (self.weather_df['AirTF_Avg'] >= -20)
        ]
        self.weather_df.reset_index(drop=True, inplace=True)
    
    def linear_regression(self, x_cols, y_col):
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
    
    def calculate_rmse(self, x_cols, y_col, params, intercept):
        X = np.column_stack([self.weather_df[col].values for col in x_cols])
        y_actual = self.weather_df[y_col].values
        y_predicted = X @ params + intercept
        
        return np.sqrt(np.mean((y_actual - y_predicted) ** 2))

if __name__ == "__main__":
    weather = Weather("Snow Weather_Daily.csv")
    
    all_predictors = [
        "AirTF_Avg", "AirTF_Max", "AirTF_Min", "WindGust", 
        "AveWindSp", "WindDir", "BP_inHg_Avg", "Rain_Tot", 
        "TdC_Min", "TdC_Max"
    ]
    
    # Single linear regression
    print("\n### Single linear Regression ###")
    params, intercept = weather.linear_regression(["AirTF_Avg"], "RH_Avg")
    print(f"Bias: {intercept}, Parameter: {params[0]}")
    
    rmse = weather.calculate_rmse(["AirTF_Avg"], "RH_Avg", params, intercept)
    print(f"RMSE: {rmse}")
    
    corr = weather.correlation("AirTF_Avg", "RH_Avg")
    print(f"Correlation: {corr}")
    
    print(f"\nRH_Avg prediction at Temperature 63.2: {63.2 * params[0] + intercept}")
    
    # Multiple linear regression
    print("\n### Multiple linear Regression ###")
    params_all, intercept_all = weather.linear_regression(all_predictors, "RH_Avg")
    
    print(f"Bias: {intercept_all}")
    [print(f"{col}: {params_all[i]}") for i, col in enumerate(all_predictors)]
    
    rmse_multi = weather.calculate_rmse(all_predictors, "RH_Avg", params_all, intercept_all)
    print(f"RMSE: {rmse_multi}")
    
    print("\nAll correlations:")
    correlations = {predictor + " & RH_Avg": weather.correlation(predictor, "RH_Avg") for predictor in all_predictors}
    [print(f"{key} = {correlations[key]}") for key in correlations]

    all_predictors = np.array([65, 77, 52, 23.2, 10, 180, 29.94, 0, 31, 42])
    multi_prediction = all_predictors @ params_all + intercept_all
    print(f"\nRH_Avg Prediction w/ all predictors: {multi_prediction}")
