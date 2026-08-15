import pandas as pd
from sklearn.preprocessing import RobustScaler

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic data cleaning: drop missing values and scale monetary/time features.
    """
    # Simply drop rows with missing values to keep it clean and robust
    df = df.dropna()
    
    # Scale Time and Amount using RobustScaler to minimize the impact of extreme outliers
    scaler = RobustScaler()
    if 'Amount' in df.columns and 'Time' in df.columns:
        df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])
        
    return df
