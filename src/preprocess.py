import pandas as pd
from sklearn.preprocessing import RobustScaler

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic data cleaning: drop missing values and scale monetary/time features.
    """
    df = df.dropna().copy()

    if 'Time' in df.columns:
        df['Time_raw'] = df['Time']

    scaler = RobustScaler()
    scale_columns = [column for column in ['Time', 'Amount'] if column in df.columns]
    if scale_columns:
        df[scale_columns] = scaler.fit_transform(df[scale_columns])

    return df
