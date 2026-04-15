import pandas as pd
from imblearn.over_sampling import SMOTE

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features like rolling averages and anomaly flags to improve model performance.
    """
    time_column = 'Time_raw' if 'Time_raw' in df.columns else 'Time'
    if time_column in df.columns:
        df = df.sort_values(time_column).copy()
        df['Hour'] = ((df[time_column] // 3600) % 24).astype(int)

    if 'Amount' in df.columns:
        df['amount_surge'] = (df['Amount'] > 200).astype(int)

    return df

def balance_data(X, y, enabled=True):
    """
    Handle Critical Class Imbalance using SMOTE (Synthetic Minority Over-sampling Technique).
    Fraud represents ~0.1%, so this synthetically creates examples of the minority class.
    """
    if not enabled:
        print("Skipping SMOTE balancing.")
        return X, y

    if y.nunique() < 2:
        print("Skipping SMOTE because only one class is present.")
        return X, y

    minority_count = int(y.value_counts().min())
    if minority_count < 2:
        print("Skipping SMOTE because the minority class is too small.")
        return X, y

    print("Balancing training data with SMOTE...")
    smote = SMOTE(sampling_strategy='minority', random_state=42, k_neighbors=min(5, minority_count - 1))
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print(f"Original shape: {X.shape}, Resampled shape: {X_resampled.shape}")
    
    return X_resampled, y_resampled
