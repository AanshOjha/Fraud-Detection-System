import pandas as pd
from imblearn.over_sampling import SMOTE

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features like rolling averages and anomaly flags to improve model performance.
    """
    # Ensure dataset is sorted by time for accurate sequential features
    if 'Time' in df.columns:
        df = df.sort_values('Time')
        
        # 1. Since we don't have Customer IDs, time gaps and rolling averages between arbitrary users add NOISE!
        # Instead, extract the Hour of the day from the 'Time' column (useful for catching late-night fraud)
        df['Hour'] = (df['Time'] // 3600) % 24
        
        # 2. Anomaly feature: Flag statistically large transactions
        if 'Amount' in df.columns:
            # Over $200 serves as a simple high-amount flag indicator
            df['amount_surge'] = (df['Amount'] > 200).astype(int)

    return df

def balance_data(X, y):
    """
    Handle Critical Class Imbalance using SMOTE (Synthetic Minority Over-sampling Technique).
    Fraud represents ~0.1%, so this synthetically creates examples of the minority class.
    """
    print("Balancing training data with SMOTE...")
    smote = SMOTE(sampling_strategy='minority', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print(f"Original shape: {X.shape}, Resampled shape: {X_resampled.shape}")
    
    return X_resampled, y_resampled
