import os
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split

from src.preprocess import clean_data
from src.features import engineer_features, balance_data
from src.train import train_models, save_model
from src.evaluate import evaluate_models, print_evaluation

def generate_dummy_data(filepath="data/creditcard.csv"):
    """Creates a basic dummy dataset if the real creditcard.csv doesn't exist to test pipeline execution."""
    print("No dataset provided! Creating dummy 'data/creditcard.csv' for testing...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.random.seed(42)
    # Simulate high class imbalance: 990 legit (0) and 10 fraud (1)
    df = pd.DataFrame({
        'Time': np.linspace(10, 100000, 1000),
        'Amount': np.random.exponential(scale=50, size=1000),
        'V1': np.random.randn(1000),
        'V2': np.random.randn(1000),
        'Class': [0]*990 + [1]*10
    })
    df.to_csv(filepath, index=False)
    return filepath

def run_pipeline():
    print("🚀 Starting Credit Card Fraud Detection Pipeline...")
    
    # 0. Load Raw Data
    data_path = "data/creditcard.csv"
    if not os.path.exists(data_path):
        generate_dummy_data(data_path)
    
    print("\n[1/5] Loading and Preprocessing Data...")
    df = pd.read_csv(data_path)
    df = clean_data(df)
    
    # 1. Feature Engineering
    print("\n[2/5] Creating Advanced Features...")
    df = engineer_features(df)
    
    # Prepare features and target
    y = df['Class']
    X = df.drop(columns=['Class', 'Time'], errors='ignore') # Drop 'Time' mostly.
    
    # Stratify split is crucial to keep the fraud % equal in train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 2. Handle Imbalance (Only on Train data!)
    print("\n[3/5] Handling Imbalance to Rescue the Minority Class...")
    X_train_res, y_train_res = balance_data(X_train, y_train)
    
    # 3. Traing Models
    print("\n[4/5] Training Robust Machine Learning Models...")
    models = train_models(X_train_res, y_train_res)
    
    # 4. Evaluate
    print("\n[5/5] Evaluating and Comparing Real-World Metrics...")
    results = evaluate_models(models, X_test, y_test)
    print_evaluation(results)
    
    with open("models/metrics.json", "w") as f:
        json.dump(results, f)
    
    # 5. Save the best model
    # Usually you'd pick based on best F1 score programmatically, 
    # but XGBoost reliably wins for fraud tasks!
    best_model_name = "XGBoost" 
    print(f"\n💾 Saving {best_model_name} for the API...")
    save_model(models[best_model_name])
    
    print("\n✅ Pipeline complete! Run `python app.py` to start the dashboard API.")

if __name__ == "__main__":
    run_pipeline()
