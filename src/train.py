import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

def train_models(X_train, y_train):
    """
    Train a baseline model and powerful ensembles to compare them.
    Includes class weights for handling the inherent fraud imbalance natively.
    """
    # Since we used SMOTE to balance the data 50/50, we MUST NOT use class weights!
    # Double-weighting causes the model to become overly-paranoid, leading to low Precision.
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, max_depth=8, random_state=42, eval_metric='logloss')
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
    return trained_models

def save_model(model, filepath="models/best_model.pkl"):
    """
    Save the best performing model to disk for serving.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model successfully saved to {filepath}")
