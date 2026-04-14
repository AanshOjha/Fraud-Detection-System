from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def evaluate_models(models, X_test, y_test):
    """
    Evaluate trained models using the best metrics for fraud detection.
    Accuracy is omitted because in a 99% legit dataset, predicting 'safe' always gives 99% accuracy!
    """
    results = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        # We need probability predictions to calculate ROC-AUC
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_prob)
        else:
            roc_auc = "N/A"
            
        metrics = {
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-Score": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc
        }
        results[name] = metrics
        
    return results

def print_evaluation(results):
    """
    Print the evaluation metrics cleanly to demonstrate the Precision-Recall tradeoff.
    High recall catches fraud, High precision prevents false alarms.
    """
    print("\n" + "="*35)
    print(" 📊 MODEL COMPARISON DASHBOARD")
    print("="*35)
    
    for name, metrics in results.items():
        print(f"\n{name} Performance:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  > {metric}:  {value:.4f}")
            else:
                print(f"  > {metric}:  {value}")
        print("-" * 35)
