import argparse
import json
from sklearn.model_selection import train_test_split

from src.data_ops import bootstrap_batches, load_training_dataframe
from src.preprocess import clean_data
from src.features import engineer_features, balance_data
from src.train import train_models
from src.evaluate import evaluate_models, print_evaluation, select_best_model
from src.mlops import log_run, maybe_promote
from src.settings import load_settings


def parse_args():
    parser = argparse.ArgumentParser(description="Simple local-first MLOps pipeline")
    parser.add_argument("--config", default="config/pipeline.json")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--force-bootstrap-batches", action="store_true")
    return parser.parse_args()

def run_pipeline(config_path="config/pipeline.json", max_batches=None, force_bootstrap_batches=False):
    settings = load_settings(config_path)
    if max_batches is not None:
        settings["training"]["max_batches"] = max_batches

    print("Starting MLOps pipeline...")
    bootstrap_batches(settings, force=force_bootstrap_batches)

    print("\n[1/6] Loading batch data...")
    df, batch_files = load_training_dataframe(settings, max_batches=settings["training"].get("max_batches"))

    print("\n[2/6] Cleaning data...")
    df = clean_data(df)

    print("\n[3/6] Engineering features...")
    df = engineer_features(df)

    y = df['Class']
    X = df.drop(columns=['Class', 'Time', 'Time_raw'], errors='ignore')

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=settings["training"]["test_size"],
        random_state=settings["training"]["random_state"],
        stratify=y
    )

    print("\n[4/6] Balancing training data...")
    X_train_res, y_train_res = balance_data(
        X_train,
        y_train,
        enabled=settings["training"].get("balance_with_smote", True)
    )

    print("\n[5/6] Training models...")
    models = train_models(X_train_res, y_train_res, random_state=settings["training"]["random_state"])

    print("\n[6/6] Evaluating and promoting the best model...")
    results = evaluate_models(models, X_test, y_test)
    print_evaluation(results)

    best_model_name, best_score = select_best_model(results, settings["selection"]["primary_metric"])
    best_model = models[best_model_name]

    tracking = log_run(
        settings,
        best_model_name,
        best_model,
        results[best_model_name],
        {
            "batch_count": len(batch_files),
            "test_size": settings["training"]["test_size"],
            "random_state": settings["training"]["random_state"]
        }
    )

    bundle = {
        "model_name": best_model_name,
        "model": best_model,
        "feature_columns": list(X_train.columns),
        "metrics": results[best_model_name],
        "selection_metric": settings["selection"]["primary_metric"],
        "selection_score": best_score
    }

    promotion_result = maybe_promote(
        settings,
        bundle,
        results,
        [batch_file.name for batch_file in batch_files],
        tracking
    )

    summary = {
        "best_model_name": best_model_name,
        "best_score": best_score,
        "promotion_result": promotion_result
    }
    print(json.dumps(summary, indent=2))
    return summary

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        config_path=args.config,
        max_batches=args.max_batches,
        force_bootstrap_batches=args.force_bootstrap_batches
    )
