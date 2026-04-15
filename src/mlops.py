from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib

from src.settings import resolve_path

try:
    import mlflow
    from mlflow import MlflowClient
except ImportError:  # pragma: no cover
    mlflow = None
    MlflowClient = None


def mlflow_is_available(settings: dict) -> bool:
    return bool(settings["mlflow"].get("enabled", True) and mlflow is not None)


def log_run(settings: dict, model_name: str, model, metrics: dict, params: dict) -> dict:
    if not mlflow_is_available(settings):
        return {"enabled": False}

    mlflow.set_tracking_uri(settings["mlflow"]["tracking_uri"])
    mlflow.set_experiment(settings["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_param("model_name", model_name)
        mlflow.log_params(params)

        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(metric_name.lower().replace("-", "_"), float(metric_value))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=settings["mlflow"]["registered_model_name"],
        )

        return {"enabled": True, "run_id": run.info.run_id}


def save_candidate(settings: dict, bundle: dict, all_results: dict) -> None:
    candidate_model_path = resolve_path(settings["deployment"]["candidate_model_path"])
    candidate_metrics_path = resolve_path(settings["deployment"]["candidate_metrics_path"])

    candidate_model_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(bundle, candidate_model_path)
    with candidate_metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2)


def maybe_promote(settings: dict, bundle: dict, all_results: dict, batch_names: list[str], tracking: dict) -> dict:
    save_candidate(settings, bundle, all_results)

    production_metadata_path = resolve_path(settings["deployment"]["metadata_path"])
    production_model_path = resolve_path(settings["deployment"]["production_model_path"])
    production_metrics_path = resolve_path(settings["deployment"]["production_metrics_path"])

    metric_name = settings["promotion"]["metric"]
    candidate_score = float(bundle["metrics"][metric_name])
    candidate_recall = float(bundle["metrics"]["Recall"])
    minimum_recall = float(settings["promotion"].get("minimum_recall", 0.0))

    if candidate_recall < minimum_recall:
        return {"promoted": False, "reason": "candidate recall is below the minimum threshold"}

    current_score = None
    if production_metadata_path.exists():
        with production_metadata_path.open("r", encoding="utf-8") as handle:
            current_metadata = json.load(handle)
        current_score = float(current_metadata["metrics"][metric_name])

    minimum_improvement = float(settings["promotion"].get("minimum_improvement", 0.0))
    if current_score is not None and candidate_score < current_score + minimum_improvement:
        return {"promoted": False, "reason": "candidate did not beat the current production model"}

    production_model_path.parent.mkdir(parents=True, exist_ok=True)
    production_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    production_metadata_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(bundle, production_model_path)
    with production_metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2)

    metadata = {
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "model_name": bundle["model_name"],
        "metrics": bundle["metrics"],
        "feature_columns": bundle["feature_columns"],
        "batches_used": batch_names,
        "tracking": tracking,
    }
    with production_metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    if mlflow_is_available(settings) and tracking.get("enabled") and MlflowClient is not None:
        client = MlflowClient()
        try:
            latest_versions = client.get_latest_versions(settings["mlflow"]["registered_model_name"])
            for version_info in latest_versions:
                if version_info.run_id == tracking["run_id"]:
                    client.transition_model_version_stage(
                        name=settings["mlflow"]["registered_model_name"],
                        version=version_info.version,
                        stage="Production",
                        archive_existing_versions=True,
                    )
                    break
        except Exception:
            pass

    return {"promoted": True, "reason": "candidate promoted to production"}
