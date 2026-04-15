import os
import joblib
import json
import time
import pandas as pd
import uvicorn
from fastapi import FastAPI, Form, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates

from src.settings import load_settings, resolve_path

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, REGISTRY, generate_latest
except ImportError:  # pragma: no cover
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"
    Counter = None
    Histogram = None
    REGISTRY = None
    generate_latest = None

app = FastAPI(title="Fraud Detection API")
templates = Jinja2Templates(directory="templates")
settings = load_settings()

MODEL_PATH = resolve_path(os.getenv("PRODUCTION_MODEL_PATH", settings["deployment"]["production_model_path"]))
METRICS_PATH = resolve_path(settings["deployment"]["production_metrics_path"])
METADATA_PATH = resolve_path(settings["deployment"]["metadata_path"])
LEGACY_MODEL_PATH = resolve_path("models/best_model.pkl")

def get_or_create_metric(metric_type, name, documentation, labelnames):
    if metric_type is None or REGISTRY is None:
        return None

    existing_metric = REGISTRY._names_to_collectors.get(name)
    if existing_metric is not None:
        return existing_metric

    return metric_type(name, documentation, labelnames)


REQUEST_COUNT = get_or_create_metric(Counter, "fraud_api_requests_total", "Total API requests", ["method", "endpoint"])
REQUEST_LATENCY = get_or_create_metric(Histogram, "fraud_api_request_latency_seconds", "API latency", ["endpoint"])
PREDICTION_COUNT = get_or_create_metric(Counter, "fraud_api_predictions_total", "Predictions by class", ["label"])

def load_metrics():
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return None


def load_metadata():
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_model_bundle():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    if LEGACY_MODEL_PATH.exists():
        legacy_model = joblib.load(LEGACY_MODEL_PATH)
        return {
            "model_name": "Legacy Model",
            "model": legacy_model,
            "feature_columns": list(getattr(legacy_model, "feature_names_in_", ["Amount", "V1"])),
            "metrics": load_metrics() or {}
        }

    return None


model_bundle = load_model_bundle()
model = model_bundle["model"] if model_bundle else None


@app.middleware("http")
async def collect_metrics(request: Request, call_next):
    start_time = time.perf_counter()
    if REQUEST_COUNT:
        REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()

    response = await call_next(request)

    if REQUEST_LATENCY:
        REQUEST_LATENCY.labels(endpoint=request.url.path).observe(time.perf_counter() - start_time)

    return response

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    metrics = load_metrics()
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html", 
        context={"metrics": metrics, "prediction": None}
    )

@app.get("/health", response_class=JSONResponse)
async def health():
    return {
        "status": "ok" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_name": model_bundle.get("model_name") if model_bundle else None
    }

@app.get("/model-info", response_class=JSONResponse)
async def model_info():
    return {
        "model_name": model_bundle.get("model_name") if model_bundle else None,
        "metrics": model_bundle.get("metrics") if model_bundle else None,
        "deployment": load_metadata()
    }

@app.get("/metrics")
async def metrics_endpoint():
    if generate_latest is None:
        return PlainTextResponse("prometheus_client is not installed.\n", status_code=503)
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_class=HTMLResponse)
async def make_prediction(request: Request, amount: float = Form(...), v1: float = Form(...)):
    if model is None:
        return HTMLResponse(content="<h3>Error: Model not found! Please run `python main.py` first.</h3>")

    expected_columns = model_bundle.get("feature_columns", list(getattr(model, "feature_names_in_", ["Amount", "V1"])))
    input_data = {col: 0.0 for col in expected_columns}
    if 'Amount' in input_data:
        input_data['Amount'] = amount
    if 'V1' in input_data:
        input_data['V1'] = v1
    if 'amount_surge' in input_data and amount > 5000:
        input_data['amount_surge'] = 1.0
    if 'Hour' in input_data:
        input_data['Hour'] = 0.0

    df = pd.DataFrame([input_data])
    prediction = int(model.predict(df)[0])

    if PREDICTION_COUNT:
        PREDICTION_COUNT.labels(label=str(prediction)).inc()

    metrics = load_metrics()
    
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html", 
        context={"metrics": metrics, "prediction": prediction}
    )

if __name__ == "__main__":
    print("Starting Fraud Detection API on http://localhost:8000")
    print("Press CTRL+C to quit.")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
