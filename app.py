import os
import joblib
import json
import pandas as pd
import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Fraud Detection API")

# Setting up Jinja templates folder (clean separation of frontend and backend)
templates = Jinja2Templates(directory="templates")

MODEL_PATH = "models/best_model.pkl"
METRICS_PATH = "models/metrics.json"

# Load the trained model gracefully at startup
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

def load_metrics():
    """Helper function to load the latest evaluation metrics json."""
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return None

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """Serves main dashboard website."""
    metrics = load_metrics()
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html", 
        context={"metrics": metrics, "prediction": None}
    )

@app.post("/predict", response_class=HTMLResponse)
async def make_prediction(request: Request, amount: float = Form(...), v1: float = Form(...)):
    """Receives user input from the dashboard, runs ML inference, and renders results."""
    if model is None:
        return HTMLResponse(content="<h3>Error: Model not found! Please run `python main.py` first.</h3>")
        
    try:
        expected_columns = model.feature_names_in_
    except AttributeError:
        expected_columns = ['Amount', 'V1', 'V2', 'time_gap', 'amount_rolling_mean', 'amount_surge']
        
    # Map input into our dataset format
    input_data = {col: 0.0 for col in expected_columns}
    if 'Amount' in input_data: input_data['Amount'] = amount
    if 'V1' in input_data: input_data['V1'] = v1
    
    # Tiny demonstration logic to prove the model catches anomalies
    if amount > 5000:  
        input_data['amount_surge'] = 1.0 
        
    df = pd.DataFrame([input_data])
    prediction = int(model.predict(df)[0])
    
    metrics = load_metrics()
    
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html", 
        context={"metrics": metrics, "prediction": prediction}
    )

if __name__ == "__main__":
    print("Starting Fraud Detection API on 👉 http://localhost:8000")
    print("Press CTRL+C to quit.")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
