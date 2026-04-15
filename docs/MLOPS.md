# Simple Local MLOps Guide

This project now supports a beginner-friendly local MLOps workflow.

## What It Does

- Splits the main CSV into smaller batch files to simulate new incoming data
- Trains multiple models on the available batches
- Picks the best model using a chosen metric
- Saves a latest candidate model
- Promotes the candidate to production if it beats the current production model
- Lets the FastAPI app serve the promoted production model
- Adds Docker and GitHub Actions for deployment practice

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Simulate Incoming Data

```bash
python scripts/simulate_batches.py --force
```

This creates sequential files inside `data/raw/batches/`.

## 3. Train the Pipeline

Use all available batches:

```bash
python main.py
```

Use only the first few batches:

```bash
python main.py --max-batches 3
```

## 4. View MLflow Runs

If MLflow is installed, runs are logged automatically.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## 5. Run the API

```bash
python app.py
```

Useful endpoints:

- `/`
- `/health`
- `/model-info`
- `/metrics`

## 6. Build the Container

```bash
docker build -t fraud-api .
```

## 7. What GitHub Actions Does

On push or pull request it:

1. Installs dependencies
2. Rebuilds the simulated batches
3. Runs a smoke training pipeline
4. Builds the Docker image
