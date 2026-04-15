import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from main import run_pipeline


if __name__ == "__main__":
    run_pipeline(max_batches=2)

    if not Path("models/production_model.pkl").exists():
        raise SystemExit("Smoke test failed: production model was not created.")

    print("Smoke test passed.")
