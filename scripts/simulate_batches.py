import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_ops import bootstrap_batches
from src.settings import load_settings


def parse_args():
    parser = argparse.ArgumentParser(description="Split the source CSV into simulated incoming batches.")
    parser.add_argument("--config", default="config/pipeline.json")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    settings = load_settings(args.config)
    batch_files = bootstrap_batches(settings, force=args.force)
    print(f"Prepared {len(batch_files)} batch file(s).")
    for batch_file in batch_files:
        print(batch_file)
