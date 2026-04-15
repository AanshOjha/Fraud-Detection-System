from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.settings import resolve_path


def list_batch_files(batch_dir: str | Path) -> list[Path]:
    directory = resolve_path(batch_dir)
    if not directory.exists():
        return []
    return sorted(directory.glob("batch_*.csv"))


def bootstrap_batches(settings: dict, force: bool = False) -> list[Path]:
    source_csv = resolve_path(settings["data"]["source_csv"])
    batch_dir = resolve_path(settings["data"]["batch_dir"])
    batch_dir.mkdir(parents=True, exist_ok=True)

    existing_batches = list_batch_files(batch_dir)
    if existing_batches and not force:
        return existing_batches

    if force:
        for batch_file in existing_batches:
            batch_file.unlink()

    if not source_csv.exists():
        raise FileNotFoundError(f"Dataset not found at {source_csv}")

    df = pd.read_csv(source_csv)
    sample_rows = settings["simulation"].get("sample_rows")
    if sample_rows:
        df = df.head(int(sample_rows))

    if "Time" in df.columns:
        df = df.sort_values("Time").reset_index(drop=True)

    batch_count = max(1, int(settings["simulation"].get("bootstrap_batch_count", 1)))
    chunk_size = max(1, len(df) // batch_count)

    batch_files = []
    for batch_index, start in enumerate(range(0, len(df), chunk_size), start=1):
        batch = df.iloc[start : start + chunk_size]
        if batch.empty:
            continue

        batch_path = batch_dir / f"batch_{batch_index:03d}.csv"
        batch.to_csv(batch_path, index=False)
        batch_files.append(batch_path)

    return batch_files


def load_training_dataframe(settings: dict, max_batches: int | None = None) -> tuple[pd.DataFrame, list[Path]]:
    batch_files = list_batch_files(settings["data"]["batch_dir"])
    if not batch_files:
        batch_files = bootstrap_batches(settings)

    if max_batches:
        batch_files = batch_files[:max_batches]

    if not batch_files:
        raise RuntimeError("No batch files available for training.")

    dataframes = [pd.read_csv(batch_file) for batch_file in batch_files]
    return pd.concat(dataframes, ignore_index=True), batch_files
