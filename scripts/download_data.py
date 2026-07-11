"""Download the Heart Disease UCI dataset and save it to data/raw/.

Usage:
    python scripts/download_data.py
"""
import pathlib

import pandas as pd
from ucimlrepo import fetch_ucirepo

RAW_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"
UCI_DATASET_ID = 45  # Heart Disease dataset on UCI ML Repository


def download() -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    dataset = fetch_ucirepo(id=UCI_DATASET_ID)
    features = dataset.data.features
    targets = dataset.data.targets

    df = pd.concat([features, targets], axis=1)
    out_path = RAW_DIR / "heart_disease.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows x {len(df.columns)} cols to {out_path}")
    return df


if __name__ == "__main__":
    download()
