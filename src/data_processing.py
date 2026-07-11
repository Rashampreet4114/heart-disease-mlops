"""Cleaning utilities for the Heart Disease UCI dataset.

The raw UCI target ``num`` is multi-class severity (0-4). Per the assignment's
binary risk-prediction framing, it is collapsed to 0 (no disease) / 1 (disease).
"""
import pathlib

import pandas as pd

RAW_PATH = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw" / "heart_disease.csv"
PROCESSED_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "processed"

NUMERIC_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
TARGET_COL = "target"


def load_raw(path: pathlib.Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ca and thal have a handful of missing values in the raw UCI export;
    # median/mode imputation keeps the small sample size intact.
    df["ca"] = df["ca"].fillna(df["ca"].median())
    df["thal"] = df["thal"].fillna(df["thal"].mode()[0])

    df[TARGET_COL] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])

    return df


def load_clean_data() -> pd.DataFrame:
    return clean(load_raw())


def save_processed(df: pd.DataFrame) -> pathlib.Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "heart_disease_clean.csv"
    df.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    cleaned = load_clean_data()
    path = save_processed(cleaned)
    print(f"Saved cleaned dataset ({cleaned.shape[0]} rows) to {path}")
