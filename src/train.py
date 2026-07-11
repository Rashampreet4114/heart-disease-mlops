"""Train, tune, evaluate and MLflow-track Logistic Regression, Random Forest and
XGBoost classifiers on the Heart Disease dataset, then persist the best pipeline.

Usage:
    python -m src.train
"""
import json
import pathlib

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data_processing import load_clean_data
from src.pipeline import ALL_FEATURES, build_preprocessor

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
SCREENSHOTS_DIR = ROOT / "screenshots"
RANDOM_STATE = 42

MODEL_SPECS = {
    "logistic_regression": {
        "estimator": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "param_grid": {
            "model__C": [0.01, 0.1, 1, 10],
            "model__solver": ["lbfgs", "liblinear"],
        },
    },
    "random_forest": {
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
        "param_grid": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 5, None],
            "model__min_samples_leaf": [1, 2, 4],
        },
    },
    "xgboost": {
        "estimator": XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE),
        "param_grid": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 4, 5],
            "model__learning_rate": [0.01, 0.1, 0.2],
        },
    },
}


def evaluate(estimator, X_test, y_test) -> dict:
    y_pred = estimator.predict(X_test)
    y_proba = estimator.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }, y_pred, y_proba


def log_confusion_matrix(y_test, y_pred, name: str) -> pathlib.Path:
    fig, ax = plt.subplots(figsize=(4, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    path = SCREENSHOTS_DIR / f"confusion_matrix_{name}.png"
    plt.savefig(path, dpi=120)
    plt.close(fig)
    return path


def log_roc_curve(estimator, X_test, y_test, name: str) -> pathlib.Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_estimator(estimator, X_test, y_test, ax=ax)
    ax.set_title(f"ROC Curve - {name}")
    plt.tight_layout()
    path = SCREENSHOTS_DIR / f"roc_curve_{name}.png"
    plt.savefig(path, dpi=120)
    plt.close(fig)
    return path


def train_and_log_model(name: str, spec: dict, X_train, X_test, y_train, y_test) -> dict:
    pipe = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model", spec["estimator"]),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(pipe, spec["param_grid"], scoring="roc_auc", cv=cv, n_jobs=-1)

    with mlflow.start_run(run_name=name):
        search.fit(X_train, y_train)
        best_pipe = search.best_estimator_

        cv_scores = cross_val_score(best_pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        test_metrics, y_pred, y_proba = evaluate(best_pipe, X_test, y_test)

        mlflow.log_params(search.best_params_)
        mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
        mlflow.log_metric("cv_roc_auc_std", cv_scores.std())
        for metric_name, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", value)

        cm_path = log_confusion_matrix(y_test, y_pred, name)
        roc_path = log_roc_curve(best_pipe, X_test, y_test, name)
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(roc_path))

        mlflow.sklearn.log_model(best_pipe, artifact_path="model")

        print(f"[{name}] best_params={search.best_params_}")
        print(f"[{name}] cv_roc_auc={cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"[{name}] test_metrics={test_metrics}")

        return {
            "name": name,
            "pipeline": best_pipe,
            "cv_roc_auc_mean": cv_scores.mean(),
            "test_metrics": test_metrics,
        }


def main():
    mlflow.set_experiment("heart-disease-classification")

    df = load_clean_data()
    X = df[ALL_FEATURES]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    results = [
        train_and_log_model(name, spec, X_train, X_test, y_train, y_test)
        for name, spec in MODEL_SPECS.items()
    ]

    best = max(results, key=lambda r: r["test_metrics"]["roc_auc"])
    print(f"\nBest model: {best['name']} (test ROC-AUC={best['test_metrics']['roc_auc']:.4f})")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "model.joblib"
    joblib.dump(best["pipeline"], model_path)

    metadata = {
        "best_model": best["name"],
        "features": ALL_FEATURES,
        "test_metrics": best["test_metrics"],
        "cv_roc_auc_mean": best["cv_roc_auc_mean"],
        "all_results": {
            r["name"]: {"test_metrics": r["test_metrics"], "cv_roc_auc_mean": r["cv_roc_auc_mean"]}
            for r in results
        },
    }
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved best pipeline to {model_path}")
    print(f"Saved metadata to {MODELS_DIR / 'metadata.json'}")


if __name__ == "__main__":
    main()
