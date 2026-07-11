"""Preprocessing pipeline shared by training and inference."""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]

ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="passthrough",  # binary features pass through as-is (already 0/1)
    )
