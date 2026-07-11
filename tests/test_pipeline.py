import numpy as np

from src.data_processing import load_clean_data
from src.pipeline import ALL_FEATURES, build_preprocessor


def test_all_features_present_in_clean_data():
    df = load_clean_data()
    assert set(ALL_FEATURES).issubset(df.columns)


def test_preprocessor_transforms_without_error():
    df = load_clean_data()
    X = df[ALL_FEATURES]

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[0] == X.shape[0]
    assert not np.isnan(transformed).any()


def test_preprocessor_handles_unseen_category():
    df = load_clean_data()
    X = df[ALL_FEATURES]

    preprocessor = build_preprocessor()
    preprocessor.fit(X)

    unseen = X.iloc[[0]].copy()
    unseen["thal"] = 999  # category never seen during fit

    transformed = preprocessor.transform(unseen)
    assert transformed.shape[0] == 1
