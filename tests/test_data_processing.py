import pandas as pd
import pytest

from src.data_processing import clean, load_raw


@pytest.fixture(scope="module")
def raw_df():
    return load_raw()


@pytest.fixture(scope="module")
def clean_df(raw_df):
    return clean(raw_df)


def test_raw_data_loads(raw_df):
    assert len(raw_df) > 0
    assert "num" in raw_df.columns


def test_clean_has_no_missing_values(clean_df):
    assert clean_df.isna().sum().sum() == 0


def test_clean_target_is_binary(clean_df):
    assert set(clean_df["target"].unique()).issubset({0, 1})
    assert "num" not in clean_df.columns


def test_clean_preserves_row_count(raw_df, clean_df):
    assert len(clean_df) == len(raw_df)


def test_clean_is_idempotent_on_missing_columns():
    df = pd.DataFrame({
        "ca": [0.0, None, 2.0],
        "thal": [3.0, 3.0, None],
        "num": [0, 1, 2],
    })
    result = clean(df)
    assert result["ca"].isna().sum() == 0
    assert result["thal"].isna().sum() == 0
    assert list(result["target"]) == [0, 1, 1]
