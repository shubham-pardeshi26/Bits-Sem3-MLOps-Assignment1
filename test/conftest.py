"""Shared offline fixtures: synthetic patients and a small fitted pipeline."""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from cardiorisk.features import INPUT_FEATURES, build_pipeline


def make_patients(n: int = 240, seed: int = 0) -> pd.DataFrame:
    """Synthetic patients in realistic ranges, with missing values and a learnable target."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "age": rng.integers(29, 78, n),
            "sex": rng.integers(0, 2, n),
            "cp": rng.integers(1, 5, n),
            "trestbps": rng.normal(132, 18, n).round(),
            "chol": rng.normal(240, 50, n).round(),
            "fbs": rng.integers(0, 2, n),
            "restecg": rng.integers(0, 3, n),
            "thalach": rng.normal(140, 25, n).round(),
            "exang": rng.integers(0, 2, n),
            "oldpeak": rng.gamma(1.2, 0.9, n).round(1),
            "slope": rng.integers(1, 4, n),
            "ca": rng.integers(0, 4, n),
            "thal": rng.choice([3, 6, 7], n),
        }
    ).astype(float)
    risk = (
        0.05 * (df.age - 54)
        + 0.9 * (df.cp == 4)
        + 0.9 * df.exang
        + 0.6 * df.oldpeak
        - 0.02 * (df.thalach - 140)
        + 0.6 * df.sex
    )
    df["target"] = (risk + rng.normal(0, 0.8, n) > 1.0).astype(int)
    for col, frac in {"chol": 0.15, "ca": 0.5, "thal": 0.4, "slope": 0.3, "fbs": 0.1}.items():
        df.loc[rng.random(n) < frac, col] = np.nan
    return df


@pytest.fixture
def patients() -> pd.DataFrame:
    return make_patients()


@pytest.fixture(scope="session")
def model_file(tmp_path_factory) -> str:
    """A fitted pipeline saved like models/cardiorisk_model.joblib (independent of real data)."""
    df = make_patients()
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(df[INPUT_FEATURES], df["target"])
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    joblib.dump(pipe, path)
    return str(path)
