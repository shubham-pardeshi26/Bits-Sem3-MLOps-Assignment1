"""Feature engineering and preprocessing, packaged inside the model pipeline.

Everything that turns a raw patient record into model input lives in `build_pipeline`, so the
pickled pipeline reproduces training-time transforms exactly at inference time.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# Raw clinical inputs the model (and the API) accepts. "num", "site" and "target" are never inputs.
INPUT_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]  # fmt: skip
DERIVED_FEATURES = ["hr_achieved_frac"]

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "hr_achieved_frac"]
BINARY = ["sex", "fbs", "exang"]
CATEGORICAL = ["cp", "restecg", "slope", "ca", "thal"]


def add_clinical_features(X: pd.DataFrame) -> pd.DataFrame:
    """Keep only the model inputs (as floats; None -> NaN) and add derived clinical features.

    hr_achieved_frac = thalach / (220 - age): share of the age-predicted maximum heart rate
    reached on the exercise test. A low value (chronotropic incompetence) is a cardiac risk marker.
    """
    X = X[INPUT_FEATURES].astype(float)
    return X.assign(hr_achieved_frac=X["thalach"] / (220 - X["age"]))


def _feature_names_out(_transformer, _input_features):
    return [*INPUT_FEATURES, *DERIVED_FEATURES]


def build_preprocessor() -> ColumnTransformer:
    """Impute (with missing-value indicators), scale numerics, one-hot encode categoricals.

    Missingness is informative in this dataset (whole sites never recorded ca/thal/chol), so
    numeric/binary imputers add indicator columns and the encoder keeps NaN as its own category.
    """
    return ColumnTransformer(
        [
            (
                "num",
                make_pipeline(
                    SimpleImputer(strategy="median", add_indicator=True), StandardScaler()
                ),
                NUMERIC,
            ),
            ("bin", SimpleImputer(strategy="most_frequent", add_indicator=True), BINARY),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ]
    )


def build_pipeline(model) -> Pipeline:
    """Full inference pipeline: raw patient DataFrame -> features -> preprocessing -> model."""
    return Pipeline(
        [
            (
                "features",
                FunctionTransformer(add_clinical_features, feature_names_out=_feature_names_out),
            ),
            ("preprocess", build_preprocessor()),
            ("model", model),
        ]
    )
