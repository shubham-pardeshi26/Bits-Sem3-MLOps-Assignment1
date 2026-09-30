"""The preprocessing that ships inside the model: derived feature, input contract, missing data."""

import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from cardiorisk.features import INPUT_FEATURES, add_clinical_features, build_pipeline


def fit(X: pd.DataFrame, y: pd.Series):
    return build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)


def test_hr_achieved_frac_is_the_share_of_age_predicted_max_heart_rate():
    X = pd.DataFrame({"age": [40, 60, 50], "thalach": [180, 120, None]})
    out = add_clinical_features(X.reindex(columns=INPUT_FEATURES))

    assert list(out.columns) == [*INPUT_FEATURES, "hr_achieved_frac"]
    np.testing.assert_allclose(out["hr_achieved_frac"], [180 / 180, 120 / 160, np.nan])


def test_extra_columns_never_reach_the_model(patients):
    y = patients["target"]
    # num and target would leak the label and site is not known for a new patient: all dropped.
    leaky = patients.assign(site="switzerland", num=2 * y)
    reference = fit(patients[INPUT_FEATURES], y)
    leaky_fit = fit(leaky, y)

    assert list(leaky_fit[:-1].get_feature_names_out()) == list(
        reference[:-1].get_feature_names_out()
    )
    expected = reference.predict_proba(patients[INPUT_FEATURES])
    np.testing.assert_array_equal(leaky_fit.predict_proba(leaky), expected)
    # Column order does not matter either: inputs are selected by name.
    np.testing.assert_array_equal(reference.predict_proba(leaky[INPUT_FEATURES[::-1]]), expected)


@pytest.mark.parametrize(
    "record",
    [
        {"age": 58, "sex": 1, "cp": 4},  # only the API's required fields
        {"age": 58, "sex": 1, "cp": 4, "chol": None, "thal": None, "ca": np.nan},
    ],
)
def test_a_record_with_only_age_sex_and_cp_is_scored(patients, record):
    model = fit(patients[INPUT_FEATURES], patients["target"])
    X = pd.DataFrame([record], columns=INPUT_FEATURES)

    matrix = model[:-1].transform(X)
    assert np.isfinite(matrix).all()
    assert 0 < model.predict_proba(X)[0, 1] < 1


def test_the_preprocessed_matrix_has_no_missing_values(patients):
    assert patients[INPUT_FEATURES].isna().any().any()  # the fixture has gaps to fill
    model = fit(patients[INPUT_FEATURES], patients["target"])
    assert not np.isnan(model[:-1].transform(patients[INPUT_FEATURES])).any()


def test_pickle_round_trip_gives_identical_probabilities(patients):
    model = fit(patients[INPUT_FEATURES], patients["target"])
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(
        restored.predict_proba(patients[INPUT_FEATURES]),
        model.predict_proba(patients[INPUT_FEATURES]),
    )
