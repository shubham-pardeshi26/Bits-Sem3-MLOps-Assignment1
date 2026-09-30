"""The training entry point on a reduced synthetic run, and a quality gate on the real data."""

import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from conftest import make_patients
from mlflow import MlflowClient
from sklearn.model_selection import StratifiedKFold, cross_val_score

from cardiorisk import config, train
from cardiorisk.data import load_processed
from cardiorisk.features import INPUT_FEATURES, build_pipeline

CARD_FIELDS = {
    "version", "family", "trained_at", "threshold", "metrics", "features", "mlflow_run_id",
    "sklearn_version",
}  # fmt: skip
FIGURES = [
    "model_comparison",
    "model_champion_roc",
    "model_champion_confusion_matrix",
    "model_champion_permutation_importance",
    "model_loso_by_site",
]


def test_every_grid_names_real_hyper_parameters():
    """A typo in a grid would otherwise only surface in the full (CI) training run."""
    for family, (estimator, grid, note) in train.CANDIDATES.items():
        pipeline = build_pipeline(estimator)
        pipeline.set_params(**{f"model__{name}": values[0] for name, values in grid.items()})
        assert note, family


def test_reduced_training_run_writes_model_card_reports_and_mlflow_runs(tmp_path, monkeypatch):
    df = make_patients(320, seed=1).assign(site=lambda d: np.resize(list(config.SITE_FILES), 320))
    estimator, _, note = train.CANDIDATES["logistic_regression"]
    candidates = {
        "baseline_majority": train.CANDIDATES["baseline_majority"],
        "logistic_regression": (estimator, {"C": [0.1, 1.0]}, note),
    }
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)  # mlflow exports it; undone after the test

    card = train.train(df, tmp_path, uri, candidates)

    assert json.loads((tmp_path / "models" / "model_card.json").read_text()) == card
    assert CARD_FIELDS <= card.keys()
    assert card["family"] == "logistic_regression"  # the baseline is never the champion
    assert card["threshold"] == 0.5 and card["features"] == INPUT_FEATURES
    assert {"cv", "test", "loso"} <= card["metrics"].keys()
    assert set(card["metrics"]["loso"]) == set(config.SITE_FILES)
    assert card["n_train"] + card["n_test"] == len(df)

    model = joblib.load(tmp_path / "models" / "cardiorisk_model.joblib")
    proba = model.predict_proba(df[INPUT_FEATURES].head())[:, 1]
    assert ((proba > 0) & (proba < 1)).all()
    metrics = json.loads((tmp_path / "reports" / "metrics.json").read_text())
    assert set(metrics["families"]) == set(candidates) and metrics["champion"] == card["family"]
    assert "**logistic_regression** (champion)" in (tmp_path / "reports" / "metrics.md").read_text()
    for name in FIGURES:
        assert (tmp_path / "reports" / "figures" / f"{name}.png").stat().st_size > 0, name

    client = MlflowClient(uri)
    experiment = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT)
    assert experiment.artifact_location.startswith(tmp_path.as_uri())  # not the repo's mlruns/
    runs = client.search_runs([experiment.experiment_id])
    families = {r.info.run_name: r for r in runs if "mlflow.parentRunId" not in r.data.tags}
    assert set(families) == set(candidates)
    nested = [r for r in runs if "mlflow.parentRunId" in r.data.tags]
    assert len(nested) == 1 + 2  # one nested run per grid point
    for run in nested:
        folds = json.loads(
            Path(client.download_artifacts(run.info.run_id, "cv_folds.json")).read_text()
        )
        assert set(folds) == set(train.METRICS) and all(len(v) == 5 for v in folds.values())
    champion_run = families["logistic_regression"]
    assert champion_run.info.run_id == card["mlflow_run_id"]
    assert champion_run.data.tags["champion"] == "true"
    assert "loso_va_roc_auc" in champion_run.data.metrics
    assert "test_roc_auc" in champion_run.data.metrics and "C" in champion_run.data.params
    version = client.get_model_version_by_alias(config.REGISTERED_MODEL, "champion")
    assert str(version.version) == card["version"]  # the card stores it as a string


@pytest.mark.skipif(not config.PROCESSED_CSV.is_file(), reason="run 'make data' first")
def test_logistic_regression_clears_the_quality_gate_on_the_real_data():
    df = load_processed()
    estimator = train.CANDIDATES["logistic_regression"][0]
    cv = StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE)
    X, y = df[INPUT_FEATURES].astype(float), df[config.TARGET]
    scores = cross_val_score(build_pipeline(estimator), X, y, cv=cv, scoring="roc_auc")
    assert scores.mean() >= 0.80
