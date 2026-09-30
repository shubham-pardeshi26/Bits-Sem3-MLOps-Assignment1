"""The single inference path (predict_records) and its CLI, on the synthetic conftest model."""

import json
from pathlib import Path

import pandas as pd
import pytest
from conftest import make_patients

from cardiorisk.features import INPUT_FEATURES
from cardiorisk.predict import load_model, load_model_card, main, predict_records

SAMPLE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "sample_request.json"
KEYS = {"prediction", "label", "confidence", "probability_disease"}


def assert_valid(result: dict) -> None:
    assert set(result) == KEYS
    p = result["probability_disease"]
    assert isinstance(p, float) and 0 <= p <= 1 and round(p, 4) == p
    assert result["prediction"] == int(p >= 0.5)
    assert result["label"] == ("heart_disease" if result["prediction"] else "no_heart_disease")
    assert result["confidence"] == pytest.approx(max(p, 1 - p), abs=1e-9)


def test_predict_records_scores_full_and_partial_patients(model_file):
    model = load_model(model_file)
    records = [
        json.loads(SAMPLE_PATH.read_text()),
        {"age": 45, "sex": 0, "cp": 2},  # optional fields omitted
        {"age": 70, "sex": 1, "cp": 4, "chol": None, "thal": None},  # explicit nulls
    ]
    results = predict_records(model, records)

    assert len(results) == 3
    for result in results:
        assert_valid(result)
    assert results[0]["label"] == "heart_disease"  # the sample is a clearly high-risk patient
    # Column 1 of predict_proba is the disease class, and batching does not change a score.
    expected = model.predict_proba(pd.DataFrame(records, columns=INPUT_FEATURES))[:, 1]
    assert [r["probability_disease"] for r in results] == [round(float(p), 4) for p in expected]
    assert [predict_records(model, [r])[0] for r in records] == results


def test_model_card_is_optional(tmp_path):
    assert load_model_card(tmp_path / "missing.json") == {}
    card = tmp_path / "card.json"
    card.write_text('{"version": "7"}')
    assert load_model_card(card) == {"version": "7"}


def test_cli_scores_a_json_patient(model_file, capsys):
    main(["--input", str(SAMPLE_PATH), "--model", model_file])
    assert_valid(json.loads(capsys.readouterr().out))


def test_cli_scores_a_csv_and_ignores_extra_columns(model_file, tmp_path, capsys):
    csv = tmp_path / "patients.csv"
    make_patients(5).to_csv(csv, index=False)  # has a target column and empty (missing) cells
    main(["--csv", str(csv), "--model", model_file])
    results = json.loads(capsys.readouterr().out)
    assert len(results) == 5
    for result in results:
        assert_valid(result)
