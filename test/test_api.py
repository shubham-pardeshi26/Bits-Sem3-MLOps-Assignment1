"""HTTP contract of the API, served offline: the lifespan loads the synthetic conftest model."""

import json

import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families
from test_predict import SAMPLE_PATH, assert_valid

# Imported once per session: the Prometheus metrics live in prometheus_client's global registry.
from cardiorisk.api import app
from cardiorisk.features import INPUT_FEATURES

SAMPLE = json.loads(SAMPLE_PATH.read_text())
CARD = {
    "version": "test-7",
    "family": "logistic_regression",
    "trained_at": "2026-09-29T12:00:00+00:00",
    "threshold": 0.5,
    "metrics": {"cv": {"roc_auc": 0.91}, "test": {"roc_auc": 0.88}},
    "features": INPUT_FEATURES,
    "mlflow_run_id": "0123456789abcdef",
    "sklearn_version": "1.9.1",
}


@pytest.fixture
def served_model(model_file, tmp_path, monkeypatch):
    """Point the app at the synthetic model and a matching card, as the container env would."""
    card = tmp_path / "model_card.json"
    card.write_text(json.dumps(CARD))
    monkeypatch.setenv("MODEL_PATH", model_file)
    monkeypatch.setenv("MODEL_CARD_PATH", str(card))


@pytest.fixture
def client(served_model):
    with TestClient(app) as client:  # the context manager runs the lifespan (model load)
        yield client


def test_health_reports_the_served_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_version": CARD["version"]}


def test_predict_returns_a_consistent_risk(client):
    response = client.post("/predict", json=SAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert body.pop("model_version") == CARD["version"]
    assert_valid(body)
    assert body["label"] == "heart_disease"  # the sample is a clearly high-risk patient


@pytest.mark.parametrize(
    "patient",
    [{"age": 45, "sex": 0, "cp": 2}, {"age": 45, "sex": 0, "cp": 2, "chol": None, "thal": None}],
    ids=["omitted", "null"],
)
def test_optional_measurements_may_be_missing(client, patient):
    response = client.post("/predict", json=patient)
    assert response.status_code == 200, response.text
    body = response.json()
    body.pop("model_version")
    assert_valid(body)


@pytest.mark.parametrize(
    ("field", "patient"),
    [
        ("age", {k: v for k, v in SAMPLE.items() if k != "age"}),
        ("age", {**SAMPLE, "age": None}),
        ("trestbps", {**SAMPLE, "trestbps": 20}),
        ("thal", {**SAMPLE, "thal": 4}),
        ("bmi", {**SAMPLE, "bmi": 31.2}),
    ],
    ids=["missing-age", "null-age", "out-of-range", "bad-code", "unknown-field"],
)
def test_invalid_patients_are_rejected(client, field, patient):
    response = client.post("/predict", json=patient)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert [error["loc"] for error in detail] == [["body", field]]
    assert all(set(error) == {"type", "loc", "msg"} for error in detail)  # no patient values


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_numbers_are_rejected_not_crashed(client, value):
    """Python's JSON accepts these; the default 422 handler echoed them and failed with a 500."""
    body = '{"age": 61, "sex": 1, "cp": 4, "chol": ' + value + "}"
    response = client.post("/predict", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert [error["loc"] for error in detail] == [["body", "chol"]]
    assert all(set(error) == {"type", "loc", "msg"} for error in detail)


def test_request_id_is_echoed_or_generated(client):
    echoed = client.get("/health", headers={"X-Request-ID": "trace-42"})
    assert echoed.headers["X-Request-ID"] == "trace-42"
    assert len(client.get("/health").headers["X-Request-ID"]) == 32  # uuid4 hex


def test_metrics_expose_the_contract_series(client):
    client.post("/predict", json=SAMPLE)
    client.get("/docs")
    client.get("/no/such/page")
    response = client.get("/metrics")
    assert response.status_code == 200
    series = [
        (sample.name, sample.labels, sample.value)
        for family in text_string_to_metric_families(response.text)
        for sample in family.samples
    ]

    def labels_of(name):
        return [labels for sample_name, labels, _ in series if sample_name == name]

    requests = labels_of("cardiorisk_http_requests_total")
    assert {"method": "POST", "route": "/predict", "status": "200"} in requests
    # 'route' is a template, never a raw path, so its cardinality stays bounded.
    assert {"method": "GET", "route": "/docs", "status": "200"} in requests
    assert {"method": "GET", "route": "unmatched", "status": "404"} in requests
    assert {"method": "POST", "route": "/predict"} in labels_of(
        "cardiorisk_http_request_duration_seconds_count"
    )
    assert {"label": "heart_disease"} in labels_of("cardiorisk_predictions_total")
    # Exported before its first prediction too (at 0), so rate() counts that prediction.
    assert {"label": "no_heart_disease"} in labels_of("cardiorisk_predictions_total")
    buckets = {labels["le"] for labels in labels_of("cardiorisk_prediction_probability_bucket")}
    assert buckets == {str(i / 10) for i in range(1, 11)} | {"+Inf"}
    model_info = {"version": CARD["version"], "family": CARD["family"]}
    assert ("cardiorisk_model_info", model_info, 1.0) in series


def test_model_info_summarises_the_card(client):
    info = client.get("/model-info").json()
    for key in ("version", "family", "trained_at", "metrics", "features"):
        assert info[key] == CARD[key]


def test_docs_example_is_the_sample_request(client):
    """api.SAMPLE_PATIENT prefills /docs and must stay equal to scripts/sample_request.json."""
    schema = client.get("/openapi.json").json()["components"]["schemas"]["Patient"]
    assert schema["examples"] == [SAMPLE]
    assert set(SAMPLE) == set(INPUT_FEATURES)


def test_logs_are_json_lines_without_patient_data(served_model, capsys):
    with TestClient(app) as client:  # started inside the test so logging binds to capsys
        client.post("/predict", json=SAMPLE, headers={"X-Request-ID": "log-check"})
    out = capsys.readouterr().out
    lines = [json.loads(line) for line in out.splitlines()]  # every stdout line is JSON
    events = {line["event"]: line for line in lines if line.get("request_id") == "log-check"}
    access, prediction = events["request"], events["prediction"]
    assert access["method"] == "POST" and access["route"] == "/predict"
    assert access["status"] == 200 and access["duration_ms"] >= 0
    assert prediction["label"] == "heart_disease"
    assert prediction["model_version"] == CARD["version"]
    assert 0 <= prediction["probability_disease"] <= 1
    for feature in INPUT_FEATURES:  # health data: no feature name (hence value) is ever logged
        assert f'"{feature}"' not in out


def test_probes_and_scrapes_stay_out_of_the_info_log(served_model, capsys):
    with TestClient(app) as client:
        client.get("/health")
        client.get("/metrics")
        client.post("/predict", json={**SAMPLE, "age": 5})  # a 422 is still an access line
        metrics = client.get("/metrics").text
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    routes = [line["route"] for line in lines if line["event"] == "request"]
    assert routes == ["/predict"]
    assert 'route="/health",status="200"' in metrics  # still counted
    assert 'route="/predict",status="422"' in metrics


def test_startup_fails_fast_without_a_model(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    with pytest.raises(RuntimeError, match="Cannot load model"), TestClient(app):
        pass
