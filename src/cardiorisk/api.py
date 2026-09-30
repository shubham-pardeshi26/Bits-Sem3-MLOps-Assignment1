"""HTTP service for the cardiorisk model: validated input, one inference path, observability.

Every request gets an id (X-Request-ID in and out), Prometheus samples and one JSON log line on
stdout. Patient features are never logged: they are health data.
"""

import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Literal

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field
from starlette.routing import Match

from cardiorisk import config
from cardiorisk.features import INPUT_FEATURES
from cardiorisk.predict import LABELS, THRESHOLD, load_model, load_model_card, predict_records

log = logging.getLogger(__name__)
QUIET_ROUTES = {"/health", "/metrics"}  # probes and scrapes: access lines at DEBUG, not INFO

REQUESTS = Counter(
    "cardiorisk_http_requests_total",
    "HTTP requests by method, route template and status code.",
    ["method", "route", "status"],
)
LATENCY = Histogram(
    "cardiorisk_http_request_duration_seconds",
    "Request latency by method and route template.",
    ["method", "route"],
)
PREDICTIONS = Counter("cardiorisk_predictions_total", "Predictions by returned label.", ["label"])
for _label in LABELS:  # exported at 0 from the start: rate() never counts a series' first sample
    PREDICTIONS.labels(_label)
PROBABILITY = Histogram(
    "cardiorisk_prediction_probability",
    "Predicted probability of heart disease; a shift in this distribution hints at input drift.",
    buckets=[i / 10 for i in range(1, 11)],
)
MODEL_INFO = Gauge("cardiorisk_model_info", "Model being served (always 1).", ["version", "family"])

# Same values as scripts/sample_request.json (a test keeps them in sync); prefilled in /docs.
SAMPLE_PATIENT = {
    "age": 61, "sex": 1, "cp": 4, "trestbps": 148, "chol": 286, "fbs": 0, "restecg": 2,
    "thalach": 118, "exang": 1, "oldpeak": 2.4, "slope": 2, "ca": 2, "thal": 7,
}  # fmt: skip


class Patient(BaseModel):
    """One patient in UCI Heart Disease coding.

    Only age, sex and chest pain type are required. The other measurements may be null or omitted:
    the model pipeline imputes them, as it does for the hospitals that never recorded them.
    """

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [SAMPLE_PATIENT]})

    age: int = Field(ge=18, le=100, description="Age in years.")
    sex: Literal[0, 1] = Field(description="1 = male, 0 = female.")
    cp: Literal[1, 2, 3, 4] = Field(
        description="Chest pain type: 1 = typical angina, 2 = atypical angina, "
        "3 = non-anginal pain, 4 = asymptomatic."
    )
    trestbps: float | None = Field(
        None, ge=60, le=250, description="Resting blood pressure on admission, mm Hg."
    )
    chol: float | None = Field(None, ge=80, le=700, description="Serum cholesterol, mg/dl.")
    fbs: Literal[0, 1] | None = Field(
        None, description="Fasting blood sugar above 120 mg/dl: 1 = yes, 0 = no."
    )
    restecg: Literal[0, 1, 2] | None = Field(
        None,
        description="Resting ECG: 0 = normal, 1 = ST-T wave abnormality, "
        "2 = probable or definite left ventricular hypertrophy (Estes' criteria).",
    )
    thalach: float | None = Field(
        None, ge=50, le=230, description="Maximum heart rate achieved on the exercise test, bpm."
    )
    exang: Literal[0, 1] | None = Field(
        None, description="Exercise-induced angina: 1 = yes, 0 = no."
    )
    oldpeak: float | None = Field(
        None, ge=-3, le=7, description="ST depression induced by exercise relative to rest, mm."
    )
    slope: Literal[1, 2, 3] | None = Field(
        None,
        description="Slope of the peak exercise ST segment: "
        "1 = upsloping, 2 = flat, 3 = downsloping.",
    )
    ca: Literal[0, 1, 2, 3] | None = Field(
        None, description="Number of major vessels (0-3) coloured by fluoroscopy."
    )
    thal: Literal[3, 6, 7] | None = Field(
        None,
        description="Thallium stress test: 3 = normal, 6 = fixed defect, 7 = reversible defect.",
    )


class RiskPrediction(BaseModel):
    prediction: Literal[0, 1] = Field(description=f"1 when probability_disease >= {THRESHOLD}.")
    label: Literal["heart_disease", "no_heart_disease"]
    confidence: float = Field(description="Probability of the returned class.")
    probability_disease: float = Field(description="Predicted probability of heart disease.")
    model_version: str


class _JsonLines(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "event": record.getMessage(),
            **getattr(record, "fields", {}),
        }
        return json.dumps(entry, default=str)


def _configure_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonLines())
    package_log = logging.getLogger("cardiorisk")
    package_log.handlers[:] = [handler]  # replace, so a restarted app (tests) never doubles lines
    package_log.setLevel(level.upper())
    # The middleware writes the access log; uvicorn's plain-text one would break JSON-only stdout.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once per process. Env paths are read at startup so tests can swap them."""
    _configure_logging(os.environ.get("LOG_LEVEL", "INFO"))
    model_path = os.environ.get("MODEL_PATH", str(config.MODEL_PATH))
    card_path = os.environ.get("MODEL_CARD_PATH", str(config.MODEL_CARD_PATH))
    try:
        model, card = load_model(model_path), load_model_card(card_path)
    except Exception as exc:
        # Fail fast: a pod without its model must crash and say why, not pass health checks.
        raise RuntimeError(
            f"Cannot load model (MODEL_PATH={model_path}, MODEL_CARD_PATH={card_path}): {exc}"
        ) from exc
    if not card:
        log.warning("model_card_missing", extra={"fields": {"path": card_path}})
    info = {
        "version": str(card.get("version", "unknown")),
        "family": str(card.get("family", "unknown")),
        "trained_at": card.get("trained_at"),
        "threshold": THRESHOLD,
        "metrics": card.get("metrics", {}),
        "features": card.get("features", INPUT_FEATURES),
        "mlflow_run_id": card.get("mlflow_run_id"),
    }
    MODEL_INFO.clear()
    MODEL_INFO.labels(version=info["version"], family=info["family"]).set(1)
    log.info(
        "model_loaded",
        extra={"fields": {"model_version": info["version"], "family": info["family"]}},
    )
    yield {"model": model, "info": info}


app = FastAPI(
    title="CardioRisk API",
    version="1.0.0",
    summary="Heart disease risk from routine clinical measurements (UCI Heart Disease coding).",
    description="Coursework model trained on the four UCI Heart Disease hospitals. "
    "Not for clinical use.",
    lifespan=lifespan,
)


def _route_label(request: Request) -> str:
    """Route template for metric labels, so cardinality stays bounded ('unmatched' for 404s).

    Matched here because FastAPI only records APIRoutes in the scope, not /docs or /openapi.json.
    """
    for route in request.app.routes:
        if route.matches(request.scope)[0] != Match.NONE:
            return route.path
    return "unmatched"


@app.middleware("http")
async def observe(request: Request, call_next):
    """Request id, latency, Prometheus samples and one JSON access-log line per request."""
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = request_id
    start, status = time.perf_counter(), 500  # stays 500 if the handler raises
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        elapsed = time.perf_counter() - start
        route = _route_label(request)
        REQUESTS.labels(request.method, route, str(status)).inc()
        LATENCY.labels(request.method, route).observe(elapsed)
        access = {
            "request_id": request_id,
            "method": request.method,
            "route": route,
            "status": status,
            "duration_ms": round(elapsed * 1000, 2),
        }
        level = logging.DEBUG if route in QUIET_ROUTES else logging.INFO
        log.log(level, "request", extra={"fields": access})


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
    """422 with type, loc and msg only.

    The default handler echoes 'input' and 'ctx': patient values leak back, and a JSON NaN or
    Infinity there is not serialisable, so the reply would crash into a 500.
    """
    detail = [{key: error[key] for key in ("type", "loc", "msg")} for error in exc.errors()]
    return JSONResponse({"detail": detail}, status_code=422)


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse("/docs")


@app.get("/health")
async def health(request: Request) -> dict:
    """Liveness and readiness: the model loads at startup, so a running process can predict."""
    return {"status": "ok", "model_version": request.state.info["version"]}


# Sync on purpose: sklearn is CPU-bound, so FastAPI runs it in the threadpool and the event loop
# stays free for the async probe and scrape handlers.
@app.post("/predict")
def predict(patient: Patient, request: Request) -> RiskPrediction:
    """Heart disease risk for one patient."""
    result = predict_records(request.state.model, [patient.model_dump()])[0]
    version = request.state.info["version"]
    PREDICTIONS.labels(result["label"]).inc()
    PROBABILITY.observe(result["probability_disease"])
    outcome = {
        "request_id": request.state.request_id,
        "label": result["label"],
        "probability_disease": result["probability_disease"],
        "model_version": version,
    }
    log.info("prediction", extra={"fields": outcome})
    return RiskPrediction(**result, model_version=version)


@app.get("/model-info")
async def model_info(request: Request) -> dict:
    """Summary of the model card written at training time."""
    return request.state.info


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus text format. A plain route: a mounted app would redirect /metrics to /metrics/."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
