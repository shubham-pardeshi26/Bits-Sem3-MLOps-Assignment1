"""Smoke test a running cardiorisk API (docker, kind, minikube). Stdlib only; exit 1 on failure.

python scripts/smoke_test.py                                 # http://127.0.0.1:8081 (Makefile PORT)
python scripts/smoke_test.py http://127.0.0.1 --requests 200 # minikube Ingress

Proxy settings are honoured like curl's: behind HTTP(S)_PROXY, list the target in NO_PROXY.
"""

import argparse
import http.client
import json
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

SAMPLE = json.loads(Path(__file__).with_name("sample_request.json").read_text())
RESPONSE_KEYS = {"prediction", "label", "confidence", "probability_disease", "model_version"}
LABELS = {0: "no_heart_disease", 1: "heart_disease"}
INVALID = {
    "missing age": {k: v for k, v in SAMPLE.items() if k != "age"},
    "unknown chest pain code": {**SAMPLE, "cp": 5},
    "cholesterol out of range": {**SAMPLE, "chol": 2000},
    "unknown field": {**SAMPLE, "bmi": 31.2},
}
failures = []


def call(base, path, payload=None, headers=None):
    """(status, headers, body) without raising on HTTP errors; JSON bodies are decoded."""
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json", **(headers or {})}
    request = urllib.request.Request(base + path, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=10) as resp:
            status, resp_headers, body = resp.status, resp.headers, resp.read()
    except urllib.error.HTTPError as err:
        status, resp_headers, body = err.code, err.headers, err.read()
    is_json = resp_headers.get("Content-Type", "").startswith("application/json")
    return status, resp_headers, json.loads(body) if is_json else body.decode()


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f": {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def wait_for_health(base, timeout=60):
    deadline = time.monotonic() + timeout
    while True:
        try:
            status, _, body = call(base, "/health")
            if status == 200:
                return body
        except (OSError, http.client.HTTPException):
            pass  # not listening yet
        if time.monotonic() > deadline:
            return None
        time.sleep(2)


def prediction_problems(body):
    """Contract violations in a /predict response body (empty list = valid)."""
    if not isinstance(body, dict) or set(body) != RESPONSE_KEYS:
        return [f"unexpected body {body!r}"]
    pred, p, conf = body["prediction"], body["probability_disease"], body["confidence"]
    if type(pred) is not int or not all(isinstance(v, float) for v in (p, conf)):
        return [f"wrong types {body!r}"]
    rules = {
        "prediction is 0 or 1": pred in (0, 1),
        "probability_disease in [0, 1]": 0 <= p <= 1,
        "prediction == (probability_disease >= 0.5)": pred == int(p >= 0.5),
        "label matches prediction": body["label"] == LABELS.get(pred),
        "confidence == max(p, 1 - p)": abs(conf - max(p, 1 - p)) < 1e-4,
        "model_version is set": isinstance(body["model_version"], str) and body["model_version"],
    }
    return [rule for rule, ok in rules.items() if not ok]


def random_patient(rng):
    """A plausible patient within the UCI ranges. Some optional measurements are left out, as at
    the sites that never recorded them."""
    patient = {
        "age": rng.randint(29, 77),
        "sex": rng.randint(0, 1),
        "cp": rng.randint(1, 4),
        "trestbps": rng.randint(94, 200),
        "chol": rng.randint(126, 564),
        "fbs": rng.randint(0, 1),
        "restecg": rng.randint(0, 2),
        "thalach": rng.randint(71, 202),
        "exang": rng.randint(0, 1),
        "oldpeak": round(min(rng.expovariate(1.0), 6.2), 1),
        "slope": rng.randint(1, 3),
        "ca": rng.randint(0, 3),
        "thal": rng.choice([3, 6, 7]),
    }
    for field in ("chol", "slope", "ca", "thal"):
        if rng.random() < 0.3:
            del patient[field]
    return patient


def send_traffic(base, n):
    """n varied /predict calls, every 10th deliberately invalid, to feed the dashboards."""
    rng = random.Random()
    statuses, labels, latencies, wrong = Counter(), Counter(), [], 0
    for i in range(n):
        invalid = i % 10 == 9
        payload = rng.choice(list(INVALID.values())) if invalid else random_patient(rng)
        start = time.perf_counter()
        status, _, body = call(base, "/predict", payload)
        latencies.append((time.perf_counter() - start) * 1000)
        statuses[status] += 1
        if invalid:
            ok = status == 422
        else:
            ok = status == 200 and not prediction_problems(body)
            if ok:
                labels[body["label"]] += 1
        if not ok:
            wrong += 1
    by_status = ", ".join(f"HTTP {code} x{count}" for code, count in sorted(statuses.items()))
    by_label = ", ".join(f"{label} {count}" for label, count in sorted(labels.items()))
    summary = (
        f"{n} requests ({by_status}; {by_label or 'no predictions'}); "
        f"latency median {statistics.median(latencies):.1f} ms, max {max(latencies):.1f} ms"
    )
    check("traffic", wrong == 0, summary + (f"; {wrong} unexpected responses" if wrong else ""))


def main():
    parser = argparse.ArgumentParser(description="Smoke test a running cardiorisk API.")
    parser.add_argument("base_url", nargs="?", default="http://127.0.0.1:8081")
    parser.add_argument(
        "--requests",
        type=int,
        default=0,
        metavar="N",
        help="also send N varied patients (about 10%% invalid) to generate monitoring traffic",
    )
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    print(f"Smoke testing {base}")

    health = wait_for_health(base)
    if not isinstance(health, dict) or health.get("status") != "ok":
        check("health", False, f"no healthy /health response within 60 s (last: {health})")
        return 1
    check("health", True, f"model_version {health.get('model_version')}")

    try:
        status, headers, body = call(base, "/predict", SAMPLE, {"X-Request-ID": "smoke-sample"})
        problems = prediction_problems(body) if status == 200 else [f"HTTP {status}: {body}"]
        detail = "; ".join(problems) or f"{body['label']}, p={body['probability_disease']}"
        check("predict sample patient", not problems, detail)
        check("X-Request-ID echoed", headers.get("X-Request-ID") == "smoke-sample")

        for name, payload in INVALID.items():
            status, _, _ = call(base, "/predict", payload)
            check(f"422 for {name}", status == 422, f"HTTP {status}")

        status, _, text = call(base, "/metrics")
        check(
            "metrics expose cardiorisk_predictions_total",
            status == 200 and "cardiorisk_predictions_total" in text,
        )
        if args.requests > 0:
            send_traffic(base, args.requests)
    except (OSError, http.client.HTTPException) as exc:
        check("connection", False, repr(exc))

    print("SMOKE TEST PASSED" if not failures else f"SMOKE TEST FAILED: {', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
