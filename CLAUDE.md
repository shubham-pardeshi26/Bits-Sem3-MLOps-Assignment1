# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

CardioRisk: BITS Pilani AIMLCZG523 MLOps Assignment 1 (spec: `MLOps Assignment 1 Instructions.pdf`).
Heart disease classifier on all four UCI Heart Disease sites, served by FastAPI with Prometheus,
Grafana and Kubernetes. `README.md` is the user-facing guide; `reports/REPORT.md` is the report.

## Commands

Use the venv interpreter (`.venv/bin/python`); `make` (or `make help`) lists every target with the
current variable values.

```bash
make data            # verify data/raw by SHA-256 (no network when it matches), write data/processed/
make train           # ~30 s, deterministic; rewrites models/, reports/metrics.*, reports/figures/model_*
make eda             # reports/figures/eda_*.png
make test            # pytest (39 tests, ~20 s, offline)
make lint            # ruff check . && ruff format --check .  (exactly what CI runs)
make format          # ruff fixes + formatting
make predict         # CLI inference on scripts/sample_request.json
make report          # reports/REPORT.md -> reports/report.docx (editable; reports/reference.docx styles)
make report-pdf      # reports/REPORT.md -> HTML (reports/report.css) -> reports/report.pdf via Chrome
.venv/bin/python -m pytest test/test_api.py::test_predict_returns_a_consistent_risk
.venv/bin/python -m pytest test/test_data.py -k checksum
.venv/bin/python -m cardiorisk.data          # each stage is a module: data, eda, train, predict
.venv/bin/python -m cardiorisk.predict --csv data/processed/heart_disease_clean.csv
.venv/bin/python scripts/architecture_diagram.py   # regenerates reports/figures/architecture.png
```

Serving and infra targets (`serve`, `smoke`, `load`, `docker-build`, `docker-run`, `compose-up`,
`minikube-start`, `k8s-image`, `k8s-deploy`, `k8s-status`, `k8s-forward`, `k8s-delete`) take
`PORT`, `URL`, `PIP_INDEX_URL`, `IMAGE`, `KUBE_CONTEXT`, `OVERLAY`, `MINIKUBE`. Render manifests
offline with `kubectl kustomize deploy/k8s/overlays/<minikube|kind|cloud>`.

## Architecture

- **Data** (`data.py`): four raw site files pinned by `RAW_SHA256`; `clean()` turns impossible
  zeros (`chol`, `trestbps`) into NaN, drops 2 exact duplicates (920 -> 918 rows) and binarises
  `num` into `target`. `site` stays in the table for EDA, stratification and leave-one-site-out,
  but is never a model input (`features.INPUT_FEATURES` is the 13 UCI fields).
- **Pipeline** (`features.build_pipeline`): FunctionTransformer(`add_clinical_features`, adds
  `hr_achieved_frac`) -> ColumnTransformer (imputers with missing indicators, scaler, one-hot with
  NaN as a category) -> estimator. All preprocessing is inside the pickled object, so any change to
  features means retraining and re-baking the image.
- **Training** (`train.py`): 80/20 hold-out and 5 CV folds, both stratified on site x class;
  `CANDIDATES` = majority baseline + 3 families, each a GridSearchCV logged as one MLflow run with
  one nested run per grid point (85 per session). Champion = best mean CV ROC-AUC (hold-out never
  used for selection); registered as `cardiorisk-classifier` with alias `champion`; saved as
  `models/cardiorisk_model.joblib` + `models/model_card.json`. The card's `version` is the MLflow
  registry version, which is what `/health`, `model_version` in responses and
  `cardiorisk_model_info` report. `train()` takes `out_dir`, `tracking_uri` and `candidates` so
  tests run a reduced set into `tmp_path`.
- **Inference**: `predict.predict_records` is the only scoring path (CLI and API). It builds the
  DataFrame with `INPUT_FEATURES` columns, so missing keys become NaN and extra keys are dropped;
  the API's Pydantic `Patient` (`extra="forbid"`) is where bad input is rejected; the 422 handler
  returns only `type`, `loc`, `msg` per error (no `input`/`ctx`, so NaN/Infinity cannot 500 and
  patient values are not echoed). `/health` and `/metrics` access lines log at DEBUG. The threshold
  decision is taken on the rounded probability so responses are self-consistent.
- **Paths**: `config.ROOT` is `CARDIORISK_ROOT` or `parents[2]` of `config.py`: the repo root with
  the editable install (`-e .` in requirements.txt), `/app` in the image (`PYTHONPATH=/app/src`,
  not installed). The joblib pickle references `cardiorisk.features` functions by import path, so
  `cardiorisk` must be importable wherever the model is loaded. The MLflow-logged copy is
  different: skops format with `code_paths`, loadable without the package.
- **Metrics contract**: the metric names and labels in `api.py`
  (`cardiorisk_http_requests_total`, `cardiorisk_http_request_duration_seconds`,
  `cardiorisk_predictions_total`, `cardiorisk_prediction_probability`, `cardiorisk_model_info`),
  the PromQL in `deploy/k8s/monitoring/grafana/cardiorisk-api.json`, the Prometheus job name
  `cardiorisk-api` (k8s `prometheus.yml`, `deploy/compose/prometheus.yml`, the CI target check),
  `test_api.py::test_metrics_expose_the_contract_series` and `scripts/smoke_test.py` must change
  together. `api.SAMPLE_PATIENT` must equal `scripts/sample_request.json` (a test enforces it).
- **Kubernetes** (Kustomize): `deploy/k8s/base` = namespace (PSS restricted), API Deployment
  (no fixed `replicas`: the HPA, 2-5, owns the count), ClusterIP Service, HPA; `monitoring` =
  Prometheus (namespace-scoped pod discovery via annotations) + Grafana, config from hashed
  generators; the Grafana files are also bind-mounted by docker-compose (which has its own
  `deploy/compose/prometheus.yml`); overlays: `minikube` (tag 1.0.0, LoadBalancer on 8081, nginx
  Ingress to the Service port `http`), `kind` (tag `ci`, 1 replica, HPA min 1; used by CI),
  `cloud` (registry image `ghcr.io/OWNER/REPO`, LoadBalancer + Ingress).
- **CI** (`.github/workflows/ci.yml`): lint, test -> train -> docker (build from the `model`
  artifact, smoke) -> deploy-kind (kind overlay, smoke, Prometheus targets up) -> publish (push to
  main only, the same saved image, tags `sha-<7>` and `latest`). Registry comes from repository
  variables `REGISTRY`, `IMAGE_NAME`, `REGISTRY_USERNAME` and secret `REGISTRY_PASSWORD`,
  defaulting to GHCR with `GITHUB_TOKEN`.

## Gotchas

- **Always pin kubectl to `--context minikube`**, as the Makefile does
  (`KUBECTL = kubectl --context $(KUBE_CONTEXT)`), so nothing is applied to another cluster in the
  kubeconfig; keep every new target and every documented command pinned the same way. Offline
  `kubectl kustomize` needs no cluster.
- UCI data ships in `data/raw` (checksums pinned), so nothing needs archive.ics.uci.edu. The API
  is testable without a socket through `fastapi.testclient.TestClient`.
- Default host ports are 8081 (`PORT`: serve, docker-run, compose-up, smoke, load; the container
  still listens on 8000) and 5001 (`MLFLOW_PORT`) because 8000/8080/5000 are often taken. The
  minikube LoadBalancer is http://127.0.0.1:8081 and the Ingress http://127.0.0.1/ while
  `minikube tunnel` runs; the kind overlay (CI) uses a port-forward on 8000.
- On an arm64 Mac an x86 minikube in `/usr/local/bin` can shadow the arm64 Homebrew one
  (`MINIKUBE=`).
- Ruff runs on the whole repo: `ruff check .` also lints `notebooks/*.ipynb`, and
  `ruff format .` also formats Python code fences in `*.md` (so a malformed Python snippet in a
  doc fails CI). Use bash/json/text fences for non-Python snippets. `mlruns/` is excluded because
  MLflow copies `src/cardiorisk` into every logged model.
- Training outputs (`models/`, `reports/metrics.*`, `reports/figures/model_*.png`) are regenerated
  by `make train` and are committed; `mlflow.db` and `mlruns/` are gitignored. Every training run
  bumps the registry version, so the model card `version` changes even though metrics do not.
  README and `reports/REPORT.md` quote the metrics (re-sync them if they change) but never a fixed
  registry version: example responses say `model_version` grows with every `make train`.
- Serving pins live in `requirements-serve.txt` (included by `requirements.txt`); training and the
  image must share scikit-learn/numpy/pandas versions to unpickle the model. `requirements.txt`
  pins `skops==0.16.0` for the MLflow-logged copy.
- The screenshot file names (`screenshots/NN_*.png`) are shared by `reports/REPORT.md`, README and
  `screenshots/README.md` (the checklist, which also says which 5 sit in the report body and which
  12 in its Appendix C); rename in all of them or none. Image widths come from
  `reports/pandoc-figures.lua` by file name, so REPORT.md stays plain Markdown. The report body
  (Sections 1-12) is kept to about 10 A4 pages; rebuild `make report` and `make report-pdf` after
  the screenshots are added or REPORT.md changes.
- Do not `git commit` or `git push` unless the user asks.
- Academic integrity: the spec penalises work that looks copied (feature choices, EDA plots,
  pipelines, Docker/K8s configs, API structure, workflows, reports). Keep code and prose specific
  to this project and its numbers; no boilerplate templates.
