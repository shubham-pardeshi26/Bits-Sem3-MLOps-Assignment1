# CardioRisk: heart disease risk, from UCI data to a monitored Kubernetes API

[![CI/CD](https://github.com/shubham-pardeshi26/Bits-Sem3-MLOps-Assignment1/actions/workflows/ci.yml/badge.svg)](https://github.com/shubham-pardeshi26/Bits-Sem3-MLOps-Assignment1/actions/workflows/ci.yml)

CardioRisk predicts the presence of heart disease from 13 routine clinical measurements. It is
trained on all four hospitals of the UCI Heart Disease dataset (918 patients after cleaning), not
only the usual Cleveland subset, and served as a FastAPI service with Prometheus metrics, JSON logs,
a Grafana dashboard and Kubernetes manifests. One GitHub Actions workflow lints, tests, trains,
builds, deploys the image to a throwaway kind cluster and publishes it.

BITS Pilani AIMLCZG523 (MLOps), Assignment 1. Full write-up: [reports/REPORT.md](reports/REPORT.md)
(also as [reports/report.docx](reports/report.docx) and reports/report.pdf); evidence:
[screenshots/](screenshots/). Slide walkthrough of the whole pipeline:
[presentation](https://shubham-pardeshi26.github.io/Bits-Sem3-MLOps-Assignment1/presentation/)
(source: [presentation/index.html](presentation/index.html)).

## Contents

- [Results at a glance](#results-at-a-glance)
- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [Quickstart](#quickstart)
- [Using the API](#using-the-api)
- [Docker](#docker)
- [Local monitoring stack (docker compose)](#local-monitoring-stack-docker-compose)
- [Kubernetes on minikube](#kubernetes-on-minikube)
- [CI/CD](#cicd)
- [Switching registry or cloud](#switching-registry-or-cloud)
- [Monitoring and logging](#monitoring-and-logging)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Assignment deliverables](#assignment-deliverables)

## Results at a glance

734 training and 184 hold-out patients (80/20 split stratified on site x class, because
prevalence ranges from 36% in Hungary to 93% in Switzerland, which has only 8 patients without
disease). CV is 5 folds of the training split, stratified the same way; the champion is the family
with the best mean CV ROC-AUC, which is threshold-free and insensitive to prevalence, so the
hold-out never influences selection. Metrics at threshold 0.5. Full table with CV accuracy,
precision, recall and F1: [reports/metrics.md](reports/metrics.md).

| Family | CV ROC-AUC | Test ROC-AUC | Test accuracy | Test precision | Test recall | Test F1 |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_majority | 0.500 ± 0.000 | 0.500 | 0.554 | 0.554 | 1.000 | 0.713 |
| **logistic_regression** (champion) | **0.907 ± 0.029** | 0.906 | 0.842 | 0.829 | 0.902 | 0.864 |
| random_forest | 0.897 ± 0.023 | 0.896 | 0.853 | 0.844 | 0.902 | 0.872 |
| hist_gradient_boosting | 0.895 ± 0.026 | 0.900 | 0.853 | 0.844 | 0.902 | 0.872 |

The champion is an elastic-net logistic regression (`C=0.3`, `l1_ratio=0.5`). The three real
models are within one CV standard deviation of each other; the tree ensembles classify two more of
the 184 hold-out patients correctly, which is noise at this sample size, and selection deliberately
ignores the hold-out. The linear model is also the cheapest to serve and the easiest to explain.

**Leave-one-site-out** (champion hyper-parameters, trained on three hospitals, scored on the
fourth):

| Held-out site | Patients | Disease | ROC-AUC | Accuracy | Recall |
| --- | --- | --- | --- | --- | --- |
| Cleveland | 303 | 46% | 0.876 | 0.762 | 0.878 |
| Hungary | 293 | 36% | 0.902 | 0.795 | 0.906 |
| Switzerland | 123 | 93% | 0.749 | 0.772 | 0.791 |
| VA Long Beach | 199 | 74% | 0.730 | 0.759 | 0.831 |

The random hold-out (0.906) overstates how well the model transfers to a new hospital. Cleveland
and Hungary stay near it; Switzerland and VA drop to 0.75 and 0.73. Those are the sites with the
highest prevalence and the most gaps beyond `ca`/`thal` (Switzerland never recorded `chol`; VA lacks
`trestbps`, `thalach` and `oldpeak` for about 28% of patients), and Switzerland's ROC-AUC rests on
8 negatives. `site` is kept in the data for this analysis and for the EDA but is never a model
input: on its own it ranks patients with ROC-AUC 0.72 (EDA notebook), which is referral pattern,
not physiology, and a deployed model sees a patient, not a hospital label.

## Architecture

![CardioRisk architecture](reports/figures/architecture.png)

- **Data and training.** `cardiorisk.data` verifies the four raw UCI files against pinned SHA-256
  digests (downloading only when a file is missing or differs), cleans them and writes
  `data/processed/heart_disease_clean.csv`. `cardiorisk.train` grid-searches four model families
  around one scikit-learn pipeline, logs every candidate to MLflow and registers the champion as
  `cardiorisk-classifier@champion`.
- **One artifact, one inference path.** The feature step (`hr_achieved_frac = thalach / (220 -
  age)`), imputation, scaling and one-hot encoding live inside the pickled pipeline
  `models/cardiorisk_model.joblib`, next to `models/model_card.json`. The CLI and the API both
  score through `cardiorisk.predict.predict_records`, so serving cannot drift from training. The
  API loads this file, not the registry, so the image needs neither MLflow nor a tracking server;
  the registered copy (skops format, package source attached as `code_paths`) is the audit trail
  and loads without `cardiorisk` installed.
- **Serving.** The Docker image holds only the serving pins, the package and the model. Kubernetes
  runs it as a Deployment (2-5 pods under an HPA) behind a LoadBalancer Service and an nginx
  Ingress, with Prometheus discovering the pods and Grafana showing a provisioned dashboard.
- **CI/CD.** GitHub Actions retrains the model on every run, builds the image from that model,
  smoke-tests it in Docker and in kind, and pushes exactly that image to GHCR from `main`.

## Repository layout

```text
src/cardiorisk/        the package: config, data, eda, features, train, predict, api
test/                  pytest suite (offline: synthetic fixtures, throwaway MLflow store)
notebooks/01_eda.ipynb EDA narrative built on cardiorisk.eda, outputs saved
scripts/               smoke_test.py (stdlib only), sample_request.json, architecture_diagram.py
data/raw/              the four UCI site files and the codebook, SHA-256 pinned in data.py
data/processed/        heart_disease_clean.csv, 918 rows (written by make data)
models/                cardiorisk_model.joblib + model_card.json: the served model
reports/               metrics.md/.json, figures/ (EDA, model, architecture), REPORT.md, report.docx/.pdf
deploy/k8s/            base/ (API), monitoring/ (Prometheus, Grafana), overlays/ (minikube, kind, cloud)
deploy/compose/        Prometheus config for docker compose
screenshots/           report evidence; checklist in screenshots/README.md
.github/workflows/     ci.yml: lint, test -> train -> docker -> deploy-kind -> publish
Makefile               one target per stage; `make` lists them
Dockerfile, docker-compose.yml
requirements.txt       full environment (training, MLflow, notebooks, tests, lint)
requirements-serve.txt API runtime pins only; included by requirements.txt
```

`mlflow.db` and `mlruns/` (the local tracking store) are gitignored and recreated by
`make train`; CI uploads its own copy as the `mlflow-tracking` artifact.

## Quickstart

**Prerequisites:** Python 3.12, `make`, git. For the container and cluster parts: a Docker engine,
minikube and kubectl. Everything below runs from the repository root.

```bash
make setup        # .venv with requirements.txt (includes `-e .`, the cardiorisk package)
make data         # verify data/raw by SHA-256, write data/processed/heart_disease_clean.csv
make eda          # reports/figures/eda_*.png
make notebook     # re-execute notebooks/01_eda.ipynb in place (outputs saved)
make train        # grid search, MLflow tracking, champion -> models/ and reports/ (~30 s)
make mlflow-ui    # MLflow UI on http://127.0.0.1:5001
make test         # pytest
make lint         # ruff check + ruff format --check, as in CI
```

Notes on each stage:

- **`make setup`** uses `python3.12`; pass `PYTHON=/path/to/python3.12` if it is named
  differently.
- **`make data`** needs no network: the raw files are committed and their digests match, so the
  download is skipped. If a file is missing or altered it downloads the UCI archive, and it refuses
  to write anything whose digest differs from the pinned one (a silently changed upstream file
  would shift every number in this README). Where `archive.ics.uci.edu` is blocked, download
  `heart+disease.zip` elsewhere and extract the four `processed.*.data` files and
  `heart-disease.names` into `data/raw/`; the error message says the same. Cleaning turns `?` and
  the impossible zeros in `chol` and `trestbps` (Switzerland's and VA's code for "not measured")
  into missing values, drops 2 exact duplicates (920 -> 918 rows) and maps `num` 1-4 to
  `target = 1`, because Hungary only coded 0/1. Rows with gaps are kept: outside Cleveland, fewer
  than 1% of patients are complete. It prints rows, prevalence and missing share per site.
- **`make eda`** writes six figures; the notebook explains what each one implies for the pipeline.
  To work in it interactively: `.venv/bin/python -m jupyter lab notebooks/01_eda.ipynb`.
- **`make train`** is deterministic (seed 42): the metrics above reproduce exactly. Each run adds
  one MLflow session (4 family runs, 85 nested grid runs), registers a new version of
  `cardiorisk-classifier`, moves the `@champion` alias to it and overwrites `models/`,
  `reports/metrics.*` and `reports/figures/model_*.png`. The model card's `version` is that
  registry version, so it increases with every run.
- **`make mlflow-ui`**: experiment `cardiorisk-heart-disease`. Each family run has its CV and test
  metrics, the tuned and fixed parameters, `plots/` (ROC, precision-recall, confusion matrix,
  permutation importance; LOSO for the champion), `cv_results.csv`, a classification report and the
  model itself; each nested grid run also logs `cv_folds.json` with its per-fold test scores. The
  UI uses port 5001 because macOS AirPlay Receiver holds MLflow's default 5000.
- **`make test`**: 39 tests, about 20 s, no network. They use synthetic patients and a temporary
  MLflow store; one quality gate (logistic regression CV ROC-AUC >= 0.80) runs on the real cleaned
  CSV and is skipped if `make data` has not run.
- **`make format`** applies ruff's fixes. `ruff check` also lints the notebook, and
  `ruff format` also covers Python code blocks in Markdown files.

## Using the API

```bash
make serve                         # uvicorn with reload on http://127.0.0.1:8081
make serve PORT=8090               # if 8081 is taken
make predict                       # the same model from the CLI, no server
```

Score the sample patient (`scripts/sample_request.json`, a 61-year-old man with asymptomatic chest
pain, exercise-induced angina, two vessels coloured on fluoroscopy and a reversible thallium
defect):

```bash
curl -s -X POST http://127.0.0.1:8081/predict \
  -H 'Content-Type: application/json' -d @scripts/sample_request.json
```

```json
{
  "prediction": 1,
  "label": "heart_disease",
  "confidence": 0.9851,
  "probability_disease": 0.9851,
  "model_version": "3"
}
```

`prediction` is 1 when `probability_disease >= 0.5`; `confidence` is the probability of the returned
class, so `{"age": 45, "sex": 0, "cp": 2}` returns `prediction 0`, `probability_disease 0.3477`,
`confidence 0.6523`. `model_version` is the MLflow registry version of the served model (from
`models/model_card.json`): it grows by one with every `make train` (a fresh clone or CI starts at
`"1"`), so yours may differ from the `"3"` above. The decision is taken on the rounded
probability, so a response can never show 0.5 next to class 0.

**Validation** (Pydantic). Invalid input gets HTTP 422 with one entry per problem: `type`, `loc`
and `msg` only, so the patient's values are never echoed back. For `{"age": 61, "sex": 1, "cp": 5,
"bmi": 30}`:

```json
{
  "detail": [
    {"type": "literal_error", "loc": ["body", "cp"], "msg": "Input should be 1, 2, 3 or 4"},
    {"type": "extra_forbidden", "loc": ["body", "bmi"], "msg": "Extra inputs are not permitted"}
  ]
}
```

- `age` (18-100), `sex` (0/1) and `cp` (1-4) are required. The other ten fields are optional and may
  be omitted or `null`: the pipeline imputes them, exactly as it does for the hospitals that never
  recorded `ca`, `thal` or `chol`.
- Codes must be valid UCI codes (`thal` in 3/6/7, `slope` 1-3, `ca` 0-3, ...) and measurements must
  be in physiological ranges (`chol` 80-700 mg/dl, `trestbps` 60-250 mm Hg, `thalach` 50-230 bpm,
  `oldpeak` -3 to 7 mm). JSON `NaN` and `Infinity` fail these range checks, so they also get 422.
- Unknown fields are rejected (`extra_forbidden`), so a misspelt `thalac` fails loudly instead of
  being imputed silently.

**Other endpoints**

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | `{"status": "ok", "model_version": "<registry version>"}`; liveness, readiness and Docker health check |
| `GET /docs` | Swagger UI; `/predict` is prefilled with the sample patient (`/` redirects here) |
| `GET /model-info` | the model card summary: version, family, trained_at, threshold, CV/test/LOSO metrics, features, MLflow run id |
| `GET /metrics` | Prometheus text format (see [Monitoring and logging](#monitoring-and-logging)) |

Every response carries `X-Request-ID` (the caller's value, or a generated one), which also appears
in the log lines of that request. The model loads once at startup; if it cannot, the process exits
with the path in the error instead of starting unhealthy.

## Docker

```bash
make docker-build                  # cardiorisk-api:1.0.0 with models/ baked in (train first)
make docker-run                    # foreground on http://127.0.0.1:8081, Ctrl-C removes it
make smoke                         # in a second terminal
```

- The image is `python:3.12-slim` with `requirements-serve.txt` only: no MLflow, Jupyter or
  training stack. The `.dockerignore` is an allowlist (`requirements-serve.txt`, `src/`,
  `models/`), so data, the tracking store and `.venv` never enter the build context. The pip layer
  is rebuilt only when the pins change.
- The app runs as uid 10001 with code and model root-owned, the same uid the Kubernetes
  Deployment's `runAsUser` names (the `restricted` Pod Security Standard rejects root). The build
  fails with a clear message if `models/cardiorisk_model.joblib` is missing. A `HEALTHCHECK` polls
  `/health`.
- `make smoke` (`scripts/smoke_test.py`, stdlib only) waits up to 60 s for `/health`, checks the
  `/predict` contract on the sample patient (keys, types, `prediction == (p >= 0.5)`,
  `confidence == max(p, 1 - p)`), the `X-Request-ID` echo, four invalid payloads that must return
  422 (missing `age`, `cp: 5`, `chol: 2000`, unknown field `bmi`) and that `/metrics` exposes
  `cardiorisk_predictions_total`. It exits 1 on any failure; CI runs the same script.

Knobs: `PORT=<free port>` for `docker-run` and `smoke` if 8081 is taken (the container itself
listens on 8000); `PIP_INDEX_URL=<mirror>/simple` for `docker-build` (and `k8s-image`,
`compose-up`) when the Docker VM cannot reach pypi.org; `IMAGE=<name:tag>` to change the tag.

## Local monitoring stack (docker compose)

```bash
make compose-up                    # API + Prometheus + Grafana in the background
make load                          # smoke test + 300 varied requests (every 10th invalid)
make compose-down
```

| Service | URL |
| --- | --- |
| API | http://127.0.0.1:8081/docs (pass the same `PORT=` to `compose-up` and `load` if 8081 is taken) |
| Prometheus | http://127.0.0.1:9090/targets (job `cardiorisk-api`, target `api:8000`) |
| Grafana | http://127.0.0.1:3000: the CardioRisk API dashboard is the home page; anonymous users can view, `admin` signs in with `cardiorisk-demo` (override with `GRAFANA_ADMIN_PASSWORD`) |

Compose mounts the same Grafana provisioning files and dashboard JSON as Kubernetes, and both
Prometheus configs scrape every 5 s under the job name `cardiorisk-api`, so the dashboard behaves
identically in both stacks. All ports bind to 127.0.0.1 only.

## Kubernetes on minikube

> **Every kubectl call is pinned to `--context minikube`.** The Makefile runs
> `kubectl --context $(KUBE_CONTEXT)` with `KUBE_CONTEXT=minikube`, and `make k8s-deploy` refuses
> to run when that context does not exist. An unpinned `kubectl apply -k` would create this
> namespace, RBAC and workloads in whatever cluster your current context points at (a shared or
> production one, if that is your default). Type any manual command with the context too, for
> example `kubectl --context minikube -n cardiorisk get pods`.

```bash
make minikube-start                # docker driver + ingress and metrics-server addons
make k8s-image                     # build cardiorisk-api:1.0.0 inside minikube (no registry)
make k8s-deploy                    # apply deploy/k8s/overlays/minikube, wait for 3 rollouts
make k8s-status                    # pods, services, HPA, ingress in namespace cardiorisk

minikube tunnel                    # separate terminal, keep it running (asks for sudo)
make smoke                         # through the LoadBalancer Service: http://127.0.0.1:8081
make smoke URL=http://127.0.0.1    # through the Ingress (nginx, no host rule)
make load URL=http://127.0.0.1     # traffic for the dashboard

make k8s-forward                   # Grafana http://127.0.0.1:3000, Prometheus http://127.0.0.1:9090
make k8s-delete                    # remove everything the overlay created, namespace included
```

What the minikube overlay deploys into namespace `cardiorisk` (Pod Security Standard `restricted`
enforced):

- **API:** Deployment `cardiorisk-api` (no fixed replica count: the HPA keeps 2-5 at 70% CPU;
  rolling update with `maxUnavailable: 0`, startup/readiness/liveness probes on `/health`,
  read-only root filesystem, all capabilities dropped), Service `cardiorisk-api` of type
  LoadBalancer on port 8081 (target port `http`, 8000 in the pod), Ingress class `nginx` on `/`
  pointing at the Service port by name. At least two replicas with `maxUnavailable: 0` keep
  `/predict` answering through a rollout or a pod restart. The CPU request is only 100m so that
  the HPA reacts to demo-sized load; the 512Mi memory limit is about twice the ~205 MiB peak
  measured with the model loaded. The Service and the Ingress are two ways in to the same pods:
  the spec accepts either, and the steps above smoke-test both.
- **Monitoring:** Prometheus (Kubernetes pod discovery limited to this namespace by a Role, no
  ClusterRole) and Grafana, both ClusterIP and reached through `make k8s-forward`. Their config is
  generated by Kustomize with a content hash in each ConfigMap name, so editing the dashboard JSON
  or the scrape config rolls exactly the pods that mount it.

The image is built inside the minikube node, so the cluster never pulls it
(`imagePullPolicy: IfNotPresent`). After changing code or retraining, rerun `make k8s-image` and
restart the pods: `kubectl --context minikube -n cardiorisk rollout restart deployment/cardiorisk-api`.

When done: `make k8s-delete`, `minikube stop`, then switch your kubeconfig back (see
[Troubleshooting](#troubleshooting)).

## CI/CD

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on pull requests, pushes to `main` and
manual dispatch; a newer run on the same ref cancels the older one. Every `run` step uses bash with
`-eo pipefail`, so a failing command fails the job even when its output is piped through `tee`.

| Job | Needs | What it proves | Evidence on the run page |
| --- | --- | --- | --- |
| `lint` | | `ruff check` (as inline annotations) and `ruff format --check` with the pinned ruff; both run even if the first fails | annotations |
| `test` | | the 39 unit tests pass on a clean `pip install -r requirements.txt` | summary with the result line; artifact `test-results` (junit.xml, pytest.log) |
| `train` | lint, test | the raw data verifies and the full training run completes from scratch | summary with the model comparison table; artifacts `model`, `training-report` (metrics, model figures, train.log), `mlflow-tracking` (mlflow.db + mlruns/) |
| `docker` | train | the image builds with the model trained in this run and serves it: smoke test plus 20 varied requests | container logs; artifact `image` (the tested image as a gzipped archive, kept 1 day) |
| `deploy-kind` | docker | the Kustomize manifests work on a real cluster: the kind overlay applies, all three rollouts finish, the smoke test passes through the Service, and Prometheus reports every `cardiorisk-api` pod target `up` within 2 minutes | summary with the Prometheus targets; events, pod descriptions and logs on failure |
| `publish` | deploy-kind | only on a push to `main`: pushes the image archive that passed both smoke tests, tagged `sha-<7>` and `latest` | summary with the pushed tags |

Design points:

- **Nothing is rebuilt after testing.** `docker` saves the image it smoke-tested; `deploy-kind`
  loads that archive into kind (`kind load image-archive`) and `publish` pushes it. The image in
  the registry is byte-for-byte the one that passed.
- **The kind job validates the manifests, not a mock.** It applies `deploy/k8s/overlays/kind`,
  which reuses `base/` and `monitoring/` unchanged and only sets the image tag `ci`, one replica
  and HPA `minReplicas: 1` (kind has no metrics-server). Its kubectl calls carry no `--context`
  because the runner's only context is that ephemeral cluster.
- **The training report and the tracking store are uploaded even when training fails**, so a
  broken run still leaves `train.log` and `mlflow.db` for debugging. Test results are uploaded
  whatever the outcome too.
- The published image is `ghcr.io/<owner>/<repo>` in lower case, labelled with its source
  repository so GHCR links the package to it. It needs no secret: `GITHUB_TOKEN` gets
  `packages: write` in that job only.

## Switching registry or cloud

The workflow reads the registry from repository settings (Settings > Secrets and variables >
Actions); no edit to `ci.yml` is needed.

| Name | Kind | Default when unset |
| --- | --- | --- |
| `REGISTRY` | variable | `ghcr.io` |
| `IMAGE_NAME` | variable | `github.repository` (`OWNER/REPO`) |
| `REGISTRY_USERNAME` | variable | `github.actor` |
| `REGISTRY_PASSWORD` | secret | `GITHUB_TOKEN` |

| Registry | `REGISTRY` | `IMAGE_NAME` | `REGISTRY_USERNAME` / `REGISTRY_PASSWORD` |
| --- | --- | --- | --- |
| Docker Hub | `docker.io` | `<user>/cardiorisk-api` | user / access token with write scope |
| AWS ECR | `<account>.dkr.ecr.<region>.amazonaws.com` | `cardiorisk-api` | IAM access key id / secret key (create the repository first) |
| Google Artifact Registry | `<region>-docker.pkg.dev` | `<project>/<repository>/cardiorisk-api` | `_json_key` / service account JSON key |
| Azure ACR | `<name>.azurecr.io` | `cardiorisk-api` | service principal client id / client secret |

**Managed cluster (`deploy/k8s/overlays/cloud`).** Base and monitoring stay unchanged; everything
provider-specific is in this overlay:

1. In `kustomization.yaml`, set `images[0].newName` to the published image (replace
   `ghcr.io/OWNER/REPO`) and prefer the immutable `sha-<7>` tag over `latest`: re-applying an
   unchanged `latest` rolls nothing, and nodes that cached it keep the old image.
2. In `ingress.yaml`, pick the Ingress class: GKE uses the annotation
   `kubernetes.io/ingress.class: gce`; EKS `ingressClassName: alb` with the AWS Load Balancer
   Controller annotations (`healthcheck-path: /health`, because `/` answers 307); AKS
   `webapprouting.kubernetes.azure.com`. Add a host and a `tls` section for a real domain.
3. Image access: if the GHCR package is private (check Package settings > visibility), make it
   public or add an image pull secret (`regcred`) and a Deployment patch setting
   `imagePullSecrets`. ECR on EKS, GAR on GKE and ACR on AKS (`az aks update --attach-acr`) pull
   with the node identity.
4. Replace the demo Grafana password (the `grafana-admin` secretGenerator, `behavior: replace`)
   before exposing Grafana beyond a port-forward.
5. Deploy with the context of that cluster named explicitly (never rely on the current context):
   `make k8s-deploy OVERLAY=cloud KUBE_CONTEXT=<your-cluster-context>`.

The cloud overlay also switches the Service to LoadBalancer next to the Ingress; drop one of the two
if the provider charges per load balancer.

## Monitoring and logging

**Metrics** (`GET /metrics`; the names are a contract shared by `api.py`, the Grafana dashboard,
the tests and the CI Prometheus check):

| Metric | Type | Labels | Meaning |
| --- | --- | --- | --- |
| `cardiorisk_http_requests_total` | counter | method, route, status | requests; `route` is the route template (`unmatched` for 404s), so cardinality stays bounded |
| `cardiorisk_http_request_duration_seconds` | histogram | method, route | request latency |
| `cardiorisk_predictions_total` | counter | label | predictions per returned label, both exported at 0 from startup so `rate()` counts the first one |
| `cardiorisk_prediction_probability` | histogram | | predicted P(disease) in 0.1 buckets; a shift hints at input drift |
| `cardiorisk_model_info` | gauge | version, family | the served model (always 1) |

**Grafana dashboard "CardioRisk API"** (folder CardioRisk; provisioned from
`deploy/k8s/monitoring/grafana/cardiorisk-api.json`, read-only in the UI, so change the JSON):
API pods up, model version, heart-disease share of predictions, request rate by route, error share
(4xx vs 5xx, excluding `/health` and `/metrics`), `/predict` latency p50/p95/p99, memory per API
pod, predictions per minute by label, and the distribution of predicted probabilities.

**Logs.** stdout carries one JSON object per line: `model_loaded` at startup, then a `request`
line for every request and a `prediction` line for every `/predict`, linked by `request_id`.
Requests to `/health` and `/metrics` (probes and scrapes, every few seconds per pod) are logged
at DEBUG, so at the default `LOG_LEVEL=INFO` the log shows real traffic only; Prometheus
still counts them. `model_version` is the registry version (see [Using the API](#using-the-api)):

```json
{"ts": "2026-09-29T21:22:36.404+00:00", "level": "INFO", "event": "prediction", "request_id": "956c15d6694c44d484669cda22e1fd15", "label": "no_heart_disease", "probability_disease": 0.3477, "model_version": "3"}
{"ts": "2026-09-29T21:22:36.405+00:00", "level": "INFO", "event": "request", "request_id": "956c15d6694c44d484669cda22e1fd15", "method": "POST", "route": "/predict", "status": 200, "duration_ms": 5.33}
```

Patient features are health data and are never logged; a unit test asserts that no feature name
appears in the output. Uvicorn's plain-text access log is switched off so stdout stays pure JSON.
On Kubernetes: `kubectl --context minikube -n cardiorisk logs deployment/cardiorisk-api --tail=20`.

## Configuration

| Variable | Used by | Default |
| --- | --- | --- |
| `MODEL_PATH`, `MODEL_CARD_PATH` | API, read at startup | `models/cardiorisk_model.joblib`, `models/model_card.json` under the project root |
| `LOG_LEVEL` | API | `INFO` (`DEBUG` adds the `/health` and `/metrics` access lines) |
| `CARDIORISK_ROOT` | every stage: base for data/, models/, reports/ | the repository root (`/app` in the image) |
| `MLFLOW_TRACKING_URI` | training, `make mlflow-ui` | `sqlite:///<root>/mlflow.db` |
| `GRAFANA_ADMIN_PASSWORD` | docker compose | `cardiorisk-demo` |

Make variables (`make help` prints the current values): `PORT` (8081; the host port, the
container listens on 8000), `URL` (`http://127.0.0.1:$(PORT)`), `MLFLOW_PORT` (5001),
`PIP_INDEX_URL` (`https://pypi.org/simple`), `IMAGE` (`cardiorisk-api:1.0.0`), `KUBE_CONTEXT`
(`minikube`), `OVERLAY` (`minikube`), `MINIKUBE` (`minikube`), `PYTHON` (`python3.12`), `PY`
(`.venv/bin/python`).

## Troubleshooting

- **Port 8081 already in use.** The defaults avoid 8000 and 8080, which local servers often hold.
  Find the owner with `lsof -nP -iTCP:8081 -sTCP:LISTEN` and pick another host port for the local
  targets: `make serve PORT=8090`, `make docker-run PORT=8090` with `make smoke PORT=8090`,
  `make compose-up PORT=8090` with `make load PORT=8090`. If the minikube LoadBalancer cannot take
  127.0.0.1:8081 either, test through the Ingress (`make smoke URL=http://127.0.0.1`) or run
  `minikube service -n cardiorisk cardiorisk-api --url` in its own terminal (with the docker
  driver on macOS it keeps a tunnel open until Ctrl-C) and pass the printed URL:
  `make smoke URL=<printed URL>`.
- **pip cannot reach pypi.org inside the image build.** The Docker VM does not inherit the host's
  pip configuration, so pass the index your host uses (`pip config list` shows it):
  `make docker-build PIP_INDEX_URL=https://<mirror>/simple` (same for `k8s-image` and
  `compose-up`).
- **minikube warns that it is the amd64 binary on an M1 system** (or the node runs under
  emulation). An x86 minikube in `/usr/local/bin` shadows the arm64 Homebrew one in
  `/opt/homebrew/bin`. Run `which -a minikube`, then either fix `PATH` or pass the binary:
  `make minikube-start MINIKUBE=/opt/homebrew/bin/minikube` (and the same `MINIKUBE=` for
  `k8s-image`).
- **`make k8s-deploy` fails with `failed calling webhook "validate.nginx.ingress.kubernetes.io"`.**
  The ingress addon's admission webhook is not ready yet right after `make minikube-start`. Wait a
  minute and rerun `make k8s-deploy`; `apply` is idempotent.
- **Your other kubectl work now talks to minikube.** `minikube start` makes `minikube` the current
  context. The Makefile does not depend on the current context, so switch back when done:
  `kubectl config get-contexts`, then `kubectl config use-context <your usual context>` (both
  only read or edit the local kubeconfig; neither talks to a cluster).
- **`make smoke` fails behind a proxy.** The smoke test honours `HTTP(S)_PROXY` like curl does;
  add the target host to `NO_PROXY`, for example `NO_PROXY=127.0.0.1,localhost`.
- **The MLflow UI does not start on port 5001.** Something else listens there (check with
  `lsof -nP -iTCP:5001 -sTCP:LISTEN`): `make mlflow-ui MLFLOW_PORT=5002`. Avoid 5000, which macOS
  AirPlay Receiver holds.
- **`make k8s-forward` fails to bind 3000 or 9090.** The compose stack still holds them:
  `make compose-down` first.

## Assignment deliverables

| Spec item | Where |
| --- | --- |
| Code, Dockerfile, requirements | `src/cardiorisk/`, `Dockerfile`, `requirements.txt`, `requirements-serve.txt` |
| Cleaned dataset and download script | `data/processed/heart_disease_clean.csv`, `make data` (`cardiorisk.data`) |
| Notebooks and scripts for EDA, training, inference | `notebooks/01_eda.ipynb`, `cardiorisk.eda`, `cardiorisk.train`, `cardiorisk.predict` |
| Unit tests | `test/` |
| GitHub Actions workflow | `.github/workflows/ci.yml` |
| Deployment manifests | `deploy/k8s/` (Kustomize), `docker-compose.yml` |
| Screenshots | `screenshots/` (checklist: [screenshots/README.md](screenshots/README.md)) |
| Report | [reports/REPORT.md](reports/REPORT.md); `make report` builds the editable [reports/report.docx](reports/report.docx), `make report-pdf` the final reports/report.pdf (needs Google Chrome); rebuild both after adding the screenshots |
| Link to code repository | <https://github.com/shubham-pardeshi26/Bits-Sem3-MLOps-Assignment1> |
| API access | local: [Using the API](#using-the-api), [Docker](#docker), [Kubernetes on minikube](#kubernetes-on-minikube); no public deployment |
| Pipeline video | recorded separately; add its link here |

**Screenshot evidence** (captured by running the pipeline once; how to reach each view is in
[screenshots/README.md](screenshots/README.md)):

| File | Shows |
| --- | --- |
| `01_mlflow_experiment_runs.png` | MLflow experiment `cardiorisk-heart-disease`: the four family runs with key metric columns |
| `02_mlflow_champion_run-main-page.png` | the champion run: metrics and parameters |
| `02_mlflow_champion_run-artifact-plots.png` | the champion run: `plots/` artifacts (leave-one-site-out chart), `cv_results.csv` |
| `03_mlflow_compare_runs.png` | parallel-coordinates compare view of the nested grid runs of one family |
| `04_mlflow_model_registry.png` | registered model `cardiorisk-classifier` with the `@champion` alias |
| `05_github_actions_pipeline.png` | a workflow run graph with all six jobs green |
| `06_github_actions_summary.png` | the run summary: test summary and model comparison table |
| `07_github_actions_artifacts.png` | the run's artifacts list |
| `08_ghcr_package.png` | the GHCR package with the `sha-<7>` and `latest` tags |
| `09_docker_build.png` | `make docker-build` and `make smoke` passing |
| `09_docker_build_run.png` | `make docker-run`: the container's JSON logs |
| `10_fastapi_docs.png` | Swagger UI `/docs` with a `/predict` request and response |
| `11_k8s_resources.png` | `kubectl --context minikube -n cardiorisk get pods,svc,ingress,hpa` |
| `12_k8s_smoke_test.png` | `minikube tunnel` running and `make smoke` passing via the LoadBalancer and the Ingress |
| `13_prometheus_targets.png` | Prometheus `/targets` with the `cardiorisk-api` pods UP |
| `14_grafana_dashboard.png` | Grafana "CardioRisk API" dashboard with traffic from `make load` |
| `15_api_json_logs.png` | `kubectl --context minikube -n cardiorisk logs` output with JSON `request` and `prediction` lines |
