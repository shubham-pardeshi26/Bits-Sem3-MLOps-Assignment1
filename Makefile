# CardioRisk: one entry point per pipeline stage; `make` lists them. CI (.github/workflows/ci.yml)
# runs the same Python commands, so a local `make lint test train` predicts a green pipeline.

PYTHON ?= python3.12
PY ?= .venv/bin/python
IMAGE ?= cardiorisk-api:1.0.0
# Pinned on purpose: every kubectl call names this context, so nothing is ever applied to another
# cluster in your kubeconfig, whatever `kubectl config current-context` happens to be.
KUBE_CONTEXT ?= minikube
KUBECTL = kubectl --context $(KUBE_CONTEXT)
MINIKUBE ?= minikube
OVERLAY ?= minikube
NAMESPACE = cardiorisk
# Host port for serve, docker-run, compose-up, smoke and load; the container listens on 8000.
PORT ?= 8081
URL ?= http://127.0.0.1:$(PORT)
# Not MLflow's default 5000: macOS AirPlay Receiver holds 5000.
MLFLOW_PORT ?= 5001
# Index for pip inside image builds; point it at a mirror if the Docker VM cannot reach pypi.org.
PIP_INDEX_URL ?= https://pypi.org/simple
# Headless Chrome prints the PDF report (make report-pdf). Override it on Linux or Windows, e.g.
# CHROME=google-chrome, CHROME=chromium or the path to chrome.exe.
CHROME ?= "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

.PHONY: help setup data eda train test lint format notebook serve smoke load predict mlflow-ui \
	docker-build docker-run compose-up compose-down minikube-start k8s-image k8s-deploy \
	k8s-status k8s-forward k8s-delete report report-pdf clean

help: ## List the targets (the default goal)
	@echo "Usage: make <target> [VAR=value]"
	@echo "Now: PY=$(PY) IMAGE=$(IMAGE) KUBE_CONTEXT=$(KUBE_CONTEXT) OVERLAY=$(OVERLAY) URL=$(URL)"
	@awk -F ':.*## ' '/^[a-z0-9-]+:.*## / { printf "  %-15s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

setup: ## Create .venv with PYTHON and install the pinned requirements.txt
	$(PYTHON) -m venv .venv
	.venv/bin/python -m pip install -r requirements.txt

data: ## Verify (or download) the UCI files, write data/processed/heart_disease_clean.csv
	$(PY) -m cardiorisk.data

eda: ## EDA figures in reports/figures/eda_*.png
	$(PY) -m cardiorisk.eda

train: ## Tune and compare the models, track runs in MLflow, save the chosen one in models/
	$(PY) -m cardiorisk.train

test: ## Unit tests
	$(PY) -m pytest

lint: ## Ruff lint and format check, as in CI
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format: ## Apply ruff's fixes and formatting
	$(PY) -m ruff check --fix .
	$(PY) -m ruff format .

notebook: ## Re-run notebooks/01_eda.ipynb in place, saving its outputs
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

# Without watchfiles, uvicorn polls every .py file under its reload dirs: keep .venv out of them.
serve: ## API on http://127.0.0.1:PORT, reloaded on code changes
	$(PY) -m uvicorn cardiorisk.api:app --reload --reload-dir src --port $(PORT)

smoke: ## Smoke-test the API at URL
	$(PY) scripts/smoke_test.py $(URL)

load: ## Smoke test plus 300 varied requests, to feed the Grafana dashboard
	$(PY) scripts/smoke_test.py $(URL) --requests 300

predict: ## Score scripts/sample_request.json with the saved model
	$(PY) -m cardiorisk.predict --input scripts/sample_request.json

mlflow-ui: ## MLflow UI for the local tracking store on http://127.0.0.1:MLFLOW_PORT
	$(PY) -m mlflow ui --port $(MLFLOW_PORT) \
		--backend-store-uri "$$($(PY) -c 'import cardiorisk.config as c; print(c.MLFLOW_TRACKING_URI)')"

docker-build: ## Build IMAGE; the model in models/ is baked in, so train first
	docker build --build-arg PIP_INDEX_URL=$(PIP_INDEX_URL) -t $(IMAGE) .

docker-run: ## Run IMAGE on http://127.0.0.1:PORT in the foreground (Ctrl-C removes it)
	docker run --rm -p 127.0.0.1:$(PORT):8000 $(IMAGE)

compose-up: ## API (on PORT), Prometheus and Grafana with docker compose, in the background
	API_PORT=$(PORT) PIP_INDEX_URL=$(PIP_INDEX_URL) docker compose up --build -d
	@echo "API http://127.0.0.1:$(PORT)/docs | Prometheus :9090 | Grafana :3000"

compose-down: ## Stop the compose stack
	docker compose down

minikube-start: ## Start minikube (docker driver) with the ingress and metrics-server addons
	$(MINIKUBE) start --driver=docker
	$(MINIKUBE) addons enable ingress
	$(MINIKUBE) addons enable metrics-server

k8s-image: ## Build IMAGE inside minikube, so the cluster needs no registry
	$(MINIKUBE) image build --build-opt=build-arg=PIP_INDEX_URL=$(PIP_INDEX_URL) -t $(IMAGE) .

k8s-deploy: ## Apply deploy/k8s/overlays/OVERLAY to KUBE_CONTEXT and wait for the rollouts
	@kubectl config get-contexts $(KUBE_CONTEXT) >/dev/null 2>&1 || { \
		echo "Refusing to deploy: there is no kube context '$(KUBE_CONTEXT)'." >&2; \
		echo "Run 'make minikube-start', or set KUBE_CONTEXT (see kubectl config get-contexts)." >&2; \
		exit 1; }
	$(KUBECTL) apply -k deploy/k8s/overlays/$(OVERLAY)
	$(KUBECTL) -n $(NAMESPACE) rollout status deployment/cardiorisk-api --timeout=180s
	$(KUBECTL) -n $(NAMESPACE) rollout status deployment/prometheus --timeout=120s
	$(KUBECTL) -n $(NAMESPACE) rollout status deployment/grafana --timeout=120s

k8s-status: ## Workloads, services, HPA and ingress in the cardiorisk namespace
	$(KUBECTL) -n $(NAMESPACE) get all,ingress,hpa

k8s-forward: ## Grafana on :3000 and Prometheus on :9090 via port-forward (Ctrl-C stops both)
	@echo "Grafana http://127.0.0.1:3000  Prometheus http://127.0.0.1:9090  (Ctrl-C to stop)"
	@$(KUBECTL) -n $(NAMESPACE) port-forward svc/prometheus 9090:9090 & \
	trap "kill $$! 2>/dev/null" EXIT; \
	$(KUBECTL) -n $(NAMESPACE) port-forward svc/grafana 3000:3000

k8s-delete: ## Delete everything deploy/k8s/overlays/OVERLAY created, namespace included
	$(KUBECTL) delete -k deploy/k8s/overlays/$(OVERLAY) --ignore-not-found

# REPORT.md links images relative to itself (figures/..., ../screenshots/...), as GitHub renders it.
# Rebuild both after adding or replacing screenshots: they embed copies of the images.
report: ## Editable report: reports/REPORT.md -> reports/report.docx (A4, reports/reference.docx styles)
	$(PY) -c "import pypandoc; pypandoc.convert_file('reports/REPORT.md', 'docx', \
		outputfile='reports/report.docx', extra_args=['--toc', '--resource-path=reports', \
		'--lua-filter=reports/pandoc-figures.lua', '--reference-doc=reports/reference.docx'])"

report-pdf: ## Final report: REPORT.md -> HTML (reports/report.css) -> reports/report.pdf via CHROME
	@tmp=$$(mktemp -d) && trap 'rm -rf "$$tmp"' EXIT && \
	$(PY) -c "import pypandoc; pypandoc.convert_file('reports/REPORT.md', 'html5', \
		outputfile='$$tmp/report.html', extra_args=['--standalone', '--embed-resources', '--toc', \
		'--resource-path=reports', '--lua-filter=reports/pandoc-figures.lua', \
		'--css=reports/report.css', '--metadata=lang:en'])" && \
	$(CHROME) --headless=new --disable-gpu --no-pdf-header-footer --no-first-run \
		--user-data-dir="$$tmp/profile" --print-to-pdf=reports/report.pdf "file://$$tmp/report.html" && \
	$(PY) -c "import re; pdf = open('reports/report.pdf', 'rb').read(); \
		assert pdf.rstrip().endswith(b'%%EOF'), 'reports/report.pdf is incomplete'; \
		print('reports/report.pdf:', max(map(int, re.findall(rb'/Count\s+(\d+)', pdf))), 'pages')"

clean: ## Delete caches only: data/, models/, reports/ and the MLflow store stay
	find . -path ./.venv -prune -o -type d \( -name __pycache__ -o -name .pytest_cache \
		-o -name .ruff_cache -o -name .ipynb_checkpoints \) -prune -exec rm -rf {} +
