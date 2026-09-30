# CardioRisk prediction API. Only the serving pins are installed: MLflow and the training stack
# stay out of the image, and the model is baked in at build time (run `make train` first).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

RUN groupadd --gid 10001 cardiorisk \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin cardiorisk

WORKDIR /app

# Mirror override: docker build --build-arg PIP_INDEX_URL=https://<mirror>/simple .
# This layer is rebuilt only when the pins change, not on every code or model change.
ARG PIP_INDEX_URL=https://pypi.org/simple
COPY requirements-serve.txt .
RUN pip install --index-url "$PIP_INDEX_URL" -r requirements-serve.txt

COPY src/ src/
COPY models/ models/
RUN test -f models/cardiorisk_model.joblib || { \
        echo "models/cardiorisk_model.joblib is missing: run 'make train' first" >&2; exit 1; }

# Code and model stay root-owned and read-only for the app user.
USER 10001:10001
EXPOSE 8000
# ProxyHandler({}): an HTTP_PROXY that Docker injects from its client config must not capture this
# local probe. Non-2xx and connection errors raise, so the check fails.
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request as u; u.build_opener(u.ProxyHandler({})).open('http://127.0.0.1:8000/health', timeout=2)"]
# The app writes its own JSON access log, so uvicorn's is switched off.
CMD ["uvicorn", "cardiorisk.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
