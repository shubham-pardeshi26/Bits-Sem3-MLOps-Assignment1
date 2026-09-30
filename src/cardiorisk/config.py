"""Paths and constants shared by every stage (data, training, serving)."""

import os
from pathlib import Path

ROOT = Path(os.environ.get("CARDIORISK_ROOT", Path(__file__).resolve().parents[2]))

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_CSV = ROOT / "data" / "processed" / "heart_disease_clean.csv"
MODEL_PATH = ROOT / "models" / "cardiorisk_model.joblib"
MODEL_CARD_PATH = ROOT / "models" / "model_card.json"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# UCI Heart Disease (dataset id 45): one "processed" file per collecting hospital.
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
SITE_FILES = {
    "cleveland": "processed.cleveland.data",
    "hungarian": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va": "processed.va.data",
}
# The 14 attributes of the processed files, in file order ("num" = diagnosis 0-4).
RAW_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]  # fmt: skip
TARGET = "target"  # 1 = heart disease present (num > 0)

RANDOM_STATE = 42

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
MLFLOW_EXPERIMENT = "cardiorisk-heart-disease"
REGISTERED_MODEL = "cardiorisk-classifier"
