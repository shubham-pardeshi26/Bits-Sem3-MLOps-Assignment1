"""Inference shared by the API and the CLI: patient records in, thresholded risk out.

python -m cardiorisk.predict --input scripts/sample_request.json
python -m cardiorisk.predict --csv data/processed/heart_disease_clean.csv
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from cardiorisk import config
from cardiorisk.features import INPUT_FEATURES

THRESHOLD = 0.5
LABELS = ("no_heart_disease", "heart_disease")  # indexed by the predicted class


def load_model(path=config.MODEL_PATH):
    """The fitted pipeline saved by training (preprocessing included)."""
    return joblib.load(path)


def load_model_card(path=config.MODEL_CARD_PATH) -> dict:
    """Training metadata saved next to the model. Optional, so {} when the file is missing."""
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else {}


def predict_records(model, records: list[dict]) -> list[dict]:
    """Score patient dicts. Omitted or None features become NaN, which the pipeline imputes."""
    X = pd.DataFrame(records, columns=INPUT_FEATURES)
    results = []
    for p in model.predict_proba(X)[:, 1]:
        p = round(float(p), 4)
        # Decide on the rounded value so every response is self-consistent (p >= 0.5 <=> class 1).
        pred = int(p >= THRESHOLD)
        results.append(
            {
                "prediction": pred,
                "label": LABELS[pred],
                "confidence": p if pred else round(1 - p, 4),
                "probability_disease": p,
            }
        )
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score patients with the saved cardiorisk model.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="JSON file: one patient object, or a list")
    source.add_argument("--csv", type=Path, help="CSV with INPUT_FEATURES columns (others ignored)")
    parser.add_argument("--model", type=Path, default=config.MODEL_PATH)
    args = parser.parse_args(argv)

    model = load_model(args.model)
    if args.csv:
        output = predict_records(model, pd.read_csv(args.csv).to_dict("records"))
    else:
        data = json.loads(args.input.read_text())
        output = predict_records(model, data if isinstance(data, list) else [data])
        output = output if isinstance(data, list) else output[0]
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
