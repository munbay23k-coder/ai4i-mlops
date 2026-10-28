
import json
from typing import Any

import joblib
import pandas as pd

from ai4i_mlops.config import (
    FEATURES,
    MODEL_PATH,
    METRICS_PATH,
    THRESHOLD,
)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {MODEL_PATH}")

    return joblib.load(MODEL_PATH)


def load_model_info() -> dict[str, Any]:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Metadatos no encontrados: {METRICS_PATH}")

    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def predict(features: dict[str, Any]) -> dict[str, Any]:
    model = load_model()

    X = pd.DataFrame([features], columns=FEATURES)

    probabilities = model.predict_proba(X)[0]
    probability_failure = float(probabilities[1])

    prediction = int(probability_failure >= THRESHOLD)

    return {
        "prediction": prediction,
        "probability_failure": round(probability_failure, 4),
        "probability_no_failure": round(float(probabilities[0]), 4)
    }


def main() -> None:
    sample = {
        "Type": "L",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 100
    }

    result = predict(sample)

    print("Predicción:", result["prediction"])
    print("Probabilidad de falla:", result["probability_failure"])
    print("Probabilidad sin falla:", result["probability_no_failure"])


if __name__ == "__main__":
    main()
