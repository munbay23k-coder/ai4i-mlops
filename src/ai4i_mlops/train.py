
from typing import Any
from datetime import datetime
import json

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    classification_report,
)

from ai4i_mlops.config import (
    FEATURES,
    METRICS_PATH,
    MIN_RECALL,
    MIN_F1,
    MODELS_PARAMS,
    MODEL_PATH,
    THRESHOLD,
)
from ai4i_mlops.data import load_data, split_data


def train_model(X_train, y_train) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Type"])
        ],
        remainder="passthrough"
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", HistGradientBoostingClassifier(**MODELS_PARAMS))
    ])

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test) -> dict[str, Any]:
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= THRESHOLD).astype(int)

    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "average_precision": round(
            average_precision_score(y_test, y_proba), 4
        ),
        "report": classification_report(
            y_test, y_pred, output_dict=True
        )
    }


def save_model(model, metrics: dict[str, Any]) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    model_metadata = {
        "model_version": "0.1.0",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "model_type": type(model.named_steps["model"]).__name__,
        "params": MODELS_PARAMS,
        "features": FEATURES,
        "classes": model.classes_.tolist(),
        "threshold": THRESHOLD,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "average_precision": metrics["average_precision"]
    }

    METRICS_PATH.write_text(
        json.dumps(model_metadata, indent=2),
        encoding="utf-8"
    )


def main() -> None:
    print("1) Cargar datos...")
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"    Entrenamiento: {len(X_train)} filas | Test: {len(X_test)} filas")

    print("2) Entrenando modelo...")
    model = train_model(X_train, y_train)

    print("3) Evaluando modelo...")
    metrics = evaluate_model(model, X_test, y_test)

    print(f"    Accuracy: {metrics['accuracy']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall: {metrics['recall']:.4f}")
    print(f"    F1: {metrics['f1']:.4f}")
    print(f"    Average Precision: {metrics['average_precision']:.4f}")

    if metrics["recall"] < MIN_RECALL or metrics["f1"] < MIN_F1:
        raise ValueError(
            f"El modelo no alcanzó la calidad mínima: "
            f"Recall >= {MIN_RECALL}, F1 >= {MIN_F1}. No se guardará"
        )

    print("4) Guardando el modelo...")
    save_model(model, metrics)
    print(f"    Modelo: {MODEL_PATH}")
    print(f"    Metricas: {METRICS_PATH}")
    print("Done!")


if __name__ == "__main__":
    main()
