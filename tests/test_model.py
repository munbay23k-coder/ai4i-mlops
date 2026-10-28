import numpy as np

from ai4i_mlops.config import MIN_RECALL, MIN_F1
from ai4i_mlops.data import load_data, split_data
from ai4i_mlops.predict import load_model
from ai4i_mlops.train import evaluate_model


def test_model_predictions(trained_model):
    model = load_model()

    df = load_data()
    _, X_test, _, y_test = split_data(df)

    y_pred = model.predict(X_test)

    assert len(y_pred) == len(y_test)
    assert set(np.unique(y_pred)).issubset({0, 1})


def test_model_quality(trained_model):
    model = load_model()

    df = load_data()
    _, X_test, _, y_test = split_data(df)

    metrics = evaluate_model(model, X_test, y_test)

    assert metrics["recall"] >= MIN_RECALL
    assert metrics["f1"] >= MIN_F1