import numpy as np
from sklearn.ensemble import RandomForestClassifier

from starter.starter.ml.model import (
    train_model,
    compute_model_metrics,
    inference,
)


def test_train_model_returns_random_forest():
    X_train = np.array([
        [25, 0],
        [30, 1],
        [45, 0],
        [35, 1],
    ])
    y_train = np.array([0, 1, 0, 1])

    model = train_model(X_train, y_train)

    assert isinstance(model, RandomForestClassifier)


def test_inference_returns_numpy_array_with_correct_length():
    X_train = np.array([
        [25, 0],
        [30, 1],
        [45, 0],
        [35, 1],
    ])
    y_train = np.array([0, 1, 0, 1])

    model = train_model(X_train, y_train)

    X_test = np.array([
        [40, 1],
        [28, 0],
    ])

    preds = inference(model, X_test)

    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(X_test)


def test_compute_model_metrics_returns_expected_values():
    y = np.array([1, 0, 1, 1])
    preds = np.array([1, 0, 0, 1])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert round(precision, 2) == 1.00
    assert round(recall, 2) == 0.67
    assert round(fbeta, 2) == 0.80
