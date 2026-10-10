"""Testes do baseline de modelagem (Etapa 1, FR-006).

Cobre a factory de estimadores, o treino com persistência, a avaliação
com Brier/calibração e a predição em lote.
"""

import numpy as np

import src.model.train as train_module
from src.model.evaluate import calibration_bins, evaluate_frame
from src.model.factory import build_estimator
from src.model.predict import predict_proba
from tests.conftest import make_credit_frame


def test_build_estimator_factory():
    """A factory constrói ambos os classificadores configurados."""
    for name in ("xgboost", "logistic_regression"):
        estimator = build_estimator(name)
        assert hasattr(estimator, "fit")
        assert hasattr(estimator, "predict_proba")


def test_train_persists_model_and_metrics(tmp_path, monkeypatch):
    """O treino persiste o modelo e as métricas com as chaves esperadas."""
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "train_metrics.json"
    monkeypatch.setattr(train_module, "MODEL_PATH", model_path)
    monkeypatch.setattr(train_module, "TRAIN_METRICS_PATH", metrics_path)

    frame = make_credit_frame(n_rows=300, seed=3)
    metrics = train_module.train(frame)

    assert model_path.exists()
    assert metrics_path.exists()
    for key in ("roc_auc", "precision", "recall", "brier", "n_train", "n_test"):
        assert key in metrics
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_evaluate_frame_reports_metrics_and_calibration():
    """A avaliação retorna ROC-AUC, Brier e a curva de calibração."""
    frame = make_credit_frame(n_rows=200, seed=9)
    probabilities = np.clip(frame["PAY_0"].to_numpy() / 10.0, 0.0, 1.0)
    metrics = evaluate_frame(frame, probabilities, "default_next_month")
    assert {"roc_auc", "precision", "recall", "brier", "calibration"} <= set(metrics)
    assert isinstance(metrics["calibration"], list)
    assert 0.0 <= metrics["brier"] <= 1.0


def test_calibration_bins_shapes():
    """A curva de calibração agrega bins por faixa de probabilidade."""
    labels = np.array([0, 0, 1, 1, 1])
    probabilities = np.array([0.05, 0.15, 0.55, 0.85, 0.95])
    curve = calibration_bins(labels, probabilities, n_bins=5)
    assert curve
    assert all({"bin", "mean_predicted", "fraction_positive", "count"} <= set(b) for b in curve)


def test_predict_proba_returns_probabilities():
    """A predição em lote retorna probabilidades da classe positiva."""
    bundle = {
        "model": build_estimator("logistic_regression"),
        "features": [
            "LIMIT_BAL",
            "SEX",
            "EDUCATION",
            "MARRIAGE",
            "AGE",
            "PAY_0",
            "BILL_AMT1",
            "PAY_AMT1",
        ],
    }
    frame = make_credit_frame(n_rows=50, seed=4)
    feature_frame = frame.drop(columns=["default_next_month"])
    labels = frame["default_next_month"]
    bundle["model"].fit(feature_frame, labels)  # type: ignore[attr-defined]
    probabilities = predict_proba(feature_frame, bundle)
    assert probabilities.shape == (len(frame),)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
