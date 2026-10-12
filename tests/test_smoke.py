"""Teste de fumaça do pipeline de modelagem e observabilidade."""

from src.model.train import build_estimator
from src.observability.metrics import aggregate_health


def test_build_estimator_returns_model():
    """O estimador configurado deve ser construível."""
    estimator = build_estimator()
    assert hasattr(estimator, "fit")
    assert hasattr(estimator, "predict_proba")


def test_aggregate_health_computes_drop():
    """O agregador deve calcular a queda de AUC corretamente."""
    statistical_report = {"n_drifted": 3}
    performance = [{"roc_auc": 0.80}, {"roc_auc": 0.70}]
    health = aggregate_health(statistical_report, performance)
    assert health["drift_features_drifted"] == 3
    assert abs(health["model_auc_drop"] - 0.10) < 1e-9
