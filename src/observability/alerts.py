"""Lógica de alertas de drift e degradação do modelo.

Compara os resultados dos testes estatísticos e a queda de performance
com os limiares definidos em `configs/params.yaml`.
"""

import logging

from src.core.params import load_params

logger = logging.getLogger(__name__)

__all__ = ["build_alerts"]


def build_alerts(statistical_report: dict, health: dict) -> list[dict]:
    """Gera a lista de alertas a partir dos relatórios de drift e saúde.

    Args:
        statistical_report: Relatório PSI/KS por feature.
        health: Métricas agregadas de saúde.

    Returns:
        Lista de alertas (feature/severidade/valor).
    """
    thresholds = load_params().drift_thresholds
    alerts: list[dict] = []
    for feature, result in statistical_report.get("features", {}).items():
        if result.get("severity") == "severe":
            alerts.append({"type": "data_drift", "feature": feature, "psi": result["psi"]})
    if health.get("model_auc_drop", 0.0) >= thresholds.auc_drop_alert:
        alerts.append(
            {
                "type": "model_degradation",
                "feature": "model",
                "auc_drop": health["model_auc_drop"],
            }
        )
    logger.info("Alertas gerados: %d.", len(alerts))
    return alerts
