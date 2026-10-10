"""Lógica de alertas de drift e degradação do modelo.

Compara os resultados dos testes estatísticos e a queda de performance
com os limiares definidos em ``configs/params.yaml`` e aplica a
consolidação **anti-fadiga** (FR-016): um alerta é emitido quando
``breach_features`` (≥2) features entram em drift **juntas** no mesmo
mês, OU quando **1 feature crítica** permanece em severidade severa por
``consolidation_windows`` (≥3) meses consecutivos.
"""

import logging

from src.core.params import load_params

logger = logging.getLogger(__name__)

__all__ = ["build_alerts", "consolidate_history_alerts"]


def _severity_by_feature(history: list[dict]) -> dict[str, list[str]]:
    """Mapeia cada feature à lista de severidades por mês."""
    severities: dict[str, list[str]] = {}
    for entry in history:
        for feature, values in entry.get("features", {}).items():
            severities.setdefault(feature, []).append(values.get("severity", "stable"))
    return severities


def _max_consecutive(severities: list[str], target: str = "severe") -> int:
    """Maior sequência consecutiva de uma severidade alvo."""
    longest = 0
    current = 0
    for value in severities:
        current = current + 1 if value == target else 0
        longest = max(longest, current)
    return longest


def consolidate_history_alerts(history: list[dict]) -> list[dict]:
    """Consolida alertas anti-fadiga a partir da série temporal de drift.

    Args:
        history: Lista de entradas mensais com ``features[col].severity``.

    Returns:
        Lista de alertas consolidados (features, janelas, mensagem).
    """
    params = load_params().alerts
    severities = _severity_by_feature(history)
    alerts: list[dict] = []
    for entry in history:
        month = entry.get("month")
        severe = [
            feature
            for feature, values in entry.get("features", {}).items()
            if values.get("severity") == "severe"
        ]
        if len(severe) >= params.breach_features:
            alerts.append(
                {
                    "type": "data_drift",
                    "severity": "high",
                    "month": month,
                    "features": severe,
                    "windows": 1,
                    "message": (f"{len(severe)} features com drift severo juntas no mês {month}."),
                }
            )
    for feature, series in severities.items():
        consecutive = _max_consecutive(series)
        if consecutive >= params.consolidation_windows:
            alerts.append(
                {
                    "type": "data_drift",
                    "severity": "critical",
                    "month": history[-1].get("month") if history else None,
                    "features": [feature],
                    "windows": consecutive,
                    "message": (
                        f"Feature crítica {feature} em severidade severa por "
                        f"{consecutive} meses consecutivos."
                    ),
                }
            )
    return alerts


def build_alerts(
    statistical_report: dict, health: dict, history: list[dict] | None = None
) -> list[dict]:
    """Gera a lista de alertas a partir dos relatórios de drift e saúde.

    Args:
        statistical_report: Relatório PSI/KS por feature.
        health: Métricas agregadas de saúde.
        history: Série temporal mensal (para consolidação anti-fadiga).

    Returns:
        Lista de alertas consolidados (feature/severidade/valor).
    """
    thresholds = load_params().drift_thresholds
    alerts: list[dict] = []
    if history:
        alerts.extend(consolidate_history_alerts(history))
    else:
        for feature, result in statistical_report.get("features", {}).items():
            if result.get("severity") == "severe":
                alerts.append({"type": "data_drift", "feature": feature, "psi": result["psi"]})
    if health.get("model_auc_drop", 0.0) >= thresholds.auc_drop_alert:
        alerts.append(
            {
                "type": "model_degradation",
                "feature": "model",
                "severity": "high",
                "auc_drop": health["model_auc_drop"],
            }
        )
    logger.info("Alertas consolidados: %d.", len(alerts))
    return alerts
