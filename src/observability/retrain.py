"""Gatilho de re-treino com guardrails (FR-017).

Dispara o re-treino por uma regra objetiva e **distinta/later** da
consolidação de alertas: qualquer feature com PSI ≥ 0.25 por N=2 meses
consecutivos (severidade crítica). Em seguida aplica os **guardrails** de
promoção — AUC do challenger ≥ AUC do champion, disparo de impacto ≥ 0.8
e diferença de equal opportunity ≤ 0.10 — e **nunca promove
automaticamente**.

Escreve ``reports/monitoring/retrain_decision.json``.
"""

import json
import logging

from src.core.dataset import RETRAIN_DECISION_PATH, atomic_write_text
from src.core.params import load_params

logger = logging.getLogger(__name__)

__all__ = ["max_consecutive_critical", "decide_retrain"]


def max_consecutive_critical(history: list[dict], psi_max: float) -> tuple[int, float, float]:
    """Maior sequência de meses com alguma feature em severidade crítica.

    Args:
        history: Série mensal com ``features[col].psi``.
        psi_max: Limiar de PSI crítico (0.25).

    Returns:
        Tupla ``(meses_consecutivos, max_psi, fração_features_em_breach)``.
    """
    longest = 0
    current = 0
    max_psi = 0.0
    breach_share = 0.0
    for entry in history:
        features = entry.get("features", {})
        breached = [
            name for name, values in features.items() if float(values.get("psi", 0.0)) >= psi_max
        ]
        entry_max = max(
            (float(values.get("psi", 0.0)) for values in features.values()), default=0.0
        )
        max_psi = max(max_psi, entry_max)
        share = len(breached) / len(features) if features else 0.0
        breach_share = max(breach_share, share)
        current = current + 1 if breached else 0
        longest = max(longest, current)
    return longest, max_psi, breach_share


def decide_retrain(
    statistical_report: dict,
    champion_metrics: dict | None = None,
    fairness_report: dict | None = None,
    challenger_auc: float | None = None,
    model_version: str = "challenger",
) -> dict:
    """Decide o re-treino e aplica os guardrails de promoção.

    Args:
        statistical_report: Relatório de drift (contém ``history``).
        champion_metrics: Métricas do modelo campeão (``roc_auc``).
        fairness_report: Relatório de fairness do challenger.
        challenger_auc: AUC do challenger (opcional; se None, guardrail
            de AUC é considerado satisfeito, mas nada é promovido).
        model_version: Identificador/versão do challenger.

    Returns:
        Dicionário ``{decision, signals, guardrails, model_version}``.
    """
    thresholds = load_params().drift_thresholds
    params = load_params().retrain
    fairness_params = load_params().fairness
    history = statistical_report.get("history", [])
    consecutive, max_psi, breach_share = max_consecutive_critical(history, thresholds.psi_moderate)
    decision = "retrain" if consecutive >= params.consecutive_critical_windows else "no_action"

    champion_auc = float((champion_metrics or {}).get("roc_auc", 0.0))
    auc_ok = True if challenger_auc is None else challenger_auc >= champion_auc

    if fairness_report is None:
        fairness_ok = True
    else:
        by_group = fairness_report.get("by_group", {})
        fairness_ok = bool(by_group) and all(
            entry.get("passes", False) for entry in by_group.values()
        )
        fairness_ok = fairness_ok or not by_group

    promoted = bool(decision == "retrain" and auc_ok and fairness_ok and challenger_auc is not None)
    report = {
        "decision": decision,
        "signals": {
            "consecutive_critical_windows": consecutive,
            "required_windows": params.consecutive_critical_windows,
            "max_psi": max_psi,
            "breached_feature_share": breach_share,
            "champion_auc": champion_auc,
            "challenger_auc": None if challenger_auc is None else float(challenger_auc),
        },
        "guardrails": {
            "auc_ok": bool(auc_ok),
            "fairness_ok": bool(fairness_ok),
            "disparate_impact_min": fairness_params.disparate_impact_min,
            "equal_opportunity_max": fairness_params.equal_opportunity_max,
            "promoted": promoted,
            "note": "Challenger nunca é promovido automaticamente.",
        },
        "model_version": model_version,
    }
    atomic_write_text(RETRAIN_DECISION_PATH, json.dumps(report, indent=2))
    logger.info("Re-treino: %s (guardrails=%s).", decision, report["guardrails"])
    return report
