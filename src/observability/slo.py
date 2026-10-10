"""Data SLOs, Error Budget e pipeline de monitoramento (FR-015).

Define os *Service Level Objectives* de dados/modelo, calcula o Error
Budget (``valor − piso``; positivo = OK) e consolida os checks em
``reports/monitoring/slo_report.json`` no formato
``{status_geral, alertas[], checks[]}``.
"""

import json
import logging
from dataclasses import asdict, dataclass

from src.core.dataset import SLO_REPORT_PATH, atomic_write_text
from src.core.params import load_params

logger = logging.getLogger(__name__)

__all__ = ["ErrorBudget", "evaluate_check", "MonitoringPipeline"]


@dataclass
class ErrorBudget:
    """Error budget de um SLO: folga entre o valor medido e o piso.

    Attributes:
        slo: Nome do SLO.
        value: Valor medido.
        threshold: Piso/teto do SLO.
        comparison: "min" (valor ≥ threshold) ou "max" (valor ≤ threshold).
        error_budget: Folga; positivo = dentro do SLO.
        status: "ok" | "warning" | "breach".
    """

    slo: str
    value: float
    threshold: float
    comparison: str
    error_budget: float
    status: str


def evaluate_check(
    slo: str, value: float, threshold: float, comparison: str = "min"
) -> ErrorBudget:
    """Avalia um único SLO e calcula seu error budget.

    Args:
        slo: Nome do SLO.
        value: Valor medido.
        threshold: Limite do SLO.
        comparison: "min" (valor ≥ threshold) ou "max" (valor ≤ threshold).

    Returns:
        Instância de ``ErrorBudget`` com status.
    """
    budget = value - threshold if comparison == "min" else threshold - value
    status = "ok" if budget >= 0 else "breach"
    return ErrorBudget(
        slo=slo,
        value=float(value),
        threshold=float(threshold),
        comparison=comparison,
        error_budget=float(budget),
        status=status,
    )


class MonitoringPipeline:
    """Consolida os checks de SLO e alertas em um relatório único."""

    def __init__(self) -> None:
        """Carrega os parâmetros de SLO do pipeline."""
        self.slo = load_params().slo

    def _min_pvalue(self, statistical_report: dict) -> float:
        """Menor p-valor (KS/χ²) entre as features."""
        pvalues = []
        for entry in statistical_report.get("features", {}).values():
            if "ks" in entry:
                pvalues.append(entry["ks"]["pvalue"])
            elif "chi2" in entry:
                pvalues.append(entry["chi2"]["pvalue"])
        return min(pvalues) if pvalues else 1.0

    def evaluate(
        self,
        statistical_report: dict,
        performance: list[dict],
        alerts: list[dict] | None = None,
        missing_rate: float = 0.0,
    ) -> dict:
        """Avalia todos os SLOs e monta o relatório.

        Args:
            statistical_report: Relatório PSI/KS do drift.
            performance: Métricas de performance por mês.
            alerts: Alertas consolidados (anti-fadiga).
            missing_rate: Taxa de ausência do último lote validado.

        Returns:
            Relatório ``{status_geral, alertas[], checks[]}``.
        """
        accuracy = float(performance[-1].get("accuracy", 0.0)) if performance else 0.0
        max_psi = float(statistical_report.get("max_psi", 0.0))
        min_pvalue = self._min_pvalue(statistical_report)
        checks = [
            evaluate_check("accuracy_min", accuracy, self.slo.accuracy_min, "min"),
            evaluate_check("psi_max", max_psi, self.slo.psi_max, "max"),
            evaluate_check("ks_pvalue_min", min_pvalue, self.slo.ks_pvalue_min, "min"),
            evaluate_check("missing_rate_max", missing_rate, self.slo.missing_rate_max, "max"),
        ]
        breached = [check for check in checks if check.status == "breach"]
        if any(check.slo in {"psi_max", "accuracy_min"} for check in breached):
            status_geral = "critical"
        elif breached:
            status_geral = "degraded"
        else:
            status_geral = "ok"
        report = {
            "status_geral": status_geral,
            "alertas": alerts or [],
            "checks": [asdict(check) for check in checks],
        }
        atomic_write_text(SLO_REPORT_PATH, json.dumps(report, indent=2))
        logger.info("SLO: %s (%d breaches).", status_geral, len(breached))
        return report
