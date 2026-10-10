"""Definição das métricas de saúde operacional do pipeline.

Documenta e agrega as métricas exigidas pela Etapa 3: status de execução
das tarefas, latência, volume processado, contagem de alertas de
falha/drift e ROC-AUC por snapshot. As definições canônicas ficam em
``docs/metrics.md`` (FR-020).
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["HEALTH_METRICS", "aggregate_health", "aggregate_task_metrics"]

HEALTH_METRICS = {
    "pipeline_task_status": "Status de execução de cada estágio (started/success/failed)",
    "pipeline_task_latency_ms": "Latência de execução de cada tarefa em milissegundos",
    "pipeline_rows_processed": "Número de linhas processadas por estágio",
    "drift_alerts_total": "Quantidade de features com drift acima do limiar",
    "model_roc_auc": "ROC-AUC do modelo por snapshot de produção",
    "model_auc_drop": "Queda de ROC-AUC em relação ao baseline de referência",
    "slo_status": "Status geral do SLO de dados (ok/degraded/critical)",
}


def aggregate_task_metrics(log_path: Path) -> dict:
    """Agrega métricas operacionais (status, latência, volume) dos logs.

    Args:
        log_path: Caminho do arquivo JSON-lines de eventos.

    Returns:
        Dicionário com contagem de falhas, latência total/máxima e volume.
    """
    if not log_path.exists():
        return {"n_events": 0, "n_failed": 0, "total_latency_ms": 0.0, "rows_processed": 0}
    n_events = 0
    n_failed = 0
    total_latency = 0.0
    rows = 0
    with log_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            n_events += 1
            if event.get("status") == "failed":
                n_failed += 1
            total_latency += float(event.get("latency_ms", 0.0) or 0.0)
            rows += int(event.get("n_rows", 0) or 0)
    return {
        "n_events": n_events,
        "n_failed": n_failed,
        "total_latency_ms": round(total_latency, 2),
        "rows_processed": rows,
    }


def aggregate_health(
    statistical_report: dict, performance: list[dict], log_path: Path | None = None
) -> dict:
    """Agrega as métricas de saúde a partir dos relatórios de drift.

    Args:
        statistical_report: Relatório PSI/KS por feature.
        performance: Métricas de performance por mês.
        log_path: Caminho opcional dos logs para métricas operacionais.

    Returns:
        Dicionário consolidado de saúde do pipeline.
    """
    auc_values = [entry.get("roc_auc", 0.0) for entry in performance]
    brier_values = [entry.get("brier", 0.0) for entry in performance]
    baseline = auc_values[0] if auc_values else 0.0
    last = auc_values[-1] if auc_values else 0.0
    health = {
        "drift_alerts_total": int(statistical_report.get("n_drifted", 0)),
        "n_severe": int(statistical_report.get("n_severe", 0)),
        "model_roc_auc_by_month": auc_values,
        "model_brier_by_month": brier_values,
        "model_auc_baseline": baseline,
        "model_auc_last": last,
        "model_auc_drop": float(baseline - last),
        "is_breach": bool(statistical_report.get("is_breach", False)),
    }
    if log_path is not None:
        health["tasks"] = aggregate_task_metrics(log_path)
    return health
