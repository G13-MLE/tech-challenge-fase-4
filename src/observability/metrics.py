"""Definição das métricas de saúde operacional do pipeline.

Documenta e agrega as métricas exigidas pela Etapa 3: status de
execução das tarefas, latência, volume processado e contagem de
alertas de falha/drift. As definições canônicas ficam em `docs/metrics.md`.
"""

import logging

from src.core.dataset import TRAIN_METRICS_PATH, load_parquet  # noqa: F401

logger = logging.getLogger(__name__)

__all__ = ["HEALTH_METRICS", "aggregate_health"]

HEALTH_METRICS = {
    "pipeline_task_status": "Status de execução de cada estágio (started/success/failed)",
    "pipeline_task_latency_ms": "Latência de execução de cada tarefa em milissegundos",
    "volume_rows_processed": "Número de linhas processadas por estágio",
    "drift_alerts_total": "Quantidade de features com drift acima do limiar",
    "model_roc_auc": "ROC-AUC do modelo por snapshot de produção",
    "model_auc_drop": "Queda de ROC-AUC em relação ao baseline de referência",
}


def aggregate_health(statistical_report: dict, performance: list[dict]) -> dict:
    """Agrega as métricas de saúde a partir dos relatórios de drift.

    Args:
        statistical_report: Relatório PSI/KS por feature.
        performance: Métricas de performance por mês.

    Returns:
        Dicionário consolidado de saúde do pipeline.
    """
    auc_values = [entry.get("roc_auc", 0.0) for entry in performance]
    baseline = auc_values[0] if auc_values else 0.0
    last = auc_values[-1] if auc_values else 0.0
    return {
        "drift_alerts_total": int(statistical_report.get("n_drifted", 0)),
        "model_roc_auc_by_month": auc_values,
        "model_auc_baseline": baseline,
        "model_auc_last": last,
        "model_auc_drop": float(baseline - last),
    }
