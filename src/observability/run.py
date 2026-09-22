"""Orquestra a Etapa 3: consolida logs, métricas e alertas de saúde.

Lê os relatórios de performance e drift, agrega as métricas de saúde
do pipeline e grava `reports/monitoring/run_summary.json`, consumido
pelo dashboard Streamlit.
"""

import json
import logging

from src.core.dataset import (
    RUN_SUMMARY_PATH,
    STATISTICAL_REPORT_PATH,
    atomic_write_text,
)
from src.observability.alerts import build_alerts
from src.observability.metrics import aggregate_health
from src.observability.tracking import tracking_run

logger = logging.getLogger(__name__)

__all__ = ["run", "main"]

PERFORMANCE_PATH = STATISTICAL_REPORT_PATH.parent / "performance.json"


def run() -> dict:
    """Executa a consolidação de observabilidade.

    Returns:
        Resumo de saúde do pipeline com alertas.
    """
    statistical_report = json.loads(STATISTICAL_REPORT_PATH.read_text(encoding="utf-8"))
    performance: list[dict] = []
    if PERFORMANCE_PATH.exists():
        performance = json.loads(PERFORMANCE_PATH.read_text(encoding="utf-8"))
    health = aggregate_health(statistical_report, performance)
    alerts = build_alerts(statistical_report, health)
    summary = {"health": health, "alerts": alerts, "n_alerts": len(alerts)}
    atomic_write_text(RUN_SUMMARY_PATH, json.dumps(summary, indent=2))
    with tracking_run("monitoramento") as mlflow_run:
        if mlflow_run is not None:
            import mlflow

            mlflow.log_metric("drift_alerts_total", health["drift_alerts_total"])
            mlflow.log_metric("model_auc_drop", health["model_auc_drop"])
            mlflow.log_artifact(str(RUN_SUMMARY_PATH))
    logger.info("Observabilidade consolidada: %d alertas.", len(alerts))
    return summary


def main() -> None:
    """Ponto de entrada do estágio de monitoramento."""
    run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
