"""Exporter Prometheus ``/metrics`` (FR-022).

Implementa um ``Collector`` que lê, a cada scrape, os relatórios JSON
(``reports/**/*.json``) e os logs (``logs/pipeline.ndjson``), expondo as
métricas operacionais e de saúde do modelo no formato de texto Prometheus.
Servidor HTTP dedicado via ``prometheus_client.start_http_server(8000)``,
escrutinado por ``configs/prometheus.yml`` em ``exporter:8000``.
"""

import json
import logging
from pathlib import Path

from prometheus_client import CollectorRegistry, start_http_server
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import Collector

from src.core.dataset import (
    PERFORMANCE_PATH,
    RUN_SUMMARY_PATH,
    SLO_REPORT_PATH,
    STATISTICAL_REPORT_PATH,
)
from src.observability.logging import LOG_PATH

logger = logging.getLogger(__name__)

__all__ = ["ReportsCollector", "collect_metrics", "start", "main"]


def _load_json(path: Path) -> dict:
    """Carrega um JSON se existir (usado em tempo de scrape)."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _task_metrics(log_path: Path) -> dict[str, dict]:
    """Extrai o último status/latência/volume por tarefa dos logs."""
    metrics: dict[str, dict] = {}
    if not log_path.exists():
        return metrics
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        task = event.get("task")
        if task is None:
            continue
        entry = metrics.setdefault(task, {"status": 0, "latency_ms": 0.0, "rows": 0})
        if event.get("status") == "success":
            entry["status"] = 1
        elif event.get("status") == "failed":
            entry["status"] = 0
        previous_latency = float(entry["latency_ms"])
        entry["latency_ms"] = float(event.get("latency_ms", previous_latency) or previous_latency)
        entry["rows"] = int(event.get("n_rows", entry["rows"]) or entry["rows"])
    return metrics


def collect_metrics() -> list:
    """Coleta as métricas atuais a partir dos relatórios e logs.

    Returns:
        Lista de ``GaugeMetricFamily`` prontos para exposição.
    """
    families = []

    task_status = GaugeMetricFamily("pipeline_task_status", "1=success, 0=failed", labels=["task"])
    task_latency = GaugeMetricFamily("pipeline_task_latency_ms", "Latência (ms)", labels=["task"])
    task_rows = GaugeMetricFamily("pipeline_rows_processed", "Linhas processadas", labels=["task"])
    for task, values in _task_metrics(LOG_PATH).items():
        task_status.add_metric([task], values["status"])
        task_latency.add_metric([task], values["latency_ms"])
        task_rows.add_metric([task], values["rows"])
    families += [task_status, task_latency, task_rows]

    statistical = _load_json(STATISTICAL_REPORT_PATH)
    if statistical:
        features_total = GaugeMetricFamily("drift_features_total", "Features avaliadas")
        features_total.add_metric([], statistical.get("n_features", 0))
        drifted = GaugeMetricFamily(
            "drift_features_drifted", "Features com drift", labels=["severity"]
        )
        by_severity: dict[str, int] = {}
        for entry in statistical.get("features", {}).values():
            severity = entry.get("severity", "stable")
            by_severity[severity] = by_severity.get(severity, 0) + 1
        for severity, count in by_severity.items():
            drifted.add_metric([severity], count)
        families += [features_total, drifted]

        psi = GaugeMetricFamily("drift_psi", "PSI por feature/mês", labels=["feature", "month"])
        for history in statistical.get("history", []):
            month = str(history.get("month"))
            for feature, values in history.get("features", {}).items():
                psi.add_metric([feature, month], float(values.get("psi", 0.0)))
        families.append(psi)

    performance = _load_json(PERFORMANCE_PATH)
    if isinstance(performance, list) and performance:
        auc = GaugeMetricFamily("model_roc_auc", "ROC-AUC por snapshot", labels=["month"])
        for entry in performance:
            auc.add_metric([str(entry.get("month"))], float(entry.get("roc_auc", 0.0)))
        auc_drop = GaugeMetricFamily("model_auc_drop", "Queda de AUC vs baseline")
        auc_drop.add_metric(
            [], float(performance[0].get("roc_auc", 0.0) - performance[-1].get("roc_auc", 0.0))
        )
        families += [auc, auc_drop]

    summary = _load_json(RUN_SUMMARY_PATH)
    if summary:
        alerts = GaugeMetricFamily("drift_alerts_total", "Alertas por tipo", labels=["type"])
        by_type: dict[str, int] = {}
        for alert in summary.get("alerts", []):
            alert_type = alert.get("type", "unknown")
            by_type[alert_type] = by_type.get(alert_type, 0) + 1
        for alert_type, count in by_type.items():
            alerts.add_metric([alert_type], count)
        families.append(alerts)

    slo = _load_json(SLO_REPORT_PATH)
    if slo:
        budget = GaugeMetricFamily("slo_error_budget", "Error budget por SLO", labels=["slo"])
        status = GaugeMetricFamily("slo_status", "1=pass, 0=breach", labels=["slo"])
        for check in slo.get("checks", []):
            name = check.get("slo", "unknown")
            budget.add_metric([name], float(check.get("error_budget", 0.0)))
            status.add_metric([name], 1 if check.get("status") == "ok" else 0)
        families += [budget, status]

    return families


class ReportsCollector(Collector):
    """Collector que expõe as métricas derivadas dos artefatos do pipeline."""

    def collect(self):
        """Gera as métricas no momento do scrape.

        Yields:
            ``GaugeMetricFamily`` de cada métrica disponível.
        """
        yield from collect_metrics()


def start(port: int = 8000) -> None:
    """Inicia o servidor HTTP de métricas do Prometheus.

    Args:
        port: Porta de escuta (default 8000, conforme o contrato).
    """
    registry = CollectorRegistry()
    registry.register(ReportsCollector())
    start_http_server(port, registry=registry)
    logger.info("Exporter Prometheus escutando em :%d/metrics.", port)


def main() -> None:
    """Ponto de entrada do exporter (bloqueante)."""
    from src.core.config import get_settings

    start(get_settings().prometheus_exporter_port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
