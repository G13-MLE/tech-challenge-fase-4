"""Exporter Prometheus ``/metrics`` (FR-022).

Implementa um ``Collector`` que lê, a cada scrape, os relatórios JSON
(``reports/**/*.json``) e os logs (``logs/pipeline.ndjson``), expondo as
métricas operacionais e de saúde do modelo no formato de texto Prometheus.
Servidor HTTP dedicado via ``prometheus_client.start_http_server(8000)``,
escrutinado por ``configs/prometheus.yml`` em ``exporter:8000``.

Além da evidência canônica do pipeline, quando existir a evidência isolada do
demo (``make demo-drift`` → ``reports/drift/demo/``) o exporter expõe um
conjunto paralelo ``demo_*`` (mesmos nomes com prefixo), permitindo ver o
choque econômico amplificado no Grafana **sem** sobrescrever as métricas do
pipeline (FR-031, US7).
"""

import json
import logging
from pathlib import Path

from prometheus_client import CollectorRegistry, start_http_server
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import Collector

from src.core.dataset import (
    DEMO_PERFORMANCE_PATH,
    DEMO_STATISTICAL_REPORT_PATH,
    DEMO_SUMMARY_PATH,
    PERFORMANCE_PATH,
    RUN_SUMMARY_PATH,
    SLO_REPORT_PATH,
    STATISTICAL_REPORT_PATH,
)
from src.observability.logging import LOG_PATH
from src.observability.metrics import reference_baseline

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


def _task_families() -> list:
    """Famílias de métricas operacionais derivadas dos logs JSON-lines."""
    families = []
    task_status = GaugeMetricFamily("pipeline_task_status", "1=success, 0=failed", labels=["task"])
    task_latency = GaugeMetricFamily("pipeline_task_latency_ms", "Latência (ms)", labels=["task"])
    task_rows = GaugeMetricFamily("pipeline_rows_processed", "Linhas processadas", labels=["task"])
    for task, values in _task_metrics(LOG_PATH).items():
        task_status.add_metric([task], values["status"])
        task_latency.add_metric([task], values["latency_ms"])
        task_rows.add_metric([task], values["rows"])
    families += [task_status, task_latency, task_rows]
    return families


def _drift_families(statistical_path: Path, performance_path: Path, prefix: str = "") -> list:
    """Famílias de drift/performance a partir dos relatórios informados.

    Args:
        statistical_path: Relatório estatístico (PSI/KS/χ² + ``history``).
        performance_path: Série de performance por snapshot (ROC-AUC/Brier).
        prefix: Prefixo opcional dos nomes (ex.: ``demo_`` para a evidência do demo).

    Returns:
        Lista de ``GaugeMetricFamily`` (vazia se os relatórios não existirem).
    """
    families = []

    statistical = _load_json(statistical_path)
    if statistical:
        features_total = GaugeMetricFamily(f"{prefix}drift_features_total", "Features avaliadas")
        features_total.add_metric([], statistical.get("n_features", 0))
        max_psi = GaugeMetricFamily(f"{prefix}drift_max_psi", "PSI máximo entre as features")
        max_psi.add_metric([], float(statistical.get("max_psi", 0.0)))
        drifted = GaugeMetricFamily(
            f"{prefix}drift_features_drifted",
            "Features com drift",
            labels=["severity"],
        )
        by_severity: dict[str, int] = {}
        for entry in statistical.get("features", {}).values():
            severity = entry.get("severity", "stable")
            by_severity[severity] = by_severity.get(severity, 0) + 1
        for severity, count in by_severity.items():
            drifted.add_metric([severity], count)
        families += [features_total, max_psi, drifted]

        psi = GaugeMetricFamily(
            f"{prefix}drift_psi", "PSI por feature/mês", labels=["feature", "month"]
        )
        for history in statistical.get("history", []):
            month = str(history.get("month"))
            for feature, values in history.get("features", {}).items():
                psi.add_metric([feature, month], float(values.get("psi", 0.0)))
        families.append(psi)

    performance = _load_json(performance_path)
    if isinstance(performance, list) and performance:
        auc = GaugeMetricFamily(f"{prefix}model_roc_auc", "ROC-AUC por snapshot", labels=["month"])
        for entry in performance:
            auc.add_metric([str(entry.get("month"))], float(entry.get("roc_auc", 0.0)))
        auc_drop = GaugeMetricFamily(f"{prefix}model_auc_drop", "Queda de AUC vs baseline")
        auc_drop.add_metric(
            [],
            float(reference_baseline(performance) - performance[-1].get("roc_auc", 0.0)),
        )
        families += [auc, auc_drop]

    return families


def _alert_families(summary_path: Path, prefix: str = "") -> list:
    """Família de alertas por tipo a partir de um resumo com ``alerts``."""
    summary = _load_json(summary_path)
    if not summary:
        return []
    alerts = GaugeMetricFamily(f"{prefix}drift_alerts_total", "Alertas por tipo", labels=["type"])
    by_type: dict[str, int] = {}
    for alert in summary.get("alerts", []):
        alert_type = alert.get("type", "unknown")
        by_type[alert_type] = by_type.get(alert_type, 0) + 1
    for alert_type, count in by_type.items():
        alerts.add_metric([alert_type], count)
    return [alerts]


def _slo_families(slo_path: Path) -> list:
    """Famílias de SLO/error budget a partir do relatório de monitoramento."""
    slo = _load_json(slo_path)
    if not slo:
        return []
    budget = GaugeMetricFamily("slo_error_budget", "Error budget por SLO", labels=["slo"])
    status = GaugeMetricFamily("slo_status", "1=pass, 0=breach", labels=["slo"])
    for check in slo.get("checks", []):
        name = check.get("slo", "unknown")
        budget.add_metric([name], float(check.get("error_budget", 0.0)))
        status.add_metric([name], 1 if check.get("status") == "ok" else 0)
    return [budget, status]


def _demo_families() -> list:
    """Famílias ``demo_*`` quando a evidência isolada do demo existir.

    O demo (``make demo-drift``) escreve ``reports/drift/demo/`` sem tocar os
    artefatos canônicos; expor um conjunto paralelo permite comparar o choque
    amplificado no Grafana mantendo o pipeline intacto.
    """
    if not DEMO_STATISTICAL_REPORT_PATH.exists():
        return []
    return _drift_families(
        DEMO_STATISTICAL_REPORT_PATH, DEMO_PERFORMANCE_PATH, prefix="demo_"
    ) + _alert_families(DEMO_SUMMARY_PATH, prefix="demo_")


def collect_metrics() -> list:
    """Coleta as métricas atuais a partir dos relatórios e logs.

    Returns:
        Lista de ``GaugeMetricFamily`` prontos para exposição.
    """
    families = _task_families()
    families += _drift_families(STATISTICAL_REPORT_PATH, PERFORMANCE_PATH)
    families += _alert_families(RUN_SUMMARY_PATH)
    families += _slo_families(SLO_REPORT_PATH)
    families += _demo_families()
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
    """Ponto de entrada do exporter (bloqueante).

    O servidor HTTP do ``prometheus_client`` roda em uma *daemon thread*;
    sem um bloqueio no processo principal, o interpretador encerra o
    processo imediatamente após o início e o ``/metrics`` nunca responde.
    """
    from threading import Event

    from src.core.config import get_settings

    start(get_settings().prometheus_exporter_port)
    Event().wait()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
