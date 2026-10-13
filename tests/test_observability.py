"""Testes de observabilidade (Etapa 3, FR-015–FR-022).

Cobre correlação de logs (``run_id``/``trace_id``), SLO + Error Budget,
consolidação anti-fadiga de alertas, guardrails de re-treino (AUC +
fairness), fairness por subgrupo e o exporter Prometheus.
"""

import json

import numpy as np
import pandas as pd
import pytest

import src.observability.logging as logging_module
import src.observability.retrain as retrain_module
from src.observability.alerts import build_alerts, consolidate_history_alerts
from src.observability.exporter import collect_metrics
from src.observability.fairness import evaluate_fairness
from src.observability.logging import TaskTimer
from src.observability.metrics import aggregate_health
from src.observability.retrain import decide_retrain, max_consecutive_critical
from src.observability.slo import MonitoringPipeline, evaluate_check
from src.observability.tracking import tracking_run

TARGET = "default_next_month"


def _history(entries: list[dict]) -> list[dict]:
    return entries


def test_log_events_share_run_id_and_trace_id(tmp_path, monkeypatch):
    """Eventos de uma mesma tarefa compartilham run_id/trace_id."""
    log_path = tmp_path / "pipeline.ndjson"
    monkeypatch.setattr(logging_module, "LOG_PATH", log_path)
    with TaskTimer("treino", run_id="run-abc"):
        pass
    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(events) == 2
    assert {event["run_id"] for event in events} == {"run-abc"}
    assert events[0]["trace_id"] == events[1]["trace_id"]
    assert events[0]["status"] == "started"
    assert events[1]["status"] == "success"


def test_evaluate_check_error_budget():
    """O error budget sinaliza folga/estouro conforme a comparação."""
    ok = evaluate_check("accuracy_min", 0.9, 0.85, "min")
    breach = evaluate_check("psi_max", 0.4, 0.25, "max")
    assert ok.status == "ok" and ok.error_budget > 0
    assert breach.status == "breach" and breach.error_budget < 0


def test_monitoring_pipeline_status_and_checks(tmp_path, monkeypatch):
    """O pipeline de monitoramento agrega checks e define o status geral."""
    import src.observability.slo as slo_module

    monkeypatch.setattr(slo_module, "SLO_REPORT_PATH", tmp_path / "slo.json")
    statistical = {"max_psi": 0.4, "features": {"A": {"ks": {"pvalue": 0.001}}}}
    performance = [{"accuracy": 0.9}]
    report = MonitoringPipeline().evaluate(statistical, performance, alerts=[])
    assert report["status_geral"] == "critical"
    assert {check["slo"] for check in report["checks"]} >= {"accuracy_min", "psi_max"}
    assert (tmp_path / "slo.json").exists()


def test_alert_consolidation_multiple_features_together():
    """≥2 features severas no mesmo mês consolidam um alerta."""
    history = _history(
        [
            {
                "month": 1,
                "features": {"A": {"severity": "severe"}, "B": {"severity": "severe"}},
            },
            {"month": 2, "features": {"A": {"severity": "stable"}, "B": {"severity": "stable"}}},
        ]
    )
    alerts = consolidate_history_alerts(history)
    assert any(alert["type"] == "data_drift" and len(alert["features"]) >= 2 for alert in alerts)


def test_alert_consolidation_single_critical_over_window():
    """1 feature crítica por ≥3 meses consecutivos consolida um alerta."""
    history = _history(
        [
            {"month": m, "features": {"A": {"severity": "severe"}, "B": {"severity": "stable"}}}
            for m in (1, 2, 3)
        ]
    )
    alerts = consolidate_history_alerts(history)
    assert any(
        alert["type"] == "data_drift" and alert["severity"] == "critical" and alert["windows"] >= 3
        for alert in alerts
    )


def test_build_alerts_with_history_and_degradation():
    """build_alerts combina consolidação de drift e degradação do modelo."""
    history = _history(
        [
            {
                "month": m,
                "features": {"A": {"severity": "severe"}, "B": {"severity": "severe"}},
            }
            for m in (1, 2)
        ]
    )
    health = {"model_auc_drop": 0.2}
    alerts = build_alerts({"features": {}}, health, history=history)
    assert any(alert["type"] == "model_degradation" for alert in alerts)
    assert any(alert["type"] == "data_drift" for alert in alerts)


def test_retrain_trigger_and_guardrails(tmp_path, monkeypatch):
    """Regra objetiva (N=2 meses) dispara re-treino; guardrails nunca promovem."""
    monkeypatch.setattr(retrain_module, "RETRAIN_DECISION_PATH", tmp_path / "retrain.json")
    history = _history([{"month": m, "features": {"A": {"psi": 0.3}}} for m in (1, 2)])
    consecutive, max_psi, _ = max_consecutive_critical(history, 0.25)
    assert consecutive == 2 and max_psi == 0.3

    statistical = {"history": history}
    fairness_fail = {"by_group": {"SEX": {"passes": False}}}
    report = decide_retrain(
        statistical,
        champion_metrics={"roc_auc": 0.8},
        fairness_report=fairness_fail,
        challenger_auc=0.75,
    )
    assert report["decision"] == "retrain"
    assert report["guardrails"]["auc_ok"] is False
    assert report["guardrails"]["fairness_ok"] is False
    assert report["guardrails"]["promoted"] is False


def test_no_retrain_below_window(tmp_path, monkeypatch):
    """Sem meses consecutivos suficientes, a decisão é no_action."""
    monkeypatch.setattr(retrain_module, "RETRAIN_DECISION_PATH", tmp_path / "retrain.json")
    statistical = {"history": [{"month": 1, "features": {"A": {"psi": 0.3}}}]}
    report = decide_retrain(statistical, champion_metrics={"roc_auc": 0.8})
    assert report["decision"] == "no_action"


def test_fairness_metrics_and_guardrails():
    """Fairness calcula paridade, equal opportunity e impacto dispar."""
    rng = np.random.default_rng(0)
    n = 400
    frame = pd.DataFrame(
        {
            "SEX": rng.integers(1, 3, n),
            "AGE": rng.integers(18, 80, n),
            TARGET: rng.integers(0, 2, n),
        }
    )
    probabilities = rng.random(n)
    report = evaluate_fairness(frame, probabilities, TARGET)
    assert "SEX" in report and "AGE_BIN" in report
    for entry in report.values():
        expected_keys = {
            "demographic_parity_diff",
            "equal_opportunity_diff",
            "disparate_impact_ratio",
        }
        assert expected_keys <= set(entry)
        assert "passes" in entry


def test_aggregate_health_includes_slo_and_brier():
    """A saúde agrega ROC-AUC por mês, Brier e severidade."""
    statistical = {"n_drifted": 3, "n_severe": 1, "is_breach": True}
    performance = [{"roc_auc": 0.8, "brier": 0.1}, {"roc_auc": 0.7, "brier": 0.2}]
    health = aggregate_health(statistical, performance)
    assert abs(health["model_auc_drop"] - 0.1) < 1e-9
    assert health["model_brier_by_month"] == [0.1, 0.2]
    assert health["is_breach"] is True


def test_exporter_collects_without_reports():
    """O exporter não falha quando os relatórios estão ausentes."""
    families = collect_metrics()
    assert isinstance(families, list)


def test_exporter_exposes_demo_metrics_when_present(tmp_path, monkeypatch):
    """A evidência isolada do demo é exposta como métricas ``demo_*`` (US7)."""
    import src.observability.exporter as exporter_module

    statistical = {
        "n_features": 2,
        "max_psi": 4.1,
        "features": {"LIMIT_BAL": {"severity": "severe"}, "AGE": {"severity": "moderate"}},
        "history": [
            {"month": 1, "features": {"LIMIT_BAL": {"psi": 0.6}, "AGE": {"psi": 0.2}}},
            {"month": 2, "features": {"LIMIT_BAL": {"psi": 4.1}, "AGE": {"psi": 2.5}}},
        ],
    }
    performance = [
        {"month": 1, "roc_auc": 0.66},
        {"month": 2, "roc_auc": 0.61, "reference_auc": 0.76},
    ]
    summary = {"alerts": [{"type": "data_drift"}, {"type": "model_degradation"}]}

    demo_stat = tmp_path / "statistical_report.json"
    demo_perf = tmp_path / "performance.json"
    demo_summary = tmp_path / "demo_summary.json"
    demo_stat.write_text(json.dumps(statistical))
    demo_perf.write_text(json.dumps(performance))
    demo_summary.write_text(json.dumps(summary))

    monkeypatch.setattr(exporter_module, "DEMO_STATISTICAL_REPORT_PATH", demo_stat)
    monkeypatch.setattr(exporter_module, "DEMO_PERFORMANCE_PATH", demo_perf)
    monkeypatch.setattr(exporter_module, "DEMO_SUMMARY_PATH", demo_summary)

    families = {family.name: family for family in collect_metrics()}
    assert {
        "demo_drift_psi",
        "demo_drift_max_psi",
        "demo_drift_features_total",
        "demo_drift_features_drifted",
        "demo_model_roc_auc",
        "demo_model_auc_drop",
        "demo_drift_alerts_total",
    } <= set(families)

    max_psi_values = {sample.value for sample in families["demo_drift_max_psi"].samples}
    assert 4.1 in max_psi_values
    alert_types = {sample.labels["type"] for sample in families["demo_drift_alerts_total"].samples}
    assert alert_types == {"data_drift", "model_degradation"}


def test_exporter_hides_demo_metrics_without_evidence(tmp_path, monkeypatch):
    """Sem evidência do demo, nenhuma métrica ``demo_*`` é exposta."""
    import src.observability.exporter as exporter_module

    monkeypatch.setattr(exporter_module, "DEMO_STATISTICAL_REPORT_PATH", tmp_path / "missing.json")
    names = {family.name for family in collect_metrics()}
    assert not any(name.startswith("demo_") for name in names)


def test_tracking_run_fallback_when_mlflow_unavailable(monkeypatch):
    """Sem servidor MLflow, tracking_run degrada para None sem quebrar."""
    import mlflow

    def _offline(*args, **kwargs):
        raise ConnectionError("sem tracking server")

    monkeypatch.setattr(mlflow, "set_experiment", _offline)
    with tracking_run("fallback") as run:
        assert run is None


def test_tracking_run_body_exception_propagates_and_fails_run(monkeypatch):
    """Erro no corpo do with propaga limpo e encerra a run como FAILED.

    Regressão: um contextmanager não pode dar um segundo yield após
    receber um throw — isso levanta RuntimeError e mascara o erro original.
    """
    import mlflow

    statuses: list[str | None] = []
    monkeypatch.setattr(mlflow, "set_experiment", lambda *args, **kwargs: None)
    monkeypatch.setattr(mlflow, "start_run", lambda *args, **kwargs: object())
    monkeypatch.setattr(mlflow, "end_run", lambda status=None: statuses.append(status))

    with (
        pytest.raises(ValueError, match="falha no corpo"),
        tracking_run("corpo") as run,
    ):
        assert run is not None
        raise ValueError("falha no corpo")

    assert statuses == ["FAILED"]


def test_exporter_process_serves_metrics():
    """O processo do exporter permanece vivo e responde em /metrics.

    Regressão: o servidor HTTP do ``prometheus_client`` roda em daemon
    thread; sem bloqueio no processo principal, o interpretador encerra
    imediatamente e o endpoint nunca responde.
    """
    import os
    import socket
    import subprocess
    import sys
    import time
    import urllib.request

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    env = {**os.environ, "PROMETHEUS_EXPORTER_PORT": str(port)}
    process = subprocess.Popen(
        [sys.executable, "-m", "src.observability.exporter"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        body = None
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics") as response:
                    body = response.read().decode("utf-8")
                break
            except OSError:
                assert process.poll() is None, "processo do exporter encerrou antes de escutar"
                time.sleep(0.25)
        assert body is not None, "exporter não respondeu /metrics no prazo"
        assert "drift_psi" in body
        assert "model_roc_auc" in body
    finally:
        process.terminate()
        process.wait(timeout=10)
