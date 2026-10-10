"""Dashboard de observabilidade (Streamlit). Etapa 3.

Apresenta a saúde do pipeline: status/latência/volume das tarefas,
tendência de ROC-AUC, heatmap de PSI (feature × mês), SLO/Error Budget,
feed de alertas, link para o relatório Evidently e métricas de negócio
(proxy de aprovações/recusas).

Execução (Docker): `make dashboard-up`.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

__all__ = ["main"]

RUN_SUMMARY_PATH = Path("reports/monitoring/run_summary.json")
STATISTICAL_REPORT_PATH = Path("reports/drift/statistical_report.json")
PERFORMANCE_PATH = Path("reports/drift/performance.json")
SLO_REPORT_PATH = Path("reports/monitoring/slo_report.json")
DRIFT_HTML_PATH = Path("reports/drift/drift_report.html")
LOG_PATH = Path("logs/pipeline.ndjson")


def _load_json(path: Path):
    """Carrega um JSON se existir, senão retorna vazio."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _task_table() -> pd.DataFrame:
    """Extrai status/latência/volume por tarefa dos logs do pipeline."""
    if not LOG_PATH.exists():
        return pd.DataFrame()
    events = []
    for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not events:
        return pd.DataFrame()
    frame = pd.DataFrame(events)
    rows = []
    for task, group in frame.groupby("task"):
        finished = group[group["status"].isin(["success", "failed"])]
        last = finished.iloc[-1] if not finished.empty else group.iloc[-1]
        rows.append(
            {
                "task": task,
                "status": last.get("status"),
                "latency_ms": last.get("latency_ms"),
                "n_rows": last.get("n_rows"),
                "run_id": last.get("run_id"),
            }
        )
    return pd.DataFrame(rows).set_index("task")


def _psi_heatmap(statistical: dict) -> pd.DataFrame:
    """Monta o heatmap de PSI por feature (linhas) × mês (colunas)."""
    history = statistical.get("history", [])
    records = {}
    for entry in history:
        month = entry.get("month")
        for feature, values in entry.get("features", {}).items():
            records.setdefault(feature, {})[month] = values.get("psi")
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records).T
    return frame.sort_index(axis=1)


def main() -> None:
    """Renderiza o dashboard de saúde do pipeline."""
    st.set_page_config(page_title="Credit Drift Monitor", layout="wide")
    st.title("Observabilidade — Credit Scoring")
    st.caption("Saúde do pipeline, drift, SLO e governança")

    summary = _load_json(RUN_SUMMARY_PATH)
    statistical = _load_json(STATISTICAL_REPORT_PATH)
    performance = _load_json(PERFORMANCE_PATH)
    slo = _load_json(SLO_REPORT_PATH)

    if not summary and not statistical:
        st.warning("Nenhum relatório encontrado. Rode `make pipeline` primeiro.")
        return

    health = summary.get("health", {})

    # ── Cartões de topo ───────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Alertas de drift", health.get("drift_alerts_total", 0))
    col2.metric("ROC-AUC (baseline)", f"{health.get('model_auc_baseline', 0):.4f}")
    col3.metric("Queda de ROC-AUC", f"{health.get('model_auc_drop', 0):.4f}")
    col4.metric(
        "Status SLO",
        summary.get("slo", {}).get("status_geral", slo.get("status_geral", "n/a")),
    )

    st.caption(
        f"Decisão de re-treino: **{summary.get('retrain_decision', 'n/a')}** | "
        f"Fairness OK: **{summary.get('fairness_passed', 'n/a')}**"
    )

    # ── Tarefas do pipeline ───────────────────────────────────────────
    tasks = _task_table()
    if not tasks.empty:
        st.subheader("Tarefas do pipeline (última execução)")
        st.dataframe(tasks, use_container_width=True)

    # ── Degradação do modelo ──────────────────────────────────────────
    auc_values = health.get("model_roc_auc_by_month", [])
    if not auc_values and isinstance(performance, list):
        auc_values = [entry.get("roc_auc", 0.0) for entry in performance]
    if auc_values:
        st.subheader("Degradação do modelo ao longo do tempo")
        st.line_chart(pd.DataFrame({"ROC-AUC": auc_values}), x_label="Mês", y_label="ROC-AUC")

    # ── Heatmap de PSI ────────────────────────────────────────────────
    heatmap = _psi_heatmap(statistical)
    if not heatmap.empty:
        st.subheader("PSI por feature × mês")
        st.dataframe(
            heatmap.style.background_gradient(cmap="OrRd", axis=None).format("{:.2f}"),
            use_container_width=True,
        )

    # ── SLO / Error Budget ────────────────────────────────────────────
    checks = slo.get("checks", [])
    if checks:
        st.subheader("SLO / Error Budget")
        st.dataframe(pd.DataFrame(checks).set_index("slo"), use_container_width=True)

    # ── Métricas de negócio (proxy aprovações/recusas) ────────────────
    if isinstance(performance, list) and performance:
        st.subheader("Métricas de negócio (proxy)")
        business = pd.DataFrame(
            {
                "mês": [entry.get("month") for entry in performance],
                "taxa_default_prevista": [entry.get("p_positive", 0.0) for entry in performance],
                "aprovacoes_proxy": [1.0 - entry.get("p_positive", 0.0) for entry in performance],
                "brier": [entry.get("brier", 0.0) for entry in performance],
            }
        ).set_index("mês")
        st.dataframe(business, use_container_width=True)

    # ── Feed de alertas ───────────────────────────────────────────────
    alerts = summary.get("alerts", [])
    if alerts:
        st.subheader("Alertas ativos")
        st.dataframe(pd.DataFrame(alerts), use_container_width=True)

    # ── Relatório Evidently ───────────────────────────────────────────
    if DRIFT_HTML_PATH.exists():
        st.subheader("Relatório Evidently (drift)")
        st.download_button(
            "Baixar drift_report.html",
            data=DRIFT_HTML_PATH.read_bytes(),
            file_name="drift_report.html",
            mime="text/html",
        )
        with st.expander("Pré-visualizar relatório Evidently"):
            st.components.v1.html(DRIFT_HTML_PATH.read_text(encoding="utf-8"), height=600)


if __name__ == "__main__":
    main()
