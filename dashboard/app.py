"""Dashboard de observabilidade (Streamlit).

Etapa 3. Apresenta a saúde do pipeline: status das tarefas, latência,
volume processado, tendência de ROC-AUC e alertas de drift.

Execução (Docker): `make dashboard-up`
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

__all__ = ["main"]

RUN_SUMMARY_PATH = Path("reports/monitoring/run_summary.json")
STATISTICAL_REPORT_PATH = Path("reports/drift/statistical_report.json")
LOG_PATH = Path("logs/pipeline.ndjson")


def _load_json(path: Path) -> dict:
    """Carrega um JSON se existir, senão retorna vazio."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    """Renderiza o dashboard de saúde do pipeline."""
    st.set_page_config(page_title="Credit Drift Monitor", layout="wide")
    st.title("Observabilidade — Credit Scoring")
    st.caption("Saúde do pipeline, drift e degradação do modelo")

    summary = _load_json(RUN_SUMMARY_PATH)
    statistical = _load_json(STATISTICAL_REPORT_PATH)

    if not summary and not statistical:
        st.warning("Nenhum relatório encontrado. Rode `make pipeline` primeiro.")
        return

    health = summary.get("health", {})
    col1, col2, col3 = st.columns(3)
    col1.metric("Alertas de drift", health.get("drift_alerts_total", 0))
    col2.metric("ROC-AUC (baseline)", f"{health.get('model_auc_baseline', 0):.4f}")
    col3.metric("Queda de ROC-AUC", f"{health.get('model_auc_drop', 0):.4f}")

    auc_values = health.get("model_roc_auc_by_month", [])
    if auc_values:
        st.subheader("Degradação do modelo ao longo do tempo")
        st.line_chart(pd.DataFrame({"ROC-AUC": auc_values}))

    features = statistical.get("features", {})
    if features:
        st.subheader("PSI por feature")
        table = pd.DataFrame(
            {
                "feature": list(features.keys()),
                "psi": [value["psi"] for value in features.values()],
                "severity": [value["severity"] for value in features.values()],
            }
        ).set_index("feature")
        st.dataframe(table, use_container_width=True)

    alerts = summary.get("alerts", [])
    if alerts:
        st.subheader("Alertas ativos")
        st.dataframe(pd.DataFrame(alerts), use_container_width=True)

    if LOG_PATH.exists():
        st.subheader("Logs de execução do pipeline")
        logs = pd.read_json(LOG_PATH, lines=True)
        st.dataframe(logs.tail(50), use_container_width=True)


if __name__ == "__main__":
    main()
