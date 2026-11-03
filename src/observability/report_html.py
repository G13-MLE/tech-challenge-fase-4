"""Relatório HTML autocontido de saúde operacional (Etapa 3).

O edital pede "dashboards **ou relatórios HTML gerados** comprovando o
monitoramento". O dashboard Streamlit e o Grafana são interativos, mas não
deixam artefato versionável; este renderizador produz um HTML único, sem
dependências externas e determinístico, a partir dos mesmos JSONs que
alimentam o exporter e o dashboard.

Escrito por ``src.observability.run`` em ``reports/monitoring/health.html``
e commitado como evidência.
"""

from __future__ import annotations

import html

__all__ = ["render_health_html"]

_STATUS_COLOR = {
    "ok": "#1a7f37",
    "degraded": "#9a6700",
    "critical": "#cf222e",
    "retrain": "#cf222e",
    "no_action": "#1a7f37",
}

_STYLE = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       margin: 0; padding: 32px; background: #f6f8fa; color: #1f2328; }
h1 { margin: 0 0 4px; font-size: 26px; }
h2 { margin: 32px 0 8px; font-size: 18px; border-bottom: 1px solid #d0d7de; padding-bottom: 4px; }
.sub { color: #59636e; margin: 0 0 24px; font-size: 14px; }
.cards { display: flex; flex-wrap: wrap; gap: 12px; }
.card { background: #fff; border: 1px solid #d0d7de; border-radius: 8px; padding: 14px 18px;
        min-width: 150px; }
.card .label { color: #59636e; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
.card .value { font-size: 24px; font-weight: 600; margin-top: 4px; }
table { border-collapse: collapse; width: 100%; background: #fff; font-size: 13px; }
th, td { border: 1px solid #d0d7de; padding: 6px 10px; text-align: left; }
th { background: #eef1f4; }
tr:nth-child(even) td { background: #fafbfc; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 999px; color: #fff;
         font-size: 13px; font-weight: 600; }
.severe { color: #cf222e; font-weight: 600; }
.moderate { color: #9a6700; }
.stable { color: #1a7f37; }
footer { margin-top: 32px; color: #59636e; font-size: 12px; }
"""


def _escape(value: object) -> str:
    """Escapa um valor para inserção segura em HTML."""
    return html.escape(str(value))


def _badge(text: str, key: str | None = None) -> str:
    """Renderiza um selo colorido pelo status."""
    color = _STATUS_COLOR.get((key or text).lower(), "#59636e")
    return f'<span class="badge" style="background:{color}">{_escape(text)}</span>'


def _card(label: str, value: object) -> str:
    """Renderiza um cartão de KPI."""
    return (
        '<div class="card">'
        f'<div class="label">{_escape(label)}</div>'
        f'<div class="value">{_escape(value)}</div>'
        "</div>"
    )


def _table(headers: list[str], rows: list[list[object]]) -> str:
    """Renderiza uma tabela HTML simples."""
    head = "".join(f"<th>{_escape(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_escape(cell)}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_health_html(
    summary: dict,
    statistical: dict,
    performance: list[dict],
    slo: dict,
    fairness: dict | None = None,
) -> str:
    """Renderiza o relatório de saúde completo como HTML autocontido.

    Args:
        summary: Conteúdo de ``reports/monitoring/run_summary.json``.
        statistical: Último ``statistical_report.json`` (com ``history``).
        performance: ``performance.json`` (métricas por mês).
        slo: ``slo_report.json``.
        fairness: ``fairness_report.json`` (opcional).

    Returns:
        Documento HTML completo, pronto para ser escrito atomicamente.
    """
    health = summary.get("health", {})
    tasks = health.get("tasks", {})
    alerts = summary.get("alerts", [])
    status = summary.get("slo", {}).get("status_geral", slo.get("status_geral", "n/a"))

    cards = [
        _card("Status SLO", status),
        _card("Alertas consolidados", len(alerts)),
        _card("Features com drift", health.get("drift_features_drifted", 0)),
        _card("PSI máximo", f"{float(statistical.get('max_psi', 0.0)):.4f}"),
        _card("ROC-AUC baseline", f"{float(health.get('model_auc_baseline', 0.0)):.4f}"),
        _card("ROC-AUC último mês", f"{float(health.get('model_auc_last', 0.0)):.4f}"),
        _card("Queda de ROC-AUC", f"{float(health.get('model_auc_drop', 0.0)):.4f}"),
        _card("Decisão de re-treino", summary.get("retrain_decision", "n/a")),
        _card("Fairness", "ok" if summary.get("fairness_passed") else "violado"),
    ]

    sections = []

    # ── Tarefas do pipeline ───────────────────────────────────────────
    last_status = tasks.get("last_status", {})
    if last_status:
        rows = [
            [task, status, "falhou" if status == "failed" else "ok"]
            for task, status in sorted(last_status.items())
        ]
        sections.append("<h2>Tarefas do pipeline (última execução)</h2>")
        sections.append(_table(["Tarefa", "Status", "Resultado"], rows))

    # ── Alertas ───────────────────────────────────────────────────────
    if alerts:
        rows = [
            [
                alert.get("type", ""),
                alert.get("severity", ""),
                alert.get("feature", alert.get("task", "")),
                alert.get("month", ""),
                alert.get("message", ""),
            ]
            for alert in alerts
        ]
        sections.append("<h2>Alertas consolidados (anti-fadiga)</h2>")
        sections.append(_table(["Tipo", "Severidade", "Feature/Tarefa", "Mês", "Mensagem"], rows))

    # ── PSI por feature × mês ─────────────────────────────────────────
    history = statistical.get("history", [])
    if history:
        features = sorted(history[-1].get("features", {}).keys())
        months = [entry.get("month") for entry in history]
        rows = []
        for feature in features:
            row: list[object] = [feature]
            for entry in history:
                values = entry.get("features", {}).get(feature, {})
                row.append(f"{float(values.get('psi', 0.0)):.3f} ({values.get('severity', '-')})")
            rows.append(row)
        sections.append("<h2>PSI por feature × mês</h2>")
        sections.append(_table(["Feature"] + [f"Mês {month}" for month in months], rows))

    # ── Performance ao longo do tempo ─────────────────────────────────
    if performance:
        rows = [
            [
                entry.get("month", ""),
                f"{float(entry.get('roc_auc', 0.0)):.4f}",
                f"{float(entry.get('brier', 0.0)):.4f}",
                f"{float(entry.get('concept_drift', 0.0)):.4f}",
                f"{float(entry.get('concept_drift_adjusted', 0.0)):.4f}",
            ]
            for entry in performance
        ]
        sections.append("<h2>Performance por snapshot (degradação do modelo)</h2>")
        sections.append(
            _table(
                [
                    "Mês",
                    "ROC-AUC",
                    "Brier",
                    "Queda vs baseline",
                    "Concept drift (ajustado)",
                ],
                rows,
            )
        )

    # ── SLO / Error Budget ────────────────────────────────────────────
    checks = slo.get("checks", [])
    if checks:
        rows = [
            [
                check.get("slo", ""),
                check.get("value", ""),
                check.get("threshold", ""),
                f"{float(check.get('error_budget', 0.0)):+.4f}",
                check.get("status", ""),
            ]
            for check in checks
        ]
        sections.append("<h2>Data SLOs e Error Budget</h2>")
        sections.append(_table(["SLO", "Valor", "Limite", "Error budget", "Status"], rows))

    # ── Fairness ──────────────────────────────────────────────────────
    by_group = (fairness or {}).get("by_group", {})
    if by_group:
        rows = [
            [
                column,
                f"{float(entry.get('disparate_impact_ratio', 0.0)):.4f}",
                f"{float(entry.get('equal_opportunity_diff', 0.0)):.4f}",
                "passa" if entry.get("passes") else "viola",
            ]
            for column, entry in sorted(by_group.items())
        ]
        sections.append("<h2>Fairness por subgrupo</h2>")
        sections.append(
            _table(["Subgrupo", "Impacto dispar", "Equal opportunity", "Guardrail"], rows)
        )

    return (
        "<!DOCTYPE html>\n"
        '<html lang="pt-BR"><head><meta charset="utf-8">'
        "<title>Observabilidade — Credit Scoring</title>"
        f"<style>{_STYLE}</style></head><body>"
        "<h1>Observabilidade — Credit Scoring</h1>"
        '<p class="sub">Saúde do pipeline, drift, SLO, fairness e alertas ·'
        f" gerado pelo estágio <code>monitoramento</code> · {_badge(status, status)}</p>"
        f'<div class="cards">{"".join(cards)}</div>'
        f"{''.join(sections)}"
        "<footer>Fontes: <code>reports/monitoring/*.json</code>,"
        " <code>reports/drift/*.json</code> e <code>reports/fairness/*.json</code>."
        " Relatório interativo: <code>dashboard/app.py</code> (Streamlit) e"
        " Grafana (<code>make monitoring-up</code>).</footer>"
        "</body></html>"
    )
