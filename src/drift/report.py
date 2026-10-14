"""Relatório estatístico legível (Markdown) evidencia a mudança de distribuição.

Converte o ``statistical_report.json`` (PSI/KS/χ² por mês) e o
``performance.json`` (ROC-AUC por snapshot) em um documento Markdown
apresentável, cumprindo o entregável "relatório estatístico evidenciando a
mudança de distribuição (ex.: teste PSI ou Kolmogorov-Smirnov)" sem exigir a
leitura do JSON bruto.

O renderizador é **puro** (sem I/O): recebe os dicionários já computados e
devolve o texto, o que o torna trivialmente testável.
"""

from __future__ import annotations

__all__ = ["render_statistical_summary"]


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Renderiza uma tabela Markdown (stdlib apenas)."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _feature_names(report: dict) -> list[str]:
    """Ordem estável das features (as do último mês, com fallback no histórico)."""
    features = list(report.get("features", {}).keys())
    if features:
        return features
    history = report.get("history", [])
    if history:
        return list(history[0].get("features", {}).keys())
    return []


def _psi_matrix(report: dict) -> tuple[list[str], list[list[str]]]:
    """Cabeçalhos/células da matriz PSI (feature × mês + severidade)."""
    history = report.get("history", [])
    headers = ["Feature"] + [
        f"M{entry.get('month', index)}" for index, entry in enumerate(history, 1)
    ]
    rows: list[list[str]] = []
    for feature in _feature_names(report):
        row = [feature]
        for entry in history:
            values = entry.get("features", {}).get(feature)
            row.append(f"{values['psi']:.3f} {values.get('severity', '?')}" if values else "-")
        rows.append(row)
    return headers, rows


def _stat_test_cell(entry: dict) -> tuple[str, str]:
    """Extrai ``(estatística, p-valor)`` do teste KS (numérico) ou χ² (categórico)."""
    if entry.get("type") == "numeric" and "ks" in entry:
        ks = entry["ks"]
        return f"KS={ks['statistic']:.3f}", f"{ks['pvalue']:.2e}"
    if "chi2" in entry:
        chi2 = entry["chi2"]
        return f"χ²={chi2['statistic']:.1f}", f"{chi2['pvalue']:.2e}"
    return "-", "-"


def render_statistical_summary(report: dict, performance: list[dict] | None = None) -> str:
    """Renderiza o resumo estatístico de drift em Markdown.

    Args:
        report: Relatório estatístico do pipeline (``statistical_report.json``)
            com ``history`` (PSI/severidade por mês), ``features`` (testes do
            último mês), ``multivariate`` e ``drift_types``.
        performance: Métricas por mês (``performance.json``) com ``roc_auc``,
            ``concept_drift`` e ``label_drift``.

    Returns:
        Documento Markdown pronto para gravação em
        ``reports/drift/statistical_summary.md``.
    """
    performance = performance or []
    psi_bands = report.get("psi_bands", {"stable": 0.1, "moderate": 0.25})
    month = report.get("month", len(report.get("history", [])))
    is_breach = bool(report.get("is_breach", False))
    n_drifted = report.get("n_drifted", 0)
    max_psi = float(report.get("max_psi", 0.0))

    lines: list[str] = []
    lines.append("# Relatório estatístico de drift — referência × produção")
    lines.append("")
    lines.append(
        "Mudança de distribuição entre o dataset de **referência** (treino) e os "
        "**snapshots de produção** (Data + Concept Drift), medida por **PSI** "
        "(métrica de decisão) ao lado de **KS/χ²** (evidência)."
    )
    lines.append("")
    lines.append(f"- **Snapshots analisados:** {len(report.get('history', []))}")
    lines.append(f"- **Mês destacado:** {month}")
    lines.append(
        f"- **Breach de PSI (limiar {psi_bands.get('moderate', 0.25)}):** "
        f"{'sim' if is_breach else 'não'}"
    )
    lines.append(f"- **PSI máximo:** {max_psi:.3f}")
    lines.append(f"- **Features com drift severo (mês {month}):** {n_drifted}")
    lines.append("")

    lines.append("## PSI por feature e mês (severidade: stable/moderate/severe)")
    lines.append("")
    headers, rows = _psi_matrix(report)
    lines.extend(_markdown_table(headers, rows))
    lines.append("")

    lines.append(f"## Testes estatísticos do mês {month} (KS numérico / χ² categórico)")
    lines.append("")
    test_rows: list[list[str]] = []
    for feature, entry in report.get("features", {}).items():
        statistic, pvalue = _stat_test_cell(entry)
        test_rows.append(
            [
                feature,
                str(entry.get("type", "-")),
                f"{float(entry.get('psi', 0.0)):.3f}",
                str(entry.get("severity", "-")),
                statistic,
                pvalue,
            ]
        )
    lines.extend(
        _markdown_table(
            ["Feature", "Tipo", "PSI", "Severidade", "Estatística", "p-valor"], test_rows
        )
    )
    lines.append("")

    multivariate = report.get("multivariate", {})
    if multivariate:
        lines.append("## Drift multivariado (MMD² / Energy)")
        lines.append("")
        lines.append(
            f"Limiar calibrado por bootstrap (n={multivariate.get('n_bootstrap', '?')}, "
            f"quantil {multivariate.get('mmd2_threshold_quantile', '?')})."
        )
        lines.append("")
        lines.extend(
            _markdown_table(
                ["Métrica", "Valor", "Limiar", "Driftado"],
                [
                    [
                        "MMD²",
                        f"{float(multivariate.get('mmd2', 0.0)):.4f}",
                        f"{float(multivariate.get('mmd2_threshold', 0.0)):.4f}",
                        str(bool(multivariate.get("drifted", False))),
                    ],
                    [
                        "Energy",
                        f"{float(multivariate.get('energy', 0.0)):.4f}",
                        f"{float(multivariate.get('energy_threshold', 0.0)):.4f}",
                        str(bool(multivariate.get("drifted", False))),
                    ],
                ],
            )
        )
        lines.append("")

    if performance:
        lines.append("## Performance do modelo por mês")
        lines.append("")
        perf_rows: list[list[str]] = []
        for entry in performance:
            auc_drop = entry.get("reference_auc", 0.0) - entry.get("roc_auc", 0.0)
            perf_rows.append(
                [
                    f"M{entry.get('month', '?')}",
                    f"{float(entry.get('roc_auc', 0.0)):.4f}",
                    f"{auc_drop:.4f}",
                    f"{float(entry.get('label_drift', 0.0)):.4f}",
                ]
            )
        lines.extend(
            _markdown_table(
                ["Mês", "ROC-AUC", "Queda vs baseline (concept drift)", "Label drift P(Y)"],
                perf_rows,
            )
        )
        lines.append("")

    drift_types = report.get("drift_types", {})
    if drift_types:
        lines.append("## Tipos de drift")
        lines.append("")
        lines.extend(
            _markdown_table(
                ["Tipo", "Detectado"],
                [[name, str(value)] for name, value in drift_types.items()],
            )
        )
        lines.append("")

    lines.append(
        "> Proveniência: o **PSI do projeto** (binning quantílico) é a métrica de "
        "decisão e dirige o breach/SLO. O PSI reportado no HTML do Evidently usa "
        "binning próprio e pode divergir numericamente — é evidência, não decisão."
    )
    return "\n".join(lines) + "\n"
