"""Demo executável de drift: ``make demo-drift`` (US7 / FR-029–FR-035).

Camada de **apresentação** sobre os módulos de drift existentes: simula um
choque econômico amplificado, roda a pilha completa de detecção (Data +
Concept Drift, PSI/KS/χ², MMD²/Energy e Evidently) e **imprime um resumo
legível** no terminal, mostrando o drift acontecendo.

A evidência é isolada sob ``reports/drift/demo/`` (snapshots em
``data/demo/``): os artefatos canônicos ``reports/drift/*`` e
``data/production/prod_month_*.parquet`` nunca são tocados (FR-031). O demo
é autocontido (prepara referência/modelo se ausentes, FR-030), determinístico
(FR-033), emite logs JSON-lines com a task ``demo_drift`` mas **não** registra
run no MLflow (FR-034) nem emite relatório de SLO (alertas em memória).
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.core.dataset import (
    DATA_RAW_PATH,
    DEMO_DRIFT_REPORT_PATH,
    DEMO_PERFORMANCE_PATH,
    DEMO_SNAPSHOTS_DIR,
    DEMO_STATISTICAL_REPORT_PATH,
    DEMO_SUMMARY_MD_PATH,
    DEMO_SUMMARY_PATH,
    MODEL_PATH,
    REFERENCE_DATA_PATH,
    TRAIN_METRICS_PATH,
    atomic_write_text,
    load_parquet,
)
from src.core.params import DemoParams, load_params
from src.drift.concept import evaluate_snapshots, prior_label_drift
from src.drift.detect import build_evidently_report
from src.drift.multivariate import build_multivariate_report
from src.drift.simulate import simulate_snapshots
from src.drift.stats import build_statistical_report
from src.observability.alerts import build_alerts
from src.observability.metrics import aggregate_health

logger = logging.getLogger(__name__)

__all__ = [
    "DemoPrerequisiteError",
    "build_demo_summary",
    "render_demo_summary_text",
    "render_demo_summary_markdown",
    "run_demo",
    "main",
]


class DemoPrerequisiteError(RuntimeError):
    """Pré-requisito ausente e não recuperável (sem dataset bruto/rede)."""


def _month_history_entry(month: int, report: dict) -> dict:
    """Resume um relatório estatístico mensal para a série temporal do demo."""
    return {
        "month": month,
        "n_drifted": report["n_drifted"],
        "n_severe": report["n_severe"],
        "max_psi": report["max_psi"],
        "is_breach": report["is_breach"],
        "features": {
            name: {"psi": entry["psi"], "severity": entry["severity"]}
            for name, entry in report["features"].items()
        },
    }


def _now_iso() -> str:
    """Instante ISO-8601 (UTC) do resumo; excluído da checagem de determinismo."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def build_demo_summary(
    *,
    scenario: str,
    seed: int,
    prerequisites: list[str],
    history: list[dict],
    performance: list[dict],
    drift_types: dict,
    alerts: list[dict],
    shifted_features: list[str],
    is_breach: bool,
    report_paths: dict,
    generated_at: str,
) -> dict:
    """Constrói o ``DemoSummary`` (E14) a partir de relatórios já computados.

    Função **pura** (sem I/O): recebe a série temporal de PSI, a performance
    por mês e os flags/alertas e devolve o objeto serializável. Permite
    testar o resumo com uma fixture minúscula, sem rodar o pipeline.

    Args:
        scenario: Rótulo do cenário (apenas exibição).
        seed: Seed determinística usada na simulação.
        prerequisites: Ações de pré-requisito executadas (``ingestao``/``treino``).
        history: Entradas mensais com ``features[col].severity`` e ``max_psi``.
        performance: Métricas por mês (``roc_auc``, ``concept_drift``).
        drift_types: Flags ``data_drift``/``label_drift``/``concept_drift``/
            ``multivariate_drift``.
        alerts: Alertas consolidados anti-fadiga.
        shifted_features: Features intencionalmente deslocadas (linhas da tabela PSI).
        is_breach: True quando alguma feature tem PSI > limiar moderado.
        report_paths: Caminhos da evidência isolada do demo.
        generated_at: Instante ISO-8601 (excluído da comparação de determinismo).

    Returns:
        Dicionário do resumo (``demo_summary.json``).
    """
    performance_by_month = {
        int(entry.get("month", index)): entry for index, entry in enumerate(performance, start=1)
    }
    months: list[dict] = []
    for entry in history:
        month = int(entry["month"])
        metrics = performance_by_month.get(month, {})
        months.append(
            {
                "month": month,
                "max_psi": float(entry.get("max_psi", 0.0)),
                "n_severe": int(entry.get("n_severe", 0)),
                "roc_auc": float(metrics.get("roc_auc", 0.0)),
                "auc_drop": float(metrics.get("concept_drift", 0.0)),
                "psi_by_feature": {
                    name: {
                        "psi": float(values.get("psi", 0.0)),
                        "severity": str(values.get("severity", "stable")),
                    }
                    for name, values in entry.get("features", {}).items()
                },
            }
        )
    last = months[-1] if months else {}
    return {
        "scenario": scenario,
        "seed": seed,
        "prerequisites": list(prerequisites),
        "n_snapshots": len(months),
        "months": months,
        "shifted_features": list(shifted_features),
        "drift_types": dict(drift_types),
        "alerts": list(alerts),
        "is_breach": bool(is_breach),
        "auc_last_month": float(last.get("roc_auc", 0.0)),
        "auc_drop": float(last.get("auc_drop", 0.0)),
        "report_paths": dict(report_paths),
        "generated_at": generated_at,
    }


def _psi_rows(summary: dict) -> tuple[list[str], list[list[str]]]:
    """Monta cabeçalhos/células da tabela PSI (feature × mês + severidade)."""
    months = summary.get("months", [])
    features = summary.get("shifted_features") or (
        list(months[0]["psi_by_feature"]) if months else []
    )
    headers = ["Feature"] + [f"M{row['month']}" for row in months]
    rows: list[list[str]] = []
    for feature in features:
        row = [str(feature)]
        for entry in months:
            values = entry.get("psi_by_feature", {}).get(feature)
            row.append(f"{values['psi']:.3f} {values['severity']}" if values else "-")
        rows.append(row)
    return headers, rows


def _text_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Renderiza uma tabela de texto alinhada (stdlib apenas)."""
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def render(row: list[str]) -> str:
        return "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)).rstrip()

    lines = [render(headers), "  ".join("-" * width for width in widths)]
    lines.extend(render(row) for row in rows)
    return lines


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Renderiza uma tabela Markdown."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _auc_baseline(summary: dict) -> float:
    """ROC-AUC de referência reconstruído (AUC do último mês + queda)."""
    return float(summary.get("auc_last_month", 0.0) + summary.get("auc_drop", 0.0))


def render_demo_summary_text(summary: dict) -> str:
    """Renderiza o resumo do demo em texto (ordem documentada do stdout).

    Args:
        summary: Objeto retornado por :func:`build_demo_summary`.

    Returns:
        Texto pronto para impressão no terminal.
    """
    lines: list[str] = []
    prerequisites = summary.get("prerequisites") or []
    suffix = f" | pré-requisitos: {', '.join(prerequisites)}" if prerequisites else ""
    lines.append(f"=== Demonstração de drift: {summary['scenario']} ===")
    lines.append(f"Meses: {summary['n_snapshots']} | seed: {summary['seed']}{suffix}")
    lines.append("")

    lines.append("PSI por feature deslocada (severidade: stable/moderate/severe)")
    headers, rows = _psi_rows(summary)
    lines.extend(_text_table(headers, rows))
    lines.append("")

    months = summary.get("months", [])
    auc_cells = "  ".join(f"M{row['month']}={row['roc_auc']:.4f}" for row in months)
    lines.append(f"ROC-AUC por mês (baseline de referência: {_auc_baseline(summary):.4f})")
    lines.append(f"  {auc_cells}")
    lines.append(f"Queda vs baseline (último mês): {summary.get('auc_drop', 0.0):.4f}")
    lines.append("")

    flags = summary.get("drift_types", {})
    lines.append("Tipos de drift: " + "  ".join(f"{name}={value}" for name, value in flags.items()))
    lines.append("")

    alerts = summary.get("alerts", [])
    lines.append(f"Alertas consolidados ({len(alerts)}):")
    if alerts:
        for alert in alerts:
            entry = (
                f"  - [{alert.get('severity', '?')}] {alert.get('type', '?')}: "
                f"{alert.get('message', '')}"
            )
            lines.append(entry.rstrip())
    else:
        lines.append("  (nenhum)")
    lines.append("")

    paths = summary.get("report_paths", {})
    lines.append("Relatórios (evidência isolada):")
    report_keys = (
        "summary_json",
        "summary_md",
        "statistical_report",
        "performance",
        "drift_report_html",
    )
    for key in report_keys:
        if key in paths:
            lines.append(f"  {key}: {paths[key]}")
    if paths.get("drift_report_html"):
        lines.append(f"Abra o relatório: open {paths['drift_report_html']}")
    return "\n".join(lines) + "\n"


def render_demo_summary_markdown(summary: dict) -> str:
    """Renderiza o resumo do demo em Markdown (mesmo conteúdo do texto).

    Args:
        summary: Objeto retornado por :func:`build_demo_summary`.

    Returns:
        Documento Markdown.
    """
    lines: list[str] = []
    prerequisites = summary.get("prerequisites") or []
    lines.append(f"# Demonstração de drift — `{summary['scenario']}`")
    lines.append("")
    lines.append(f"- **Meses:** {summary['n_snapshots']}")
    lines.append(f"- **Seed:** {summary['seed']}")
    if prerequisites:
        lines.append(f"- **Pré-requisitos executados:** {', '.join(prerequisites)}")
    lines.append("")

    lines.append("## PSI por feature deslocada")
    lines.append("")
    headers, rows = _psi_rows(summary)
    lines.extend(_markdown_table(headers, rows))
    lines.append("")

    months = summary.get("months", [])
    lines.append("## ROC-AUC por mês")
    lines.append("")
    lines.append(f"Baseline de referência: **{_auc_baseline(summary):.4f}**.")
    lines.append("")
    lines.extend(
        _markdown_table(
            ["Mês", "ROC-AUC", "Queda vs baseline"],
            [
                [f"M{row['month']}", f"{row['roc_auc']:.4f}", f"{row['auc_drop']:.4f}"]
                for row in months
            ],
        )
    )
    lines.append("")

    flags = summary.get("drift_types", {})
    lines.append("## Tipos de drift")
    lines.append("")
    flag_rows = [[name, str(value)] for name, value in flags.items()]
    lines.extend(_markdown_table(["Tipo", "Detectado"], flag_rows))
    lines.append("")

    alerts = summary.get("alerts", [])
    lines.append(f"## Alertas consolidados ({len(alerts)})")
    lines.append("")
    if alerts:
        for alert in alerts:
            entry = (
                f"- **[{alert.get('severity', '?')}] {alert.get('type', '?')}** — "
                f"{alert.get('message', '')}"
            )
            lines.append(entry.rstrip())
    else:
        lines.append("- (nenhum)")
    lines.append("")

    paths = summary.get("report_paths", {})
    lines.append("## Evidência isolada")
    lines.append("")
    lines.extend(f"- `{key}`: `{value}`" for key, value in paths.items())
    return "\n".join(lines) + "\n"


def _ensure_prerequisites() -> list[str]:
    """Prepara referência/modelo se ausentes (FR-030).

    Returns:
        Lista das ações executadas (``ingestao``, ``treino``).

    Raises:
        DemoPrerequisiteError: Se o dataset bruto (DVC) também estiver ausente.
    """
    actions: list[str] = []
    if not REFERENCE_DATA_PATH.exists():
        if not DATA_RAW_PATH.exists():
            raise DemoPrerequisiteError(
                "Dataset de referência ausente "
                f"({REFERENCE_DATA_PATH}) e dataset bruto ausente ({DATA_RAW_PATH}). "
                "Rode `make data-download` (UCI 350) ou `make data-synthetic` "
                "(fallback offline) antes de `make demo-drift`."
            )
        from src.data.ingest import ingest

        ingest()
        actions.append("ingestao")
    if not MODEL_PATH.exists():
        from src.model.train import train

        train()
        actions.append("treino")
    return actions


def _baseline_auc() -> float | None:
    """ROC-AUC de referência do modelo campeão (``reports/train_metrics.json``)."""
    if TRAIN_METRICS_PATH.exists():
        payload = json.loads(TRAIN_METRICS_PATH.read_text(encoding="utf-8"))
        return float(payload.get("roc_auc", 0.0))
    return None


def run_demo(
    demo: DemoParams | None = None,
    *,
    evidently: bool = True,
    snapshots_dir: Path | None = None,
    reference: pd.DataFrame | None = None,
) -> dict:
    """Executa o demo de drift ponta a ponta e escreve a evidência isolada.

    Args:
        demo: Parâmetros do cenário; se None, usa ``params.demo``.
        evidently: Se True, gera o HTML do Evidently (default). ``--no-evidently``
            usa False (pula o builder, mantendo os demais relatórios).
        snapshots_dir: Diretório dos snapshots do demo (default ``data/demo/``).
        reference: Referência pré-carregada; se None, prepara/ carrega do disco.

    Returns:
        O ``DemoSummary`` (também gravado em ``demo_summary.json``).
    """
    params = load_params()
    demo = demo or params.demo
    output_dir = Path(demo.output_dir)
    snapshot_dir = Path(snapshots_dir) if snapshots_dir is not None else DEMO_SNAPSHOTS_DIR

    prerequisites: list[str] = []
    if reference is None:
        prerequisites = _ensure_prerequisites()
        reference = load_parquet(REFERENCE_DATA_PATH)
    baseline_auc = _baseline_auc()

    paths = simulate_snapshots(
        destination_dir=snapshot_dir,
        intensity=demo.intensity,
        concept_drift_strength=demo.concept_drift_strength,
        n_snapshots=demo.n_snapshots,
        seed=demo.seed,
        reference=reference,
    )
    snapshots = [load_parquet(path) for path in paths]

    history: list[dict] = []
    last_report: dict = {}
    for month, snapshot in enumerate(snapshots, start=1):
        report = build_statistical_report(reference, snapshot, write=False)
        history.append(_month_history_entry(month, report))
        last_report = report

    last_snapshot = snapshots[-1]
    last_report["history"] = history
    last_report["month"] = len(snapshots)
    last_report["multivariate"] = build_multivariate_report(reference, last_snapshot)

    performance = evaluate_snapshots(snapshots, baseline_auc=baseline_auc)
    label = prior_label_drift(reference, last_snapshot)
    label_by_month = {
        month: prior_label_drift(reference, snapshot)
        for month, snapshot in enumerate(snapshots, start=1)
    }
    for entry in performance:
        drift = label_by_month[entry["month"]]
        entry["p_positive"] = drift["production_positive_rate"]
        entry["label_drift"] = drift["label_drift"]

    drift_types = {
        "data_drift": bool(last_report["n_drifted"] > 0),
        "label_drift": bool(abs(label["label_drift"]) > 0.02),
        "concept_drift": bool(performance[-1]["concept_drift"] > 0.0),
        "multivariate_drift": bool(last_report["multivariate"]["drifted"]),
    }

    health = aggregate_health(last_report, performance)
    alerts = build_alerts(last_report, health, history)

    report_paths = {
        "summary_json": str(output_dir / DEMO_SUMMARY_PATH.name),
        "summary_md": str(output_dir / DEMO_SUMMARY_MD_PATH.name),
        "statistical_report": str(output_dir / DEMO_STATISTICAL_REPORT_PATH.name),
        "performance": str(output_dir / DEMO_PERFORMANCE_PATH.name),
        "drift_report_html": str(output_dir / DEMO_DRIFT_REPORT_PATH.name),
        "snapshots_dir": str(snapshot_dir),
    }

    summary = build_demo_summary(
        scenario=demo.scenario,
        seed=demo.seed,
        prerequisites=prerequisites,
        history=history,
        performance=performance,
        drift_types=drift_types,
        alerts=alerts,
        shifted_features=[
            column for column in params.drift.shifted_features if column in reference.columns
        ],
        is_breach=bool(last_report["is_breach"]),
        report_paths=report_paths,
        generated_at=_now_iso(),
    )
    summary["n_rows"] = int(sum(len(snapshot) for snapshot in snapshots))

    atomic_write_text(
        output_dir / DEMO_STATISTICAL_REPORT_PATH.name, json.dumps(last_report, indent=2)
    )
    atomic_write_text(output_dir / DEMO_PERFORMANCE_PATH.name, json.dumps(performance, indent=2))
    atomic_write_text(
        output_dir / DEMO_SUMMARY_PATH.name, json.dumps(summary, indent=2, ensure_ascii=False)
    )
    atomic_write_text(output_dir / DEMO_SUMMARY_MD_PATH.name, render_demo_summary_markdown(summary))

    if evidently:
        build_evidently_report(
            reference, last_snapshot, html_path=output_dir / DEMO_DRIFT_REPORT_PATH.name
        )

    logger.info("Demo de drift '%s' concluído (%d meses).", demo.scenario, len(snapshots))
    return summary


def _build_parser() -> argparse.ArgumentParser:
    """Constrói o parser de argumentos do demo (overrides do bloco ``demo:``)."""
    parser = argparse.ArgumentParser(
        prog="python -m src.drift.demo",
        description="Demo executável de Data + Concept Drift (make demo-drift).",
    )
    parser.add_argument("--scenario", default=None, help="Rótulo do cenário (apenas exibição).")
    parser.add_argument("--months", type=int, default=None, help="Número de meses simulados (≥ 2).")
    parser.add_argument(
        "--intensity", type=float, default=None, help="Intensidade do Data Drift ∈ [0, 1]."
    )
    parser.add_argument(
        "--concept-strength",
        type=float,
        default=None,
        help="Magnitude do Concept Drift P(y|X) ∈ [0, 2].",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Diretório de evidência isolada (default reports/drift/demo).",
    )
    parser.add_argument(
        "--no-evidently",
        action="store_true",
        help="Pula o HTML do Evidently (execução mais rápida).",
    )
    return parser


def _resolve_demo_params(args: argparse.Namespace) -> DemoParams:
    """Aplica os overrides de CLI sobre o bloco ``demo:`` (FR-032)."""
    base = load_params().demo
    updates: dict[str, object] = {}
    if args.scenario is not None:
        updates["scenario"] = args.scenario
    if args.months is not None:
        updates["n_snapshots"] = args.months
    if args.intensity is not None:
        updates["intensity"] = args.intensity
    if args.concept_strength is not None:
        updates["concept_drift_strength"] = args.concept_strength
    if args.output_dir is not None:
        updates["output_dir"] = args.output_dir
    return base.model_copy(update=updates)


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada do demo (``make demo-drift``).

    Args:
        argv: Argumentos de linha de comando (default ``sys.argv``).

    Returns:
        Código de saída: 0 em sucesso; 1 se um pré-requisito for irrecuperável.
    """
    from src.observability.logging import TaskTimer

    args = _build_parser().parse_args(argv)
    demo = _resolve_demo_params(args)
    try:
        with TaskTimer("demo_drift", scenario=demo.scenario) as timer:
            summary = run_demo(demo, evidently=not args.no_evidently)
            timer.record(n_rows=summary.get("n_rows", 0))
    except DemoPrerequisiteError as exc:
        logger.error("%s", exc)
        print(f"[demo-drift] {exc}")
        return 1
    print(render_demo_summary_text(summary))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
