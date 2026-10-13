"""Testes do demo executável de drift (``make demo-drift``, US7 / FR-035).

Cobre (a) o builder **puro** do resumo — PSI/severidade por mês, queda de
ROC-AUC, flags de drift e serialização determinística — e (b) um **smoke
test** rápido de ``run_demo`` sobre uma fixture sintética, com o Evidently
desabilitado, garantindo que a evidência isolada é escrita e que os
artefatos canônicos de ``reports/drift/`` permanecem intactos.
"""

import json
from pathlib import Path

from src.core.params import DemoParams
from src.drift import concept as concept_module
from src.drift import demo as demo_module
from src.drift.demo import (
    build_demo_summary,
    render_demo_summary_markdown,
    render_demo_summary_text,
    run_demo,
)
from src.model.train import build_estimator
from src.observability import logging as logging_module


def _unit_fixture() -> dict:
    """Monta os relatórios de 2 meses usados no teste unitário do builder."""
    history = [
        {
            "month": 1,
            "max_psi": 0.12,
            "n_severe": 0,
            "features": {
                "LIMIT_BAL": {"psi": 0.12, "severity": "moderate"},
                "AGE": {"psi": 0.05, "severity": "stable"},
            },
        },
        {
            "month": 2,
            "max_psi": 0.51,
            "n_severe": 2,
            "features": {
                "LIMIT_BAL": {"psi": 0.51, "severity": "severe"},
                "AGE": {"psi": 0.30, "severity": "severe"},
            },
        },
    ]
    performance = [
        {"month": 1, "roc_auc": 0.70, "concept_drift": 0.02, "reference_auc": 0.72},
        {"month": 2, "roc_auc": 0.61, "concept_drift": 0.11, "reference_auc": 0.72},
    ]
    drift_types = {
        "data_drift": True,
        "label_drift": False,
        "concept_drift": True,
        "multivariate_drift": True,
    }
    alerts = [
        {
            "type": "data_drift",
            "severity": "high",
            "features": ["LIMIT_BAL", "AGE"],
            "windows": 1,
            "message": "2 features com drift severo juntas no mês 2.",
        }
    ]
    report_paths = {
        "summary_json": "reports/drift/demo/demo_summary.json",
        "summary_md": "reports/drift/demo/demo_summary.md",
        "statistical_report": "reports/drift/demo/statistical_report.json",
        "performance": "reports/drift/demo/performance.json",
        "drift_report_html": "reports/drift/demo/drift_report.html",
    }
    return {
        "scenario": "unit_shock",
        "seed": 42,
        "prerequisites": [],
        "history": history,
        "performance": performance,
        "drift_types": drift_types,
        "alerts": alerts,
        "shifted_features": ["LIMIT_BAL", "AGE"],
        "is_breach": True,
        "report_paths": report_paths,
        "generated_at": "2026-10-12T00:00:00+00:00",
    }


def test_build_demo_summary_is_pure_and_consistent():
    """O builder deriva PSI/severidade, queda de AUC e flags dos relatórios."""
    fixture = _unit_fixture()
    summary = build_demo_summary(**fixture)

    assert summary["scenario"] == "unit_shock"
    assert summary["n_snapshots"] == 2
    assert summary["shifted_features"] == ["LIMIT_BAL", "AGE"]
    assert summary["is_breach"] is True
    assert summary["drift_types"] == fixture["drift_types"]
    assert summary["alerts"] == fixture["alerts"]

    last = summary["months"][-1]
    assert last["month"] == 2
    assert last["max_psi"] == 0.51
    assert last["n_severe"] == 2
    assert last["psi_by_feature"]["LIMIT_BAL"]["severity"] == "severe"
    assert last["psi_by_feature"]["AGE"]["psi"] == 0.30
    assert summary["auc_last_month"] == 0.61
    assert abs(summary["auc_drop"] - 0.11) < 1e-9


def test_build_demo_summary_is_deterministic_excluding_generated_at():
    """Com a mesma seed e os mesmos relatórios, o resumo é idêntico (SC-013)."""
    first = build_demo_summary(**_unit_fixture())
    second = build_demo_summary(**_unit_fixture())
    without_timestamp = {key: value for key, value in first.items() if key != "generated_at"}
    second_without = {key: value for key, value in second.items() if key != "generated_at"}
    assert json.dumps(without_timestamp, sort_keys=True) == json.dumps(
        second_without, sort_keys=True
    )


def test_renderers_expose_documented_stdout_sections():
    """Os renderizadores expõem as seções do contrato de stdout."""
    summary = build_demo_summary(**_unit_fixture())

    text = render_demo_summary_text(summary)
    assert "Demonstração de drift" in text
    assert "PSI por feature deslocada" in text
    assert "LIMIT_BAL" in text and "severe" in text
    assert "ROC-AUC por mês" in text
    assert "Tipos de drift" in text
    assert "Alertas consolidados" in text
    assert "open reports/drift/demo/drift_report.html" in text

    markdown = render_demo_summary_markdown(summary)
    assert markdown.startswith("# Demonstração de drift")
    assert "| LIMIT_BAL |" in markdown
    assert "multivariate_drift" in markdown


def test_run_demo_smoke_writes_isolated_evidence(credit_frame, tmp_path, monkeypatch):
    """Smoke test rápido: escreve evidência isolada e não toca o canônico."""
    reference = credit_frame
    target = "default_next_month"
    features = [column for column in reference.columns if column != target]
    estimator = build_estimator()
    estimator.fit(reference[features], reference[target])
    bundle = {
        "model": estimator,
        "features": features,
        "model_name": "test",
        "model_version": "test",
    }
    monkeypatch.setattr(concept_module, "load_model", lambda: bundle)
    monkeypatch.setattr(logging_module, "LOG_PATH", tmp_path / "logs" / "pipeline.ndjson")
    monkeypatch.setattr(demo_module, "TRAIN_METRICS_PATH", tmp_path / "train_metrics.json")

    canonical = Path("reports/drift/statistical_report.json")
    canonical_mtime = canonical.stat().st_mtime_ns if canonical.exists() else None

    output_dir = tmp_path / "reports" / "drift" / "demo"
    snapshots_dir = tmp_path / "data" / "demo"
    demo = DemoParams(
        scenario="smoke",
        n_snapshots=3,
        intensity=0.9,
        concept_drift_strength=1.0,
        output_dir=str(output_dir),
    )

    summary = run_demo(demo, evidently=False, snapshots_dir=snapshots_dir, reference=reference)

    assert (output_dir / "demo_summary.json").exists()
    assert (output_dir / "demo_summary.md").exists()
    assert (output_dir / "statistical_report.json").exists()
    assert (output_dir / "performance.json").exists()
    assert not (output_dir / "drift_report.html").exists()
    assert summary["n_snapshots"] == 3
    assert len(list(snapshots_dir.glob("prod_month_*.parquet"))) == 3

    # Isolamento: nenhum artefato canônico de reports/drift/ foi reescrito.
    if canonical_mtime is not None:
        assert canonical.stat().st_mtime_ns == canonical_mtime
