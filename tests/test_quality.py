"""Testes de qualidade de dados (Etapa 1, FR-003/FR-004).

Cobre deduplicação antes do split, ausência de vazamento treino/teste,
diagnóstico de ausências (MCAR/MAR/MNAR), detecção de outliers e as
dimensões de qualidade do relatório.
"""

import numpy as np
import pandas as pd

from src.contracts.quality import compute_quality
from src.data.dedup import deduplicate, row_hash
from src.data.missingness import diagnose_missingness
from src.data.outliers import detect_outliers
from src.data.split import split_reference
from tests.conftest import make_credit_frame


def test_compute_quality_dimensions(credit_frame):
    """O bloco de qualidade expõe as cinco dimensões e decisões padrão."""
    quality = compute_quality(credit_frame)
    for key in ("completeness", "uniqueness", "timeliness", "consistency", "validity"):
        assert key in quality
    assert quality["completeness"] == 1.0
    assert quality["validity"] == 1.0
    assert all(decision == "keep" for decision in quality["decisions"].values())


def test_quality_decision_override_is_recorded_without_mutation(monkeypatch):
    """O override `quality.decisions` é auditado, mas não muta os dados (flag-and-keep)."""
    import src.contracts.quality as quality_module
    from src.core.params import load_params

    overridden = load_params().model_copy(deep=True)
    overridden.quality.decisions = {"AGE": "remove"}
    monkeypatch.setattr(quality_module, "load_params", lambda: overridden)

    frame = make_credit_frame(n_rows=20, seed=3)
    before = frame.copy()

    quality = quality_module.compute_quality(frame)

    assert quality["decisions"]["AGE"] == "remove"
    assert "AGE" in frame.columns
    pd.testing.assert_frame_equal(frame, before)


def test_dedup_removes_exact_and_near_duplicates():
    """Dedup remove duplicatas exatas (SHA-256) e near-duplicates (MinHash/LSH)."""
    frame = make_credit_frame(n_rows=40, seed=7)
    exact_dup = frame.iloc[[0]].copy()
    near_dup = frame.iloc[[1]].copy()
    near_dup["BILL_AMT1"] = near_dup["BILL_AMT1"] + 123.0
    combined = pd.concat([frame, exact_dup, near_dup], ignore_index=True)

    deduplicated, report = deduplicate(combined, near_dup_threshold=0.6)

    assert report["exact_duplicates_removed"] == 1
    assert report["near_duplicates_removed"] == 1
    assert report["total_removed"] == 2
    assert len(deduplicated) == len(combined) - 2


def test_dedup_before_split_has_no_leakage():
    """Após dedup+split, nenhuma linha é compartilhada entre treino e validação."""
    frame = make_credit_frame(n_rows=200, seed=11)
    deduplicated, _ = deduplicate(frame, use_near_dup=False)
    train, holdout = split_reference(deduplicated)
    train_hashes = set(row_hash(train))
    holdout_hashes = set(row_hash(holdout))
    assert train_hashes.isdisjoint(holdout_hashes)
    assert len(train) + len(holdout) == len(deduplicated)


def test_missingness_diagnoses_mechanism_and_fields():
    """O diagnóstico classifica MNAR quando a ausência depende do alvo."""
    frame = make_credit_frame(n_rows=200, seed=5)
    frame.loc[frame["default_next_month"] == 1, "AGE"] = np.nan

    report = diagnose_missingness(frame)

    age = report["columns"]["AGE"]
    assert age["n_missing"] > 0
    assert age["missing_rate"] > 0
    assert age["mechanism"] == "MNAR"
    assert age["decision"] == "keep"
    assert report["columns"]["LIMIT_BAL"]["mechanism"] == "none"


def test_outlier_detection_multiple_methods():
    """A detecção usa ≥3 métodos e sinaliza extremos sem removê-los."""
    frame = make_credit_frame(n_rows=100, seed=13).copy()
    frame.loc[0, "BILL_AMT1"] = 1e9
    frame.loc[1, "PAY_AMT1"] = 1e9

    report = detect_outliers(frame)

    assert set(report["methods"]) >= {"zscore", "iqr", "lof", "isolation_forest"}
    assert report["columns"]["BILL_AMT1"]["n_outliers_zscore"] >= 1
    assert report["columns"]["BILL_AMT1"]["n_outliers_iqr"] >= 1
    assert "n_outliers_lof" in report["multivariate"]
    assert "n_outliers_isolation_forest" in report["multivariate"]
    assert all(decision == "keep" for decision in report["decisions"].values())
