"""Testes de Concept Drift (Etapa 2, FR-009/FR-013).

Garante que o concept drift muda ``P(y|X)`` **sem** alterar ``P(Y)`` (sem
inversão aleatória de rótulos) e que a performance degrada de forma
monotônica.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from scripts.generate_synthetic_data import generate
from src.drift.concept import apply_concept_drift, fit_concept_model, prior_label_drift

RAW_TARGET = "default payment next month"
TARGET = "default_next_month"


def _reference_frame(tmp_path, n: int = 3000) -> pd.DataFrame:
    """Gera um dataset sintético com sinal e o normaliza como a ingestão."""
    path = tmp_path / "reference.csv"
    generate(path, n_rows=n, seed=42)
    frame = pd.read_csv(path)
    return frame.rename(columns={RAW_TARGET: TARGET}).drop(columns=["ID"])


def _fit_baseline(reference: pd.DataFrame):
    """Ajusta um modelo latente na referência (features padronizadas)."""
    features = reference.drop(columns=[TARGET])
    values = features.to_numpy(dtype=float)
    mean, std = values.mean(axis=0), values.std(axis=0)
    standardized = (values - mean) / std
    model = LogisticRegression(max_iter=2000).fit(standardized, reference[TARGET])
    return model, mean, std


def test_concept_drift_preserves_prior_but_shifts_conditional(tmp_path):
    """P(Y) é preservado; a degradação de P(y|X) aparece na AUC do modelo."""
    reference = _reference_frame(tmp_path)
    model, mean, std = _fit_baseline(reference)
    features = reference.drop(columns=[TARGET])
    standardized = (features.to_numpy(dtype=float) - mean) / std
    base_auc = roc_auc_score(reference[TARGET], model.predict_proba(standardized)[:, 1])

    concept_model = fit_concept_model(reference, TARGET)
    drifted = apply_concept_drift(
        reference, concept_model, progress=1.0, strength=0.5, target=TARGET
    )

    drift = prior_label_drift(reference, drifted, TARGET)
    assert abs(drift["label_drift"]) < 0.02
    drifted_auc = roc_auc_score(drifted[TARGET], model.predict_proba(standardized)[:, 1])
    assert drifted_auc < base_auc - 0.02


def test_concept_drift_degrades_monotonically(tmp_path):
    """A AUC cai de forma monotônica conforme o progresso temporal aumenta."""
    reference = _reference_frame(tmp_path)
    model, mean, std = _fit_baseline(reference)
    features = reference.drop(columns=[TARGET])
    standardized = (features.to_numpy(dtype=float) - mean) / std
    concept_model = fit_concept_model(reference, TARGET)
    aucs = []
    for month in range(1, 7):
        drifted = apply_concept_drift(
            reference, concept_model, progress=month / 6, strength=0.5, target=TARGET
        )
        aucs.append(roc_auc_score(drifted[TARGET], model.predict_proba(standardized)[:, 1]))
    assert aucs[0] > aucs[-1]
    assert all(aucs[i] >= aucs[i + 1] - 1e-6 for i in range(len(aucs) - 1))


def test_concept_drift_is_deterministic(tmp_path):
    """A relabelagem é determinística (sem inversão aleatória de rótulos)."""
    reference = _reference_frame(tmp_path)
    concept_model = fit_concept_model(reference, TARGET)
    first = apply_concept_drift(reference, concept_model, 1.0, 0.5, TARGET)
    second = apply_concept_drift(reference, concept_model, 1.0, 0.5, TARGET)
    pd.testing.assert_series_equal(first[TARGET], second[TARGET])


def test_concept_control_isolates_generator_offset(tmp_path):
    """O controle (progress=0) separa o custo do gerador do Concept Drift real.

    Sem esse controle, a queda bruta de AUC atribui ao Concept Drift o custo
    de substituir os rótulos originais pelo substituto logístico.
    """
    reference = _reference_frame(tmp_path)
    model, mean, std = _fit_baseline(reference)
    features = reference.drop(columns=[TARGET])
    standardized = (features.to_numpy(dtype=float) - mean) / std
    original_auc = roc_auc_score(reference[TARGET], model.predict_proba(standardized)[:, 1])

    concept_model = fit_concept_model(reference, TARGET)
    control = apply_concept_drift(reference, concept_model, 0.0, 0.5, TARGET)
    drifted = apply_concept_drift(reference, concept_model, 1.0, 0.5, TARGET)
    control_auc = roc_auc_score(control[TARGET], model.predict_proba(standardized)[:, 1])
    drifted_auc = roc_auc_score(drifted[TARGET], model.predict_proba(standardized)[:, 1])

    assert control_auc != original_auc
    # O gerador de rótulos tem custo próprio: ignorá-lo enviesa a estimativa
    # de Concept Drift. A correção desloca a queda exatamente por esse custo.
    generator_offset = control_auc - original_auc
    raw_drop = original_auc - drifted_auc
    adjusted_drop = control_auc - drifted_auc
    assert abs(generator_offset) > 0.005
    assert abs(adjusted_drop - raw_drop) > 0.005
