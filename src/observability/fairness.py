"""Métricas de fairness por subgrupo (FR-026).

Calcula, por subgrupo (``SEX`` e faixas de ``AGE``), as taxas de aprovação
(paridade demográfica), a *equal opportunity* (diferença de TPR) e a razão
de impacto dispar (regra dos 80%). Os guardrails são:

- ``disparate_impact_ratio >= 0.8``
- ``equal_opportunity_diff <= 0.10``

Escreve evidências em ``reports/fairness/``.
"""

import json
import logging

import numpy as np
import pandas as pd

from src.core.dataset import FAIRNESS_DIR, TARGET_COLUMN, atomic_write_text
from src.core.params import load_params

logger = logging.getLogger(__name__)

__all__ = [
    "AGE_BIN_LABELS",
    "approval_and_opportunity",
    "evaluate_fairness",
    "build_fairness_report",
    "build_fairness_time_series",
    "find_group_thresholds",
    "mitigate_fairness",
]

AGE_BINS = [0, 30, 45, 60, 100]
AGE_BIN_LABELS = ["<30", "30-44", "45-59", "60+"]

# Colunas protegidas avaliadas (quasi-identificadores).
PROTECTED_COLUMNS = ["SEX", "AGE_BIN"]


def _add_age_bin(frame: pd.DataFrame) -> pd.DataFrame:
    """Deriva a coluna ``AGE_BIN`` a partir de ``AGE``."""
    enriched = frame.copy()
    if "AGE" in enriched.columns:
        enriched["AGE_BIN"] = pd.cut(enriched["AGE"], bins=AGE_BINS, labels=AGE_BIN_LABELS)
    return enriched


def approval_and_opportunity(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """Taxa de aprovação e TPR (equal opportunity) de um subgrupo.

    Aprovação = previsão de adimplente (``prediction == 0``). A TPR é a
    taxa de inadimplentes corretamente identificados (``y=1, ŷ=1``).

    Args:
        labels: Rótulos verdadeiros (1 = inadimplente).
        predictions: Previsões binárias (1 = inadimplente).

    Returns:
        Dicionário com ``approval_rate``, ``tpr`` e ``count``.
    """
    count = len(labels)
    approval_rate = float((predictions == 0).mean()) if count else 0.0
    positives = labels == 1
    tpr = float((predictions[positives] == 1).mean()) if positives.any() else 0.0
    return {"approval_rate": approval_rate, "tpr": tpr, "count": int(count)}


def _passes(metrics: dict, disparate_impact_min: float, equal_opportunity_max: float) -> bool:
    """Aplica os guardrails de fairness a um bloco de métricas."""
    return bool(
        metrics["disparate_impact_ratio"] >= disparate_impact_min
        and metrics["equal_opportunity_diff"] <= equal_opportunity_max
    )


def _group_blocks(
    enriched: pd.DataFrame, group_columns: list[str]
) -> dict[str, dict[str, np.ndarray]]:
    """Mapeia cada coluna protegida às posições (índices) de cada subgrupo."""
    blocks: dict[str, dict[str, np.ndarray]] = {}
    for column in group_columns:
        if column not in enriched.columns:
            continue
        blocks[column] = {
            str(value): np.asarray(index)
            for value, index in enriched.groupby(column, observed=True).groups.items()
        }
    return blocks


def _column_metrics(
    labels: np.ndarray, predictions: np.ndarray, groups: dict[str, np.ndarray]
) -> dict:
    """Agrega paridade, equal opportunity e impacto dispar de uma coluna."""
    per_group = {
        name: approval_and_opportunity(labels[index], predictions[index])
        for name, index in groups.items()
    }
    approvals = [metrics["approval_rate"] for metrics in per_group.values()]
    tprs = [metrics["tpr"] for metrics in per_group.values()]
    maximum_approval = max(approvals) if approvals else 0.0
    return {
        "groups": per_group,
        "demographic_parity_diff": float(max(approvals) - min(approvals)) if approvals else 0.0,
        "equal_opportunity_diff": float(max(tprs) - min(tprs)) if tprs else 0.0,
        "disparate_impact_ratio": (
            float(min(approvals) / maximum_approval) if maximum_approval > 0 else 0.0
        ),
    }


def evaluate_fairness(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    target: str = TARGET_COLUMN,
    group_columns: list[str] | None = None,
    threshold: float = 0.5,
) -> dict:
    """Avalia fairness por subgrupo em um único snapshot.

    Args:
        frame: DataFrame com as colunas de subgrupo e o alvo.
        probabilities: Probabilidades previstas da classe positiva (default).
        target: Nome da coluna alvo.
        group_columns: Colunas de subgrupo (``SEX``, ``AGE_BIN``).
        threshold: Limiar de decisão (probabilidade ≥ threshold → 1).

    Returns:
        Dicionário com métricas por subgrupo e os guardrails.
    """
    params = load_params().fairness
    enriched = _add_age_bin(frame)
    labels = enriched[target].to_numpy()
    predictions = (probabilities >= threshold).astype(int)
    group_columns = group_columns or PROTECTED_COLUMNS
    result: dict[str, dict] = {}
    for column, groups in _group_blocks(enriched, group_columns).items():
        metrics = _column_metrics(labels, predictions, groups)
        metrics["passes"] = _passes(
            metrics, params.disparate_impact_min, params.equal_opportunity_max
        )
        result[column] = metrics
    return result


_MITIGATION_THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)


def find_group_thresholds(
    labels: np.ndarray,
    probabilities: np.ndarray,
    groups: dict[str, np.ndarray],
    target_tpr: float,
) -> dict[str, float]:
    """Limiar de decisão por subgrupo que aproxima o TPR de ``target_tpr``.

    Pós-processamento de *equal opportunity*: o grupo com TPR abaixo do alvo
    recebe um limiar menor (detecta mais inadimplentes) e os demais um
    limiar maior. Empates preferem o limiar mais alto (menos intrusivo).

    Args:
        labels: Rótulos verdadeiros (1 = inadimplente).
        probabilities: Probabilidade prevista da classe positiva.
        groups: Posições de cada subgrupo.
        target_tpr: TPR alvo (o TPR global do modelo sem mitigação).

    Returns:
        Limiar escolhido por subgrupo.
    """
    thresholds: dict[str, float] = {}
    for name, index in groups.items():
        positives = labels[index] == 1
        if not positives.any():
            thresholds[name] = 0.5
            continue
        local = probabilities[index][positives]
        best_threshold, best_error = 0.5, None
        for candidate in _MITIGATION_THRESHOLD_GRID:
            error = abs(float((local >= candidate).mean()) - target_tpr)
            if best_error is None or error < best_error - 1e-12:
                best_threshold, best_error = float(candidate), error
            elif abs(error - best_error) <= 1e-12 and candidate > best_threshold:
                best_threshold = float(candidate)
        thresholds[name] = best_threshold
    return thresholds


def _apply_group_thresholds(
    probabilities: np.ndarray, groups: dict[str, np.ndarray], thresholds: dict[str, float]
) -> np.ndarray:
    """Aplica o limiar de cada subgrupo, produzindo previsões binárias."""
    predictions = np.zeros(len(probabilities), dtype=int)
    for name, index in groups.items():
        predictions[index] = (probabilities[index] >= thresholds[name]).astype(int)
    return predictions


def mitigate_fairness(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    target: str = TARGET_COLUMN,
    group_columns: list[str] | None = None,
) -> dict:
    """Mitiga viés por pós-processamento de limiar (equal opportunity).

    Diferente de apenas **medir** fairness, esta função aplica uma mitigação
    efetiva: cada subgrupo recebe um limiar próprio, calibrado para igualar o
    TPR ao TPR global do modelo. O relatório traz antes/depois e declara o
    *trade-off* (aprovações podem cair no grupo que passa a detectar mais
    inadimplentes) — decisão de negócio, não automática.

    Args:
        frame: DataFrame com as colunas de subgrupo e o alvo.
        probabilities: Probabilidades previstas da classe positiva.
        target: Nome da coluna alvo.
        group_columns: Colunas de subgrupo (``SEX``, ``AGE_BIN``).

    Returns:
        Relatório com limiares, métricas antes/depois e guardrails.
    """
    params = load_params().fairness
    enriched = _add_age_bin(frame)
    labels = enriched[target].to_numpy()
    group_columns = group_columns or PROTECTED_COLUMNS
    baseline_predictions = (probabilities >= 0.5).astype(int)
    positives = labels == 1
    target_tpr = float(baseline_predictions[positives].mean()) if positives.any() else 0.0
    baseline: dict[str, dict] = {}
    mitigated: dict[str, dict] = {}
    for column, groups in _group_blocks(enriched, group_columns).items():
        before = _column_metrics(labels, baseline_predictions, groups)
        thresholds = find_group_thresholds(labels, probabilities, groups, target_tpr)
        after = _column_metrics(
            labels, _apply_group_thresholds(probabilities, groups, thresholds), groups
        )
        before["passes"] = _passes(
            before, params.disparate_impact_min, params.equal_opportunity_max
        )
        after["passes"] = _passes(after, params.disparate_impact_min, params.equal_opportunity_max)
        after["thresholds"] = thresholds
        baseline[column] = before
        mitigated[column] = after
    return {
        "strategy": "equal_opportunity_group_thresholds",
        "target_tpr": target_tpr,
        "baseline": baseline,
        "mitigated": mitigated,
        "passes_before": all(entry["passes"] for entry in baseline.values()),
        "passes_after": all(entry["passes"] for entry in mitigated.values()),
        "tradeoff": (
            "Equalizar o TPR reduz aprovações no grupo que passa a detectar mais "
            "inadimplentes; a escolha entre paridade e equal opportunity é de negócio."
        ),
    }


def build_fairness_report(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    target: str = TARGET_COLUMN,
) -> dict:
    """Gera e persiste o relatório de fairness de um snapshot.

    Args:
        frame: Snapshot avaliado.
        probabilities: Probabilidades previstas.
        target: Nome da coluna alvo.

    Returns:
        Relatório de fairness com os guardrails globais.
    """
    params = load_params().fairness
    by_group = evaluate_fairness(frame, probabilities, target)
    mitigation = mitigate_fairness(frame, probabilities, target)
    passes = all(entry["passes"] for entry in by_group.values()) if by_group else True
    report = {
        "guardrails": {
            "disparate_impact_min": params.disparate_impact_min,
            "equal_opportunity_max": params.equal_opportunity_max,
        },
        "by_group": by_group,
        "passes": passes,
        "mitigation": mitigation,
    }
    FAIRNESS_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_text(FAIRNESS_DIR / "fairness_report.json", json.dumps(report, indent=2))
    atomic_write_text(FAIRNESS_DIR / "fairness_mitigation.json", json.dumps(mitigation, indent=2))
    logger.info("Fairness: passes=%s; mitigação passes=%s.", passes, mitigation["passes_after"])
    return report


def build_fairness_time_series(
    frames: list[pd.DataFrame],
    probabilities: list[np.ndarray],
    target: str = TARGET_COLUMN,
) -> dict:
    """Avalia fairness por subgrupo ao longo dos meses (FR-026).

    Args:
        frames: Snapshots de produção em ordem temporal.
        probabilities: Probabilidades previstas por snapshot (mesma ordem).
        target: Nome da coluna alvo.

    Returns:
        Relatório ``{months: [{month, by_group, passes}], passes}``.
    """
    months = []
    for index, (frame, probs) in enumerate(zip(frames, probabilities, strict=False), start=1):
        by_group = evaluate_fairness(frame, probs, target)
        passes = all(entry["passes"] for entry in by_group.values()) if by_group else True
        months.append({"month": index, "by_group": by_group, "passes": passes})
    report = {"months": months, "passes": all(m["passes"] for m in months) if months else True}
    FAIRNESS_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_text(FAIRNESS_DIR / "fairness_by_month.json", json.dumps(report, indent=2))
    return report
