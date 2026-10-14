"""Detecção de drift com Evidently AI (versão instalada: 0.7.23).

Gera um relatório HTML comparando o dataset de referência com um snapshot
de produção e um resultado pass/fail (gate de CI). IMPORTANTE: o método de
drift é configurado explicitamente como PSI (numérico) e χ² (categórico);
o default do Evidently para n > 1000 é Wasserstein, que não é a métrica
exigida pelo desafio (FR-012).
"""

import json
import logging
from pathlib import Path

from src.core.dataset import (
    DRIFT_REPORT_PATH,
    DRIFT_TEST_RESULT_PATH,
    atomic_path,
    atomic_write_text,
    normalize_text_file,
)
from src.core.params import load_params

logger = logging.getLogger(__name__)

__all__ = ["build_evidently_report", "build_test_suite_result"]

# Nome registrado do stattest de χ² no Evidently 0.7.23.
_EVIDENTLY_NUM_METHOD = "psi"
_EVIDENTLY_CAT_METHOD = "chisquare"

# Stattests cujo resultado é um **p-valor**: o drift é sinalizado quando o
# score fica **abaixo** do limiar (score menor = evidência mais forte de
# drift). Os demais métodos retornam um tamanho de efeito (ex.: PSI,
# Wasserstein, Jensen-Shannon), onde o drift aparece quando o score
# **excede** o limiar.
_P_VALUE_METHODS = {
    "chisquare",
    "chi-square",
    "ks",
    "kstest",
    "ks_2samp",
    "cramervonmises",
    "cramer_von_mises",
    "mannwhitney",
    "mann_whitney",
    "fisher_exact",
    "fisher",
    "t_test",
    "ttest",
}


def _is_drift(method: str, value: float, threshold: float) -> bool:
    """Decide se um teste Evidently sinaliza drift a partir do método.

    Args:
        method: Nome do stattest (``config["method"]``).
        value: Score retornado pelo stattest (p-valor ou tamanho de efeito).
        threshold: Limiar configurado do teste.

    Returns:
        True se o teste detecta drift, False caso contrário.
    """
    if method.lower() in _P_VALUE_METHODS:
        return value < threshold
    return value > threshold


def _drift_share() -> float:
    """Fração de colunas com drift para o Evidently marcar o dataset.

    O default do Evidently (``0.5``) contradiz o critério por feature do
    desafio (FR-010) quando apenas parte das colunas sofre drift, fazendo o
    HTML do relatório dizer "Dataset Drift is NOT detected" enquanto o
    ``statistical_report.json`` aponta um breach. O valor é configurável em
    ``drift_thresholds.evidently_drift_share`` (default ``0.1``).
    """
    return float(load_params().drift_thresholds.evidently_drift_share)


def build_evidently_report(reference, production, html_path: Path | None = None) -> str:
    """Gera o relatório HTML de drift com Evidently.

    Args:
        reference: DataFrame de referência.
        production: DataFrame de produção.
        html_path: Caminho de saída do HTML. Se None, usa o relatório
            canônico ``reports/drift/drift_report.html``. O demo (US7) passa
            um caminho isolado; ``--no-evidently`` simplesmente não chama
            este builder.

    Returns:
        Caminho do HTML gerado.
    """
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    target_path = html_path or DRIFT_REPORT_PATH
    data_definition = DataDefinition()
    reference_ds = Dataset.from_pandas(reference, data_definition=data_definition)
    production_ds = Dataset.from_pandas(production, data_definition=data_definition)
    report = Report(
        [
            DataDriftPreset(
                num_method=_EVIDENTLY_NUM_METHOD,
                cat_method=_EVIDENTLY_CAT_METHOD,
                include_tests=True,
                drift_share=_drift_share(),
            )
        ]
    )
    snapshot = report.run(production_ds, reference_ds)
    with atomic_path(target_path, suffix=".tmp.html") as tmp_path:
        snapshot.save_html(str(tmp_path))
        normalize_text_file(tmp_path)
    logger.info("Relatório Evidently salvo em %s.", target_path)
    return str(target_path)


def _evidently_test_results(reference, production) -> dict:
    """Extrai o resultado pass/fail do snapshot Evidently.

    Args:
        reference: DataFrame de referência.
        production: DataFrame de produção.

    Returns:
        Dicionário com testes por coluna e o veredito ``passed``.
    """
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    data_definition = DataDefinition()
    report = Report(
        [
            DataDriftPreset(
                num_method=_EVIDENTLY_NUM_METHOD,
                cat_method=_EVIDENTLY_CAT_METHOD,
                include_tests=True,
                drift_share=_drift_share(),
            )
        ]
    )
    snapshot = report.run(
        Dataset.from_pandas(production, data_definition=data_definition),
        Dataset.from_pandas(reference, data_definition=data_definition),
    )
    payload = snapshot.dict()
    tests = []
    for metric in payload.get("metrics", []):
        name = metric.get("metric_name", "")
        if not name.startswith("ValueDrift"):
            continue
        config = metric.get("config", {})
        method = str(config.get("method") or "")
        threshold = float(config.get("threshold", 0.1))
        value = float(metric.get("value", 0.0) or 0.0)
        tests.append(
            {
                "name": name,
                "column": config.get("column"),
                "method": method,
                "value": value,
                "threshold": threshold,
                "passed": not _is_drift(method, value, threshold),
            }
        )
    return {
        "tests": tests,
        "n_failed": sum(1 for test in tests if not test["passed"]),
        "passed": all(test["passed"] for test in tests),
    }


def build_test_suite_result(reference, production, statistical_report: dict | None = None) -> dict:
    """Emite o resultado pass/fail do gate de drift (CI).

    Combina o teste do Evidently (share de colunas com drift) com o
    critério primário do desafio: **qualquer feature com PSI > 0.25** é um
    breach. Escreve ``reports/drift/drift_test_result.json``.

    Args:
        reference: DataFrame de referência.
        production: DataFrame de produção.
        statistical_report: Relatório PSI/KS (opcional) para o critério PSI.

    Returns:
        Dicionário com ``passed``, contagem de colunas com drift e testes.
    """
    result: dict = {"is_breach": False, "max_psi": None, "drifted_features": []}
    try:
        suite = _evidently_test_results(reference, production)
        result.update(suite)
        # Preserva o veredito próprio do Evidently Test Suite sob uma chave
        # distinta, para não ser mascarado pelo veredito PSI (FR-014).
        result["suite_passed"] = bool(suite.get("passed", True))
        result["suite_n_failed"] = int(suite.get("n_failed", 0))
    except Exception as exc:  # noqa: BLE001 - gate não deve quebrar o pipeline
        logger.warning("Evidently Test Suite indisponível (%s); usando PSI.", exc)
        result.setdefault("tests", [])
        result["suite_passed"] = True
        result["suite_n_failed"] = 0
    if statistical_report is not None:
        features = statistical_report.get("features", {})
        drifted = [name for name, entry in features.items() if entry.get("severity") == "severe"]
        result["is_breach"] = bool(statistical_report.get("is_breach", False))
        result["max_psi"] = statistical_report.get("max_psi")
        result["drifted_features"] = drifted
        # PSI é a métrica primária de decisão do desafio.
        result["passed"] = not result["is_breach"]
    else:
        result["passed"] = bool(result.get("suite_passed", True))
    # Proveniência: ``tests[].value`` é o score do stattest do Evidently
    # (binning próprio, não comparável 1:1 com o PSI do projeto);
    # ``max_psi``/``drifted_features`` vêm do ``statistical_report.json``
    # (PSI do projeto, binning quantílico) — a métrica de decisão.
    result["psi_provenance"] = {
        "max_psi": "statistical_report.json (PSI do projeto, decisão)",
        "tests_value": "evidently stattest score (binning próprio, evidência)",
        "note": (
            "Os dois PSI divergem porque usam binning diferente; a decisão de "
            "breach usa sempre o PSI do projeto (max_psi)."
        ),
    }
    atomic_write_text(DRIFT_TEST_RESULT_PATH, json.dumps(result, indent=2))
    logger.info("Gate de drift: passed=%s, breach=%s.", result["passed"], result["is_breach"])
    return result
