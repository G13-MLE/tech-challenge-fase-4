"""Gate de drift para o CI (FR-024).

Lê o ``statistical_report.json`` e retorna código de saída não-zero se o
critério de drift do FR-024 for violado: **qualquer feature isolada com
PSI > 0.25** (severidade crítica).

O gate tem dois modos, usados juntos no CI:

- **Modo principal (``evaluate_gate``)** — exatamente a condição do FR-024:
  **falha o build quando qualquer feature tem PSI > 0.25**. É aplicado à
  comparação de *controle* (``--self-check``: referência vs. ela mesma),
  onde nenhum drift é esperado; é a proteção contra falsos positivos e
  regressões que introduzam drift espúrio.
- **Modo detector (``evaluate_detector``, ``--expect-breach``)** — asserção
  explícita e opt-in de que o detector **continua detectando** o drift
  injetado no relatório do pipeline (proteção contra regressão do
  detector). É o *inverso* do gate do FR-024 e por isso NÃO é o modo
  padrão: o pipeline simula drift de propósito, então falhar por PSI
  alto nesse relatório seria esperado.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from src.core.dataset import DRIFT_TEST_RESULT_PATH, STATISTICAL_REPORT_PATH

logger = logging.getLogger(__name__)

__all__ = ["evaluate_gate", "evaluate_detector", "main"]


def _max_psi(report: dict) -> tuple[float, bool]:
    """Extrai ``(max_psi, is_breach)`` de um relatório estatístico."""
    max_psi = float(report.get("max_psi", 0.0))
    is_breach = bool(report.get("is_breach", max_psi > 0.25))
    return max_psi, is_breach


def evaluate_gate(report: dict, psi_max: float = 0.25) -> tuple[bool, str]:
    """Avalia o critério do FR-024: falha quando há qualquer breach de PSI.

    Args:
        report: Relatório estatístico de drift.
        psi_max: Limiar crítico de PSI (0.25).

    Returns:
        Tupla ``(passed, message)`` — ``passed`` é False quando qualquer
        feature tem PSI > ``psi_max``.
    """
    max_psi, is_breach = _max_psi(report)
    passed = not is_breach
    message = (
        f"Sem breach (max_psi={max_psi:.3f} ≤ {psi_max})."
        if passed
        else f"FALHA: breach de PSI (max_psi={max_psi:.3f} > {psi_max})."
    )
    return passed, message


def evaluate_detector(
    report: dict, psi_max: float = 0.25, test_result: dict | None = None
) -> tuple[bool, str]:
    """Assertiva opt-in de detector: exige que o drift seja detectado.

    É o inverso do gate do FR-024 e existe apenas para provar, no CI, que
    o detector não regrediu (o pipeline injeta drift de propósito). Aceita
    tanto o breach de PSI (critério primário) quanto o veredito reprovado
    do Evidently Test Suite (``reports/drift/drift_test_result.json``).

    Args:
        report: Relatório estatístico de drift (com drift injetado).
        psi_max: Limiar crítico de PSI (0.25).
        test_result: Resultado do Evidently Test Suite (opcional) consumido
            como evidência secundária do detector.

    Returns:
        Tupla ``(passed, message)`` — ``passed`` é True quando o breach
        esperado é detectado.
    """
    max_psi, is_breach = _max_psi(report)
    suite_flagged = bool(test_result) and not bool(test_result.get("suite_passed", True))
    passed = is_breach or suite_flagged
    if passed:
        message = (
            f"Detector confirmou drift (max_psi={max_psi:.3f} > {psi_max}, "
            f"suite_flagged={suite_flagged})."
        )
    else:
        message = f"FALHA: drift esperado não detectado (max_psi={max_psi:.3f})."
    return passed, message


def _load_report(args: argparse.Namespace) -> dict:
    """Carrega o relatório a avaliar conforme o modo do gate.

    Em ``--self-check`` compara a referência consigo mesma (sem drift);
    caso contrário lê o relatório estatístico do pipeline.
    """
    if args.self_check:
        from src.core.dataset import REFERENCE_DATA_PATH, load_parquet
        from src.drift.stats import build_statistical_report

        reference = load_parquet(REFERENCE_DATA_PATH)
        return build_statistical_report(reference, reference, write=False)
    if not args.report.exists():
        logger.error("Relatório não encontrado: %s.", args.report)
        sys.exit(1)
    return json.loads(args.report.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> None:
    """Ponto de entrada do gate de drift.

    Args:
        argv: Argumentos de linha de comando (se None, usa ``sys.argv``).

    Raises:
        SystemExit: Código 1 quando o critério do gate é violado.
    """
    parser = argparse.ArgumentParser(description="Gate de drift (PSI, FR-024).")
    parser.add_argument("--report", type=Path, default=STATISTICAL_REPORT_PATH)
    parser.add_argument("--psi-max", type=float, default=0.25)
    parser.add_argument(
        "--expect-breach",
        action="store_true",
        help="Modo detector (opt-in): exige que o drift seja detectado (inverso do FR-024).",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Compara a referência consigo mesma e falha se houver PSI > limite (FR-024).",
    )
    args = parser.parse_args(argv)
    report = _load_report(args)
    if args.expect_breach:
        test_result = None
        if DRIFT_TEST_RESULT_PATH.exists():
            test_result = json.loads(DRIFT_TEST_RESULT_PATH.read_text(encoding="utf-8"))
        passed, message = evaluate_detector(report, args.psi_max, test_result)
    else:
        passed, message = evaluate_gate(report, args.psi_max)
    logger.info("Gate de drift (%s): %s", "detector" if args.expect_breach else "gate", message)
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
