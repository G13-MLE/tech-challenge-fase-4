"""Gate de drift para o CI (FR-024).

Lê o ``statistical_report.json`` e retorna código de saída não-zero se o
critério de drift for violado. O critério primário é **qualquer feature
com PSI > 0.25** (breach). No CI, o gate roda com ``--expect-breach`` para
garantir que o detector continua identificando o drift injetado de forma
controlada (regressão do detector) — falha o build se o detector parar de
detectar.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from src.core.dataset import STATISTICAL_REPORT_PATH

logger = logging.getLogger(__name__)

__all__ = ["evaluate_gate", "main"]


def evaluate_gate(report: dict, psi_max: float, expect_breach: bool) -> tuple[bool, str]:
    """Avalia o critério do gate de drift.

    Args:
        report: Relatório estatístico de drift.
        psi_max: Limiar crítico de PSI (0.25).
        expect_breach: Se True, exige que haja drift (breach) — usado no CI
            para validar o detector; se False, exige ausência de drift.

    Returns:
        Tupla ``(passed, message)``.
    """
    max_psi = float(report.get("max_psi", 0.0))
    is_breach = bool(report.get("is_breach", max_psi > psi_max))
    if expect_breach:
        passed = is_breach
        message = (
            f"Detector confirmou drift (max_psi={max_psi:.3f} > {psi_max})."
            if passed
            else f"FALHA: drift esperado não detectado (max_psi={max_psi:.3f})."
        )
    else:
        passed = not is_breach
        message = (
            f"Sem breach (max_psi={max_psi:.3f} ≤ {psi_max})."
            if passed
            else f"FALHA: breach de PSI (max_psi={max_psi:.3f} > {psi_max})."
        )
    return passed, message


def main(argv: list[str] | None = None) -> None:
    """Ponto de entrada do gate de drift.

    Args:
        argv: Argumentos de linha de comando (se None, usa ``sys.argv``).

    Raises:
        SystemExit: Código 1 quando o critério do gate é violado.
    """
    parser = argparse.ArgumentParser(description="Gate de drift (PSI).")
    parser.add_argument("--report", type=Path, default=STATISTICAL_REPORT_PATH)
    parser.add_argument("--psi-max", type=float, default=0.25)
    parser.add_argument("--expect-breach", action="store_true")
    args = parser.parse_args(argv)
    if not args.report.exists():
        logger.error("Relatório não encontrado: %s.", args.report)
        sys.exit(1)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    passed, message = evaluate_gate(report, args.psi_max, args.expect_breach)
    logger.info("Gate de drift: %s", message)
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
