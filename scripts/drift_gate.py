"""Gate de drift para o CI (FR-024).

Lê o ``statistical_report.json`` e retorna código de saída não-zero se o
critério de drift for violado. O critério primário é **qualquer feature
com PSI > 0.25** (breach).

O gate tem dois modos, usados juntos no CI:

- **``--self-check``**: compara a referência **consigo mesma** e falha o
  build se **qualquer feature** apresentar PSI > 0.25 — exatamente a
  condição do FR-024 (protege contra falsos positivos e regressões que
  introduzam drift espúrio). Aqui não há drift esperado.
- **``--expect-breach``**: roda sobre o relatório do pipeline (com drift
  injetado) e falha se o detector **parar de detectar** o drift simulado
  (proteção contra regressão do detector).
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
    parser = argparse.ArgumentParser(description="Gate de drift (PSI).")
    parser.add_argument("--report", type=Path, default=STATISTICAL_REPORT_PATH)
    parser.add_argument("--psi-max", type=float, default=0.25)
    parser.add_argument("--expect-breach", action="store_true")
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Compara a referência consigo mesma e falha se houver PSI > limite.",
    )
    args = parser.parse_args(argv)
    report = _load_report(args)
    passed, message = evaluate_gate(report, args.psi_max, args.expect_breach)
    logger.info("Gate de drift (%s): %s", "self-check" if args.self_check else "report", message)
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
