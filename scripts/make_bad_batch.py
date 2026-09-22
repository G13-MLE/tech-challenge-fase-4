"""Geração de um lote corrompido para demonstrar o contrato bloqueando.

Etapa 1. Injeta violações (nulos, idade fora do range, categorias
inválidas, limite não-positivo) para provar que a validação bloqueia
a ingestão. Requer o dataset de referência já gerado.
"""

import logging
from pathlib import Path

from src.core.dataset import REFERENCE_DATA_PATH, load_parquet

logger = logging.getLogger(__name__)

__all__ = ["make_bad_batch", "main"]

BAD_BATCH_PATH = Path("data/corrupted/bad_batch.parquet")


def make_bad_batch(destination: Path = BAD_BATCH_PATH) -> Path:
    """Cria um lote com violações intencionais do contrato.

    Args:
        destination: Caminho de destino do lote corrompido.

    Returns:
        Caminho do lote corrompido.
    """
    import pandas as pd

    frame = load_parquet(REFERENCE_DATA_PATH).head(200).copy()
    frame.loc[0:4, "AGE"] = 15
    frame.loc[5:9, "LIMIT_BAL"] = -100.0
    frame.loc[10:14, "EDUCATION"] = 99
    frame.loc[15, "PAY_AMT1"] = -50.0
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)
    logger.info("Lote corrompido gerado: %s (%d linhas).", destination, len(frame))
    return destination


def main() -> None:
    """Ponto de entrada da geração do lote corrompido."""
    make_bad_batch()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
