"""Deduplicação antes do split (FR-003, R11).

Remove duplicatas **exatas** via hash SHA-256 da linha e **near-duplicates**
via MinHash/LSH (``datasketch``), registrando as contagens antes/depois.
Deve ser executado **antes** da separação treino/validação para evitar
vazamento entre conjuntos.
"""

import hashlib
import logging
from collections.abc import Iterator

import pandas as pd
from datasketch import MinHash, MinHashLSH

from src.core.params import load_params

logger = logging.getLogger(__name__)

__all__ = ["deduplicate", "row_hash"]

NUM_PERM = 128
_MINHASH_SEED = 1


def row_hash(frame: pd.DataFrame) -> pd.Series:
    """Calcula o hash SHA-256 determinístico de cada linha.

    Args:
        frame: DataFrame de entrada.

    Returns:
        Série de digests hexadecimais, uma por linha.
    """
    payload = frame.astype(str).agg("|".join, axis=1)
    return payload.map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())


def _row_tokens(row: pd.Series) -> Iterator[str]:
    """Gera tokens ``coluna=valor`` para o shingle de uma linha."""
    for column, value in row.items():
        yield f"{column}={value}"


def _minhash(row: pd.Series) -> MinHash:
    """Constrói o MinHash determinístico de uma linha."""
    minhash = MinHash(num_perm=NUM_PERM, seed=_MINHASH_SEED)
    for token in _row_tokens(row):
        minhash.update(token.encode("utf-8"))
    return minhash


def deduplicate(
    frame: pd.DataFrame,
    *,
    near_dup_threshold: float | None = None,
    use_near_dup: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Remove duplicatas exatas e near-duplicates do lote.

    Args:
        frame: DataFrame de entrada.
        near_dup_threshold: Limiar Jaccard (MinHash/LSH); se None, usa
            ``quality.near_dup_threshold`` de ``configs/params.yaml``.
        use_near_dup: Se False, remove apenas duplicatas exatas (SHA-256).

    Returns:
        Tupla ``(frame_deduplicado, relatório)`` com contagens antes/depois.
    """
    threshold = (
        load_params().quality.near_dup_threshold
        if near_dup_threshold is None
        else near_dup_threshold
    )

    n_before = int(len(frame))
    hashes = row_hash(frame)
    exact = frame.loc[~hashes.duplicated(keep="first")].reset_index(drop=True)
    exact_removed = n_before - len(exact)

    near_removed = 0
    if use_near_dup and len(exact) > 1:
        lsh = MinHashLSH(threshold=threshold, num_perm=NUM_PERM)
        kept_positions: list[int] = []
        for position, (_, row) in enumerate(exact.iterrows()):
            minhash = _minhash(row)
            if lsh.query(minhash):
                continue
            lsh.insert(str(position), minhash)
            kept_positions.append(position)
        deduplicated = exact.iloc[kept_positions].reset_index(drop=True)
        near_removed = len(exact) - len(deduplicated)
    else:
        deduplicated = exact

    report = {
        "method": "sha256+minhash_lsh" if use_near_dup else "sha256",
        "near_dup_threshold": float(threshold),
        "num_perm": NUM_PERM,
        "n_before": n_before,
        "n_after": int(len(deduplicated)),
        "exact_duplicates_removed": int(exact_removed),
        "near_duplicates_removed": int(near_removed),
        "total_removed": int(exact_removed + near_removed),
    }
    logger.info(
        "Dedup: %d -> %d (%d exatas, %d near).",
        n_before,
        report["n_after"],
        exact_removed,
        near_removed,
    )
    return deduplicated, report
