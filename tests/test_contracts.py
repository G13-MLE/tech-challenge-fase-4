"""Testes do contrato de dados (Etapa 1).

Garante que um lote válido passa e que um lote corrompido é bloqueado.
"""

import pandera.errors as pa_errors

from src.contracts.schema import CREDIT_CONTRACT


def test_valid_batch_passes(credit_frame):
    """Um lote válido não deve levantar erro de contrato."""
    validated = CREDIT_CONTRACT.validate(credit_frame, lazy=True)
    assert len(validated) == len(credit_frame)


def test_bad_batch_is_blocked(bad_frame):
    """Um lote com violações deve falhar na validação."""
    try:
        CREDIT_CONTRACT.validate(bad_frame, lazy=True)
    except pa_errors.SchemaErrors as err:
        assert len(err.failure_cases) > 0
    else:
        raise AssertionError("O contrato deveria ter bloqueado o lote corrompido.")
