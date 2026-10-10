"""Testes do contrato de dados (Etapa 1).

Garante que um lote válido passa, que um lote corrompido é bloqueado com
exit code 1 e que o relatório carrega as dimensões de qualidade (FR-001/FR-003).
"""

import pandera.errors as pa_errors
import pytest

import src.contracts.validate as validate_module
from src.contracts.schema import CREDIT_CONTRACT, CreditSchema, validate_typed


def test_valid_batch_passes(credit_frame):
    """Um lote válido não deve levantar erro de contrato."""
    validated = CREDIT_CONTRACT.validate(credit_frame, lazy=True)
    assert len(validated) == len(credit_frame)


def test_class_schema_and_typed_validation(credit_frame):
    """O esquema class-based e ``@pa.check_types`` aceitam o lote válido."""
    assert len(CreditSchema.validate(credit_frame, lazy=True)) == len(credit_frame)
    assert len(validate_typed(credit_frame)) == len(credit_frame)


def test_bad_batch_is_blocked(bad_frame):
    """Um lote com violações deve falhar na validação."""
    with pytest.raises(pa_errors.SchemaErrors) as err:
        CREDIT_CONTRACT.validate(bad_frame, lazy=True)
    assert len(err.value.failure_cases) > 0


def test_validate_batch_reports_quality(tmp_path, monkeypatch, credit_frame):
    """O relatório de um lote válido inclui as dimensões de qualidade."""
    batch_path = tmp_path / "valid.parquet"
    credit_frame.to_parquet(batch_path, index=False)
    result_path = tmp_path / "validation" / "result.json"
    monkeypatch.setattr(validate_module, "VALIDATION_RESULT_PATH", result_path)

    report = validate_module.validate_batch(str(batch_path))

    assert report["status"] == "passed"
    quality = report["quality"]
    assert quality["completeness"] == 1.0
    assert quality["validity"] == 1.0
    assert quality["decisions"]
    assert all(decision == "keep" for decision in quality["decisions"].values())
    assert result_path.exists()


def test_corrupted_batch_exits_with_code_1(tmp_path, monkeypatch, bad_frame):
    """Um lote corrompido bloqueia a ingestão com exit code 1."""
    batch_path = tmp_path / "bad.parquet"
    bad_frame.to_parquet(batch_path, index=False)
    result_path = tmp_path / "validation" / "result.json"
    monkeypatch.setattr(validate_module, "REFERENCE_DATA_PATH", batch_path)
    monkeypatch.setattr(validate_module, "VALIDATION_RESULT_PATH", result_path)

    report = validate_module.validate_batch()
    assert report["status"] == "failed"
    assert report["n_violations"] > 0
    assert report["quality"]["n_duplicate_rows"] == 0

    with pytest.raises(SystemExit) as excinfo:
        validate_module.main([])
    assert excinfo.value.code == 1


def test_main_accepts_path_argument(tmp_path, monkeypatch, bad_frame):
    """A CLI aceita o caminho do lote e bloqueia o lote corrompido."""
    batch_path = tmp_path / "bad.parquet"
    bad_frame.to_parquet(batch_path, index=False)
    monkeypatch.setattr(validate_module, "VALIDATION_RESULT_PATH", tmp_path / "result.json")

    with pytest.raises(SystemExit) as excinfo:
        validate_module.main([str(batch_path)])
    assert excinfo.value.code == 1
