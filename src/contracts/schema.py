"""Contrato de dados (data contract) do dataset de crédito.

Define as regras rígidas de qualidade. É a fonte única de verdade:
os limites aqui são usados tanto no treino quanto na validação de
lotes de produção. Regras intencionalmente incluem limites de domínio
(ex.: BILL_AMT pode ser negativo — saldo credor).

Duas representações compartilham as mesmas constantes de domínio:

- ``CREDIT_CONTRACT``: ``pa.DataFrameSchema`` usado em runtime pelo
  estágio ``validacao`` (fonte única de verdade).
- ``CreditSchema``: ``pa.DataFrameModel`` class-based + ``@pa.check_types``
  que fornece tipagem estática das fronteiras (FR-004).

Escopo das colunas ``PAY_*`` (FR-001): a ingestão projeta o batch canônico
para 8 features + alvo; das colunas de histórico de pagamento do UCI 350
(``PAY_0..PAY_6``), apenas ``PAY_0`` (status mais recente) é retido e,
portanto, validado. ``PAY_2..PAY_6`` são descartados em ``src/data/ingest.py``
e estão **fora do escopo** do contrato — o domínio ``PAY_ALLOWED`` aplica-se
a essas colunas quando retidas.
"""

import pandera.pandas as pa
from pandera.typing import DataFrame, Series

__all__ = [
    "CREDIT_CONTRACT",
    "CreditSchema",
    "validate_typed",
    "AGE_MIN",
    "AGE_MAX",
    "SEX_ALLOWED",
    "EDUCATION_ALLOWED",
    "MARRIAGE_ALLOWED",
    "PAY_ALLOWED",
]

AGE_MIN = 18
AGE_MAX = 100
SEX_ALLOWED = {1, 2}
EDUCATION_ALLOWED = {1, 2, 3, 4}
MARRIAGE_ALLOWED = {1, 2, 3}
PAY_ALLOWED = {-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8}


CREDIT_CONTRACT = pa.DataFrameSchema(
    {
        "LIMIT_BAL": pa.Column(float, pa.Check.gt(0), nullable=False),
        "SEX": pa.Column(int, pa.Check.isin(sorted(SEX_ALLOWED)), nullable=False),
        "EDUCATION": pa.Column(int, pa.Check.isin(sorted(EDUCATION_ALLOWED)), nullable=False),
        "MARRIAGE": pa.Column(int, pa.Check.isin(sorted(MARRIAGE_ALLOWED)), nullable=False),
        "AGE": pa.Column(int, pa.Check.in_range(AGE_MIN, AGE_MAX), nullable=False),
        "PAY_0": pa.Column(int, pa.Check.isin(sorted(PAY_ALLOWED)), nullable=False),
        "BILL_AMT1": pa.Column(float, nullable=False),
        "PAY_AMT1": pa.Column(float, pa.Check.ge(0), nullable=False),
        "default_next_month": pa.Column(int, pa.Check.isin([0, 1]), nullable=False),
    },
    strict=False,
    coerce=False,
)


class CreditSchema(pa.DataFrameModel):
    """Esquema class-based equivalente a ``CREDIT_CONTRACT`` (FR-004).

    Reutiliza as constantes de domínio compartilhadas e habilita a
    validação tipada via ``@pa.check_types``. Não coage tipos nem é
    estrito com colunas desconhecidas, espelhando o contrato em runtime.
    """

    LIMIT_BAL: Series[float] = pa.Field(gt=0, nullable=False)
    SEX: Series[int] = pa.Field(isin=sorted(SEX_ALLOWED), nullable=False)
    EDUCATION: Series[int] = pa.Field(isin=sorted(EDUCATION_ALLOWED), nullable=False)
    MARRIAGE: Series[int] = pa.Field(isin=sorted(MARRIAGE_ALLOWED), nullable=False)
    AGE: Series[int] = pa.Field(
        in_range={"min_value": AGE_MIN, "max_value": AGE_MAX}, nullable=False
    )
    PAY_0: Series[int] = pa.Field(isin=sorted(PAY_ALLOWED), nullable=False)
    BILL_AMT1: Series[float] = pa.Field(nullable=False)
    PAY_AMT1: Series[float] = pa.Field(ge=0, nullable=False)
    default_next_month: Series[int] = pa.Field(isin=[0, 1], nullable=False)

    class Config:
        """Configuração do ``DataFrameModel`` (espelha o contrato runtime)."""

        strict = False
        coerce = False


@pa.check_types
def validate_typed(frame: DataFrame[CreditSchema]) -> DataFrame[CreditSchema]:
    """Valida tipos/regras de um DataFrame conforme ``CreditSchema``.

    Args:
        frame: DataFrame anotado com ``CreditSchema``.

    Returns:
        O mesmo DataFrame validado (pandera ``check_types``).

    Raises:
        pandera.errors.SchemaErrors: Se alguma regra do esquema for violada.
    """
    return frame
