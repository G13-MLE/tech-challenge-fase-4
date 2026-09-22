"""Contrato de dados (data contract) do dataset de crédito.

Define as regras rígidas de qualidade. É a fonte única de verdade:
os limites aqui são usados tanto no treino quanto na validação de
lotes de produção. Regras intencionalmente incluem limites de domínio
(ex.: BILL_AMT pode ser negativo — saldo credor).
"""

import pandera.pandas as pa

__all__ = ["CREDIT_CONTRACT", "AGE_MIN", "AGE_MAX", "EDUCATION_ALLOWED", "MARRIAGE_ALLOWED"]

AGE_MIN = 18
AGE_MAX = 100
EDUCATION_ALLOWED = {1, 2, 3, 4}
MARRIAGE_ALLOWED = {1, 2, 3}
PAY_ALLOWED = {-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8}


CREDIT_CONTRACT = pa.DataFrameSchema(
    {
        "LIMIT_BAL": pa.Column(float, pa.Check.gt(0), nullable=False),
        "SEX": pa.Column(int, pa.Check.isin([1, 2]), nullable=False),
        "EDUCATION": pa.Column(int, pa.Check.isin(list(EDUCATION_ALLOWED)), nullable=False),
        "MARRIAGE": pa.Column(int, pa.Check.isin(list(MARRIAGE_ALLOWED)), nullable=False),
        "AGE": pa.Column(int, pa.Check.in_range(AGE_MIN, AGE_MAX), nullable=False),
        "PAY_0": pa.Column(int, pa.Check.isin(list(PAY_ALLOWED)), nullable=False),
        "BILL_AMT1": pa.Column(float, nullable=False),
        "PAY_AMT1": pa.Column(float, pa.Check.ge(0), nullable=False),
        "default_next_month": pa.Column(int, pa.Check.isin([0, 1]), nullable=False),
    },
    strict=False,
    coerce=False,
)
