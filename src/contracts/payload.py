"""Contrato Pydantic de payload de inferência (FR-005, R10).

Valida um único registro de crédito antes de chegar ao modelo, espelhando
as regras de domínio do contrato de dados (constantes compartilhadas em
``src/contracts/schema.py``). Campos desconhecidos são rejeitados
(``extra="forbid"``) e o payload é imutável (``frozen=True``).
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.contracts.schema import (
    AGE_MAX,
    AGE_MIN,
    EDUCATION_ALLOWED,
    MARRIAGE_ALLOWED,
    PAY_ALLOWED,
    SEX_ALLOWED,
)

__all__ = ["CreditScoreRequest", "CreditScoreResponse"]


class CreditScoreRequest(BaseModel):
    """Payload de entrada para pontuar um registro de crédito.

    Attributes:
        LIMIT_BAL: Limite de crédito concedido (> 0).
        SEX: Código de sexo (∈ {1, 2}).
        EDUCATION: Código de escolaridade (∈ {1, 2, 3, 4}).
        MARRIAGE: Código de estado civil (∈ {1, 2, 3}).
        AGE: Idade em anos ([18, 100]).
        PAY_0: Status de pagamento mais recente (∈ {-2..8}).
        BILL_AMT1: Valor da fatura mais recente (pode ser negativo — saldo credor).
        PAY_AMT1: Valor do pagamento mais recente (≥ 0).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    LIMIT_BAL: float = Field(gt=0)
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int = Field(ge=AGE_MIN, le=AGE_MAX)
    PAY_0: int = Field(ge=min(PAY_ALLOWED), le=max(PAY_ALLOWED))
    BILL_AMT1: float
    PAY_AMT1: float = Field(ge=0)

    @field_validator("SEX")
    @classmethod
    def _validate_sex(cls, value: int) -> int:
        if value not in SEX_ALLOWED:
            raise ValueError(f"SEX deve estar em {sorted(SEX_ALLOWED)}.")
        return value

    @field_validator("EDUCATION")
    @classmethod
    def _validate_education(cls, value: int) -> int:
        if value not in EDUCATION_ALLOWED:
            raise ValueError(f"EDUCATION deve estar em {sorted(EDUCATION_ALLOWED)}.")
        return value

    @field_validator("MARRIAGE")
    @classmethod
    def _validate_marriage(cls, value: int) -> int:
        if value not in MARRIAGE_ALLOWED:
            raise ValueError(f"MARRIAGE deve estar em {sorted(MARRIAGE_ALLOWED)}.")
        return value


class CreditScoreResponse(BaseModel):
    """Payload de saída da pontuação de crédito.

    Attributes:
        default_probability: Probabilidade de inadimplência no próximo mês.
        default_prediction: Decisão binária (0 = adimplente, 1 = inadimplente).
        model_name: Nome do modelo que produziu a decisão.
        model_version: Versão/alias registrado do modelo.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    default_probability: float = Field(ge=0.0, le=1.0)
    default_prediction: int = Field(ge=0, le=1)
    model_name: str
    model_version: str
