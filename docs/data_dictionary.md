# Dicionário de Dados — Credit Scoring (UCI 350)

Classificação de PII por campo. O contrato rígido (`src/contracts/schema.py`) é a fonte única de
verdade para validação; aqui documentamos semântica, tipo e sensibilidade.

## Colunas usadas pelo modelo/contrato

| Campo | Tipo | Domínio/Regra | Semântica | PII | Uso |
| ----- | ---- | ------------- | --------- | --- | --- |
| `ID` | int | único | Identificador da linha | **Direto** | Removido na ingestão |
| `LIMIT_BAL` | float | `[10 000, 1 000 000]`, não nulo | Limite de crédito concedido (NT$) — proxy de limite de renda | Financeiro | Feature |
| `SEX` | int | `{1, 2}` | Sexo (1=masc, 2=fem) | Quasi | Feature + fairness |
| `EDUCATION` | int | `{1,2,3,4}` | Escolaridade | Quasi | Feature + fairness |
| `MARRIAGE` | int | `{1,2,3}` | Estado civil | Quasi | Feature |
| `AGE` | int | `[18, 100]` | Idade (anos) | Quasi | Feature + fairness |
| `PAY_0` | int | `{-2..8}` | Status de pagamento mais recente | Financeiro | Feature |
| `BILL_AMT1` | float | não nulo (pode ser negativo) | Valor da fatura mais recente | Financeiro | Feature |
| `PAY_AMT1` | float | `>= 0` | Valor do pagamento mais recente | Financeiro | Feature |
| `default_next_month` | int | `{0, 1}` | Inadimplência no mês seguinte | Rótulo | Alvo |

## Colunas do dataset original (UCI 350)

O arquivo bruto contém o histórico completo; somente o subconjunto acima é usado pelo contrato/training
(extensível por configuração):

| Campo | Semântica | PII |
| ----- | --------- | --- |
| `PAY_0..PAY_6` | Histórico de status de pagamento (set/abr a ago) | Financeiro |
| `BILL_AMT1..BILL_AMT6` | Histórico de valores de fatura | Financeiro |
| `PAY_AMT1..PAY_AMT6` | Histórico de valores pagos | Financeiro |

## Notas de qualidade

- **Política flag-and-keep**: ausências e outliers são **sinalizados** e mantidos (`reports/validation/result.json`
  → `quality.decisions`), sem mutação silenciosa. O bloco `quality.decisions` é um **registro auditável** da
  decisão pretendida por coluna (`keep`/`remove`/`transform`); ele **não** aplica remoções/transformações
  automaticamente — qualquer mutação efetiva exige implementação explícita pelo operador.
- **BILL_AMT** pode ser negativo (saldo credor) — **não** há regra de positividade.
- **Deduplicação** (exata SHA-256 + near-duplicates MinHash/LSH) ocorre **antes** do split para evitar
  vazamento treino/teste (`src/data/dedup.py`).
