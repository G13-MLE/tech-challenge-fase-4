# Relatório estatístico de drift — referência × produção

Mudança de distribuição entre o dataset de **referência** (treino) e os **snapshots de produção** (Data + Concept Drift), medida por **PSI** (métrica de decisão) ao lado de **KS/χ²** (evidência).

- **Snapshots analisados:** 6
- **Mês destacado:** 6
- **Breach de PSI (limiar 0.25):** sim
- **PSI máximo:** 2.221
- **Features com drift severo (mês 6):** 4

## PSI por feature e mês (severidade: stable/moderate/severe)

| Feature | M1 | M2 | M3 | M4 | M5 | M6 |
| --- | --- | --- | --- | --- | --- | --- |
| LIMIT_BAL | 0.048 stable | 0.228 moderate | 0.525 severe | 1.169 severe | 1.537 severe | 2.221 severe |
| SEX | 0.000 stable | 0.000 stable | 0.000 stable | 0.000 stable | 0.000 stable | 0.000 stable |
| EDUCATION | 0.000 stable | 0.000 stable | 0.000 stable | 0.000 stable | 0.000 stable | 1.162 severe |
| MARRIAGE | 0.000 stable | 0.000 stable | 0.000 stable | 0.000 stable | 0.000 stable | 0.000 stable |
| AGE | 0.014 stable | 0.062 stable | 0.161 moderate | 0.339 severe | 0.610 severe | 1.289 severe |
| PAY_0 | 0.000 stable | 0.000 stable | 0.000 stable | 0.000 stable | 0.122 moderate | 0.987 severe |
| BILL_AMT1 | 0.000 stable | 0.000 stable | 0.000 stable | 0.000 stable | 0.000 stable | 0.000 stable |
| PAY_AMT1 | 0.000 stable | 0.000 stable | 0.000 stable | 0.000 stable | 0.000 stable | 0.000 stable |

## Testes estatísticos do mês 6 (KS numérico / χ² categórico)

| Feature | Tipo | PSI | Severidade | Estatística | p-valor |
| --- | --- | --- | --- | --- | --- |
| LIMIT_BAL | numeric | 2.221 | severe | KS=0.332 | 0.00e+00 |
| SEX | categorical | 0.000 | stable | χ²=0.0 | 1.00e+00 |
| EDUCATION | categorical | 1.162 | severe | χ²=3137.0 | 0.00e+00 |
| MARRIAGE | categorical | 0.000 | stable | χ²=0.0 | 1.00e+00 |
| AGE | numeric | 1.289 | severe | KS=0.221 | 0.00e+00 |
| PAY_0 | categorical | 0.987 | severe | χ²=12634.3 | 0.00e+00 |
| BILL_AMT1 | numeric | 0.000 | stable | KS=0.000 | 1.00e+00 |
| PAY_AMT1 | numeric | 0.000 | stable | KS=0.000 | 1.00e+00 |

## Drift multivariado (MMD² / Energy)

Limiar calibrado por bootstrap (n=200, quantil 0.95).

| Métrica | Valor | Limiar | Driftado |
| --- | --- | --- | --- |
| MMD² | 0.0763 | 0.0039 | True |
| Energy | 0.3354 | 0.0398 | True |

## Performance do modelo por mês

| Mês | ROC-AUC | Queda vs baseline (concept drift) | Concept drift ajustado | Label drift P(Y) |
| --- | --- | --- | --- | --- |
| M1 | 0.6561 | 0.1052 | 0.0129 | 0.0021 |
| M2 | 0.6418 | 0.1195 | 0.0262 | 0.0011 |
| M3 | 0.6248 | 0.1365 | 0.0402 | 0.0014 |
| M4 | 0.6130 | 0.1484 | 0.0535 | 0.0017 |
| M5 | 0.6079 | 0.1534 | 0.0854 | 0.0014 |
| M6 | 0.6087 | 0.1526 | 0.0873 | -0.0009 |

## Tipos de drift

| Tipo | Detectado |
| --- | --- |
| data_drift | True |
| label_drift | False |
| concept_drift | True |
| multivariate_drift | True |

> Proveniência: o **PSI do projeto** (binning quantílico) é a métrica de decisão e dirige o breach/SLO. O PSI reportado no HTML do Evidently usa binning próprio e pode divergir numericamente — é evidência, não decisão.
