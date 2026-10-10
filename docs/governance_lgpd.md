# Governança de Dados e LGPD — Tech Challenge Fase 4

Este documento detalha o plano de governança, privacidade e mitigação de vieses do modelo de *credit
scoring*. O dataset de referência é o UCI *Default of Credit Card Clients* (ID 350, Taiwan, ~30k
clientes) — **dados públicos, anonimizados na origem**. Ainda assim, tratamos o pipeline como se os
dados identificassem pessoas, aplicando o princípio da minimização e as salvaguardas abaixo.

## 1. Tratamento de PII (Personally Identifiable Information)

| Categoria | Campos | Tratamento no pipeline |
| --------- | ------ | ---------------------- |
| **Identificador direto** | `ID` | **Removido na ingestão** (`src/data/ingest.py`) — não entra em treino, drift ou model card. |
| **Quasi-identificadores** | `SEX`, `AGE`, `EDUCATION`, `MARRIAGE` | Mantidos para o modelo (têm valor preditivo), mas **monitorados por fairness** e usados apenas de forma agregada em relatórios. |
| **Financeiros sensíveis** | `LIMIT_BAL`, `BILL_AMT1..6`, `PAY_AMT1..6`, `PAY_0..6` | Dados financeiros do titular; nunca publicados por linha — apenas distribuições/estatísticas agregadas em `reports/`. |
| **Rótulo** | `default_next_month` | Resultado observado de inadimplência; finalidade exclusivamente de treino/avaliação. |

**Minimização**: apenas colunas necessárias ao risco de crédito são retidas. `ID` é descartado
(pseudonimização não é necessária por ser um identificador sintético de linha). Nenhum dado pessoal é
serializado em relatórios ou logs (logs carregam apenas metadados de execução: `task`, latência, volume).

**Segurança**: credenciais (ex.: Kaggle) são modeladas como `SecretStr` (`src/core/config.py`) e nunca
comitadas; `.env` é ignorado pelo Git.

## 2. Base legal (LGPD)

- **Decisão de crédito / análise de risco**: LGPD **art. 7º, X** (legítimo interesse) e **art. 11,
  II, d** para dados sensíveis, quando aplicável — tratamento necessário para a análise de risco de
  crédito e proteção ao crédito (cf. art. 5º, II e a regulamentação do Bacen/CMN sobre risco).
- **Finalidade**: avaliação de risco de inadimplência e concessão de crédito. Os dados **não** são
  reutilizados para finalidade incompatível.
- **Transparência**: o titular tem direito a revisão de decisões automatizadas (LGPD **art. 20**); o
  pipeline preserva versão do modelo (MLflow Registry, aliases `champion`/`challenger`) para viabilizar
  explicação e contestação.

## 3. Retenção e ciclo de vida

| Artefato | Retenção | Justificativa |
| -------- | -------- | ------------- |
| Dados brutos (`data/raw/`) | versão do pipeline (DVC) | reprodutibilidade |
| Dataset de referência / snapshots | versão do pipeline | auditoria do modelo |
| Relatórios (`reports/**`) | histórico do repositório | evidência de governança |
| Logs (`logs/`) | rotação local (não versionados) | operação |
| Modelos (`models/`) | registry MLflow + alias | rastreabilidade de decisões |

Recomenda-se **anonimização irreversível ou descarte** ao final do ciclo de vida e a não retenção além
do necessário à finalidade.

## 4. Mitigação de vieses e fairness

- **Guardrails objetivos** (`src/observability/fairness.py`, `configs/params.yaml`):
  - **Razão de impacto dispar ≥ 0.8** (regra dos 80%) por `SEX` e faixas de `AGE`.
  - **Diferença de equal opportunity (TPR) ≤ 0.10** entre subgrupos.
- **Monitoramento contínuo**: fairness avaliada a cada snapshot e persistida em `reports/fairness/`.
- **Promoção de challenger**: um novo modelo só é promovível se **AUC ≥ champion** **E** os guardrails
  de fairness forem satisfeitos; **nunca** há promoção automática (`src/observability/retrain.py`).
- **Dados sensíveis como proxy de risco**: `SEX`, `MARRIAGE` e `EDUCATION` podem ser proxies espúrios;
  o plano é revisar a inclusão dessas features com o time jurídico/negócio e, se possível, removê-las,
  mantendo-as apenas para auditoria de viés.

## 5. Salvaguarda por dados sintéticos

Para desenvolvimento, demo e CI usa-se um **gerador sintético** (`scripts/generate_synthetic_data.py`)
com sinal preditivo realista e **sem qualquer pessoa real**. Isso permite rodar todo o pipeline offline
(CI network-free) sem risco de exposição.

## 6. Riscos residuais e controles

| Risco | Controle |
| ----- | -------- |
| Reidentificação por quasi-identificadores | agregação em relatórios; não publicar linhas |
| Viés discriminatório | fairness por subgrupo + guardrails de promoção |
| Decisão automatizada opaca | MLflow Registry + versionamento + direito de revisão |
| Exposição de segredos | `SecretStr` + `.env` ignorado + CI sem segredos |
