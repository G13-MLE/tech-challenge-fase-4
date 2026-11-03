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

- **Decisão de crédito / análise de risco**: LGPD **art. 7º, X** — tratamento necessário *para a
  proteção do crédito*, inclusive quanto ao disposto na legislação pertinente (Bacen/CMN). Como base
  secundária, quando o titular já é cliente e há expectativa legítima, admite-se o **art. 7º, IX**
  (legítimo interesse), sempre precedido de teste de balanceamento.
- **Não há tratamento de dado pessoal sensível** (art. 5º, II): o dataset contém dados financeiros e
  demográficos, mas não origem racial/étnica, convicção religiosa, opinião política, dado de saúde,
  biometria ou vida sexual. O art. 11 **não** é invocado; se um dado sensível fosse incorporado, o
  tratamento exigiria hipótese autônoma do art. 11.
- **Finalidade**: avaliação de risco de inadimplência e concessão de crédito. Os dados **não** são
  reutilizados para finalidade incompatível (art. 6º, I).
- **Transparência e revisão**: o titular tem direito a **revisão de decisões automatizadas**
  (LGPD **art. 20**) e à explicação dos critérios; o pipeline preserva a versão do modelo no MLflow
  Registry (aliases `champion`/`challenger`) para viabilizar explicação e contestação.

## 3. Retenção e ciclo de vida

| Artefato | Retenção | Descarte | Justificativa |
| -------- | -------- | -------- | ------------- |
| Dados brutos (`data/raw/`) | 24 meses após a última execução que os utilizou | exclusão do cache DVC + `dvc gc` | reprodutibilidade limitada ao ciclo de auditoria |
| Dataset de referência e snapshots (`data/reference/`, `data/production/`) | 24 meses | idem | auditoria do modelo e dos incidentes de drift |
| Relatórios (`reports/**`) | 5 anos | remoção versionada com registro em changelog | evidência de governança e prestação de contas (Bacen/CMN) |
| Logs (`logs/`) | 90 dias, com rotação diária | rotação automática + expurgo | operação e investigação de incidentes |
| Modelos (`models/`, MLflow Registry) | 2 gerações (`champion` + `challenger`) | arquivamento no Registry | rastreabilidade de decisões e rollback |
| Segredos (`.env`) | enquanto válidos | revogação imediata | segurança |

**Direitos do titular (LGPD art. 18)**: confirmação de tratamento, acesso, correção, anonimização,
bloqueio ou eliminação de dados desnecessários, portabilidade, informação sobre compartilhamento e
revisão de decisão automatizada. O canal é o encarregado (DPO) da fintech; pedidos de eliminação são
atendidos em até 15 dias, removendo a linha do dataset de referência/produção e registrando a operação.
Como o identificador direto (`ID`) já é removido na ingestão, a eliminação por titular pressupõe o
mapeamento no sistema de origem — o pipeline analítico não mantém esse vínculo por design.

## 4. Mitigação de vieses e fairness

- **Guardrails objetivos** (`src/observability/fairness.py`, `configs/params.yaml`):
  - **Razão de impacto dispar ≥ 0.8** (regra dos 80%) por `SEX` e faixas de `AGE`.
  - **Diferença de equal opportunity (TPR) ≤ 0.10** entre subgrupos.
- **Promoção de challenger**: um novo modelo só é promovível se **AUC ≥ champion** **E** os guardrails
  de fairness forem satisfeitos; **nunca** há promoção automática (`src/observability/retrain.py`).
- **Monitoramento contínuo**: fairness avaliada a cada snapshot e persistida em `reports/fairness/`.

### 4.1 Mitigação implementada (não apenas medida)

`mitigate_fairness()` aplica **pós-processamento de limiar por subgrupo**: cada grupo recebe um limiar
de decisão calibrado para igualar seu TPR ao TPR global do modelo. O resultado é versionado em
`reports/fairness/fairness_mitigation.json` com o antes/depois por subgrupo e os limiares usados.

| Subgrupo | ΔTPR antes | ΔTPR depois | Limiares por grupo |
| -------- | ---------- | ----------- | ------------------ |
| `SEX` | 0.065 | **0.005** | 1 → 0.57 · 2 → 0.47 |
| `AGE_BIN` | 0.254 | **0.005** | `<30` → 0.18 · `30-44` → 0.38 · `45-59` → 0.62 · `60+` → 0.59 |

**Trade-off declarado**: equalizar o TPR aumenta a taxa de detecção de inadimplência no grupo
favorecido, o que **reduz a aprovação** desse grupo — ou seja, troca-se parte da paridade demográfica
por equal opportunity. A escolha do ponto de operação é decisão de negócio/jurídico, documentada e
revisável; o pipeline **mede** os dois lados e não promove modelo automaticamente.

### 4.2 Mitigação planejada (fora do escopo desta entrega)

`SEX`, `MARRIAGE` e `EDUCATION` podem ser proxies espúrios de risco. O plano é revisar a inclusão
dessas features com jurídico/negócio e, se possível, removê-las do treino, mantendo-as apenas para
auditoria de viés. Isso exige retreino e reavaliação de AUC — por isso está explicitamente separado da
mitigação já implementada (limiar por subgrupo).

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

## 7. Privacy by Design (7 princípios)

O projeto adota os **7 princípios fundacionais de Privacy by Design** (Cavoukian), implementados como
controles concretos no pipeline — não apenas declarações de intenção:

| # | Princípio | Controle no pipeline |
| - | --------- | -------------------- |
| 1 | **Proativo, não reativo; preventivo, não remediativo** | Data contracts (`src/contracts/schema.py`) e validação que **bloqueia** lotes inválidos (exit ≠ 0) antes de qualquer decisão; guardrails de drift (PSI/SLO) antecipam a degradação. |
| 2 | **Privacidade como padrão** | Apenas features com valor preditivo são retidas; `ID` (identificador direto) é **removido na ingestão** por padrão; nenhum dado pessoal por linha é serializado em `reports/`/`logs/`. |
| 3 | **Privacidade embutida no design** | Minimização e classificação de PII (`docs/data_dictionary.md`) fazem parte da arquitetura de dados; o split referência/produção e a dedup ocorrem antes do treino. |
| 4 | **Funcionalidade total (positive-sum)** | O pipeline entrega valor analítico (credit scoring + drift) **e** privacidade simultaneamente: fairness/fairness-by-subgroup e monitoramento não exigem expor PII. |
| 5 | **Segurança ponta-a-ponta (ciclo de vida)** | `SecretStr` para credenciais (nunca comitadas), `.env` ignorado, CI sem segredos, artefatos versionados (DVC/MLflow) e retenção documentada (seção 3). |
| 6 | **Visibilidade e transparência** | Logs JSON auditáveis (`logs/pipeline.ndjson` com `run_id`/`trace_id`), `reports/**` versionados e **revisão de decisões automatizadas** (LGPD art. 20) via MLflow Registry. |
| 7 | **Respeito pela privacidade do titular (user-centric)** | Direitos do titular preservados (revisão/contestação), dados sintéticos na dev/CI (sem pessoas reais) e mitigação de viés por subgrupo (`reports/fairness/`). |
