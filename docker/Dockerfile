# =============================================================================
# Estágio 1: Build - instala dependências com uv
# =============================================================================
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /build

COPY pyproject.toml uv.lock ./
COPY src/ src/

RUN uv sync --frozen --no-dev --no-install-project

# =============================================================================
# Estágio 2: Runtime - imagem leve com apenas o necessário
# =============================================================================
FROM python:3.12-slim AS runtime

RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

COPY --from=builder /build/.venv /app/.venv
COPY src/ /app/src/

# Modelos e dados NÃO são embutidos na imagem: montados via volume.
# Para um fallback offline, descomente as linhas abaixo com um modelo default:
# COPY models/model.joblib /app/models/model.joblib
# Crie os diretórios para os volumes:
RUN mkdir -p /app/models /app/data /app/reports

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Dashboard de observabilidade (Streamlit). O tracking é servido pelo MLflow.
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
