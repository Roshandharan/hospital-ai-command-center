FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y curl build-essential && rm -rf /var/lib/apt/lists/*

# Core requirements (always installed)
COPY requirements.txt .
RUN pip install --no-cache-dir fastapi==0.111.0 "uvicorn[standard]==0.29.0" \
    pydantic==2.7.1 pydantic-settings==2.2.1 httpx==0.27.0 \
    structlog==24.1.0 numpy==1.26.4

# ML + RAG + LangGraph (graceful fallback if install fails)
RUN pip install --no-cache-dir xgboost>=2.0.0 shap>=0.45.0 \
    scikit-learn>=1.4.0 || echo "ML packages optional — skipping"
RUN pip install --no-cache-dir chromadb>=0.5.0 sentence-transformers>=2.7.0 \
    || echo "RAG packages optional — skipping"
RUN pip install --no-cache-dir langgraph>=0.2.0 || echo "LangGraph optional — skipping"
RUN pip install --no-cache-dir anthropic>=0.30.0 || echo "Anthropic optional — skipping"

COPY src/ ./src/
COPY static/ ./static/
COPY data/ ./data/
COPY scripts/ ./scripts/

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

EXPOSE 8000
CMD uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
