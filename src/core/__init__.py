"""Global singletons initialized at startup."""
import os
from pathlib import Path

from src.config import get_settings

settings = get_settings()

_scoring_engine = None
_rag = None
_pipeline = None


def get_scoring_engine():
    global _scoring_engine
    if _scoring_engine is None:
        from src.ml.risk_model import RiskScoringEngine
        _scoring_engine = RiskScoringEngine(Path(settings.model_artifacts_dir))
        _scoring_engine.load()
    return _scoring_engine


def get_rag():
    global _rag
    if _rag is None:
        from src.rag.retriever import ClinicalRAG
        _rag = ClinicalRAG()
        _rag.initialize()
    return _rag


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from src.agents.pipeline import ClinicalPipeline
        anthropic_client = None
        if settings.anthropic_api_key:
            try:
                import anthropic
                anthropic_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
            except ImportError:
                pass
        _pipeline = ClinicalPipeline(
            scoring_engine=get_scoring_engine(),
            rag=get_rag(),
            anthropic_client=anthropic_client,
            model=settings.anthropic_model,
            db_persist=bool(os.getenv("DATABASE_URL")),
        )
    return _pipeline
