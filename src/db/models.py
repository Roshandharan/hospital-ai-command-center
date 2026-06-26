"""PostgreSQL schema for pipeline run persistence and accountability."""
from __future__ import annotations
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, Integer, JSON, String, Text, func, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class PipelineRun(Base):
    """One row per ADT event processed through the LangGraph pipeline."""
    __tablename__ = "pipeline_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    mrn: Mapped[str] = mapped_column(String(64), index=True)
    encounter_id: Mapped[Optional[str]] = mapped_column(String(64))
    event_type: Mapped[str] = mapped_column(String(16))
    unit: Mapped[Optional[str]] = mapped_column(String(32))
    bed: Mapped[Optional[str]] = mapped_column(String(32))
    diagnosis: Mapped[Optional[str]] = mapped_column(String(256))
    acuity_tier: Mapped[Optional[str]] = mapped_column(String(16))

    # XGBoost scores
    readmission_30d: Mapped[Optional[float]] = mapped_column(Float)
    deterioration: Mapped[Optional[float]] = mapped_column(Float)
    sepsis_risk: Mapped[Optional[float]] = mapped_column(Float)
    discharge_today: Mapped[Optional[float]] = mapped_column(Float)
    los_predicted_days: Mapped[Optional[int]] = mapped_column(Integer)

    # Agent outputs
    triage_output: Mapped[Optional[dict]] = mapped_column(JSON)
    sepsis_output: Mapped[Optional[dict]] = mapped_column(JSON)
    discharge_output: Mapped[Optional[dict]] = mapped_column(JSON)
    bed_output: Mapped[Optional[dict]] = mapped_column(JSON)
    medsafety_output: Mapped[Optional[dict]] = mapped_column(JSON)
    intervention_plan: Mapped[Optional[dict]] = mapped_column(JSON)
    agent_recommendations: Mapped[Optional[list]] = mapped_column(JSON)
    rag_sources: Mapped[Optional[list]] = mapped_column(JSON)
    risk_factors: Mapped[Optional[list]] = mapped_column(JSON)

    # Metadata
    pipeline_duration_ms: Mapped[Optional[int]] = mapped_column(Integer)
    pipeline_success: Mapped[bool] = mapped_column(Boolean, default=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)

    actions: Mapped[list[ActionRecord]] = relationship("ActionRecord", back_populates="run")


class ActionRecord(Base):
    """Immutable audit trail — every Accept/Override/Escalate decision."""
    __tablename__ = "action_records"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("pipeline_runs.id"), nullable=True)
    event_id: Mapped[str] = mapped_column(String(64), index=True)
    agent_id: Mapped[str] = mapped_column(String(64))
    decision: Mapped[str] = mapped_column(String(32))   # accepted | overridden | escalated
    actor: Mapped[str] = mapped_column(String(128))
    actor_role: Mapped[str] = mapped_column(String(128))
    note: Mapped[Optional[str]] = mapped_column(Text)
    timestamp: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)

    run: Mapped[Optional[PipelineRun]] = relationship("PipelineRun", back_populates="actions")
