from __future__ import annotations
from typing import Any, Optional, TypedDict

from src.ingestion.hl7_parser import AdmissionData


class PipelineState(TypedDict):
    # Input
    session_id: str
    admission: AdmissionData

    # Triage
    esi_level: Optional[str]
    triage_route: str
    triage_recommendation: Optional[dict]

    # Risk (XGBoost)
    readmission_30d: float
    deterioration: float
    sepsis_risk: float
    discharge_today: float
    discharge_tomorrow: float
    los_predicted_days: int
    acuity_tier: str
    risk_factors: list[dict]
    risk_demo_mode: bool

    # Sepsis
    sepsis_recommendation: Optional[dict]
    sepsis_bundle_status: Optional[str]

    # Discharge
    discharge_recommendation: Optional[dict]
    discharge_barriers: list[str]

    # Bed
    bed_recommendation: Optional[dict]

    # Med Safety
    med_safety_flags: list[dict]
    med_safety_recommendation: Optional[dict]

    # RAG
    rag_context: str
    rag_sources: list[str]

    # LLM Intervention
    intervention_plan: Optional[dict]

    # All agent recs (for dashboard)
    agent_recommendations: list[dict]

    # Audit
    audit_actions: list[str]
    errors: list[str]
    pipeline_duration_ms: int
