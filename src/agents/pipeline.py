"""
LangGraph multi-agent clinical decision pipeline.
Graph: risk → triage → [sepsis, discharge, bed, medsafety] → rag → intervention → persist
"""
from __future__ import annotations

import time
import uuid
from functools import partial
from typing import Any, Optional

import structlog

from src.agents.nodes import (
    bed_node, discharge_node, intervention_node, medsafety_node,
    rag_node, risk_node, sepsis_node, triage_node,
)
from src.agents.state import PipelineState
from src.ingestion.hl7_parser import AdmissionData

log = structlog.get_logger(__name__)


class ClinicalPipeline:
    def __init__(
        self,
        scoring_engine: Any,
        rag: Any,
        anthropic_client: Optional[Any] = None,
        model: str = "claude-sonnet-4-6",
        db_persist: bool = False,
    ) -> None:
        self._scoring = scoring_engine
        self._rag     = rag
        self._client  = anthropic_client
        self._model   = model
        self._db_persist = db_persist
        self._graph = self._build_graph()

    def _build_graph(self):
        """Build LangGraph StateGraph — falls back to sequential if langgraph unavailable."""
        try:
            from langgraph.graph import StateGraph, END
            from src.agents.state import PipelineState

            g = StateGraph(PipelineState)

            # Wrap nodes with injected dependencies
            g.add_node("risk",         partial(risk_node,         scoring_engine=self._scoring))
            g.add_node("triage",       triage_node)
            g.add_node("sepsis",       sepsis_node)
            g.add_node("discharge",    discharge_node)
            g.add_node("bed",          bed_node)
            g.add_node("medsafety",    medsafety_node)
            g.add_node("rag",          partial(rag_node,          rag=self._rag))
            g.add_node("intervention", partial(intervention_node, anthropic_client=self._client, model=self._model))

            g.set_entry_point("risk")
            g.add_edge("risk",      "triage")
            g.add_edge("triage",    "sepsis")
            g.add_edge("sepsis",    "discharge")
            g.add_edge("discharge", "bed")
            g.add_edge("bed",       "medsafety")
            g.add_edge("medsafety", "rag")
            g.add_edge("rag",       "intervention")
            g.add_edge("intervention", END)

            log.info("pipeline.langgraph_ready")
            return g.compile()
        except ImportError:
            log.info("pipeline.sequential_mode", reason="langgraph not installed")
            return None

    async def run(self, admission: AdmissionData) -> dict:
        session_id = str(uuid.uuid4())
        t0 = time.perf_counter()

        initial_state: PipelineState = {
            "session_id":           session_id,
            "admission":            admission,
            "esi_level":            None,
            "triage_route":         "",
            "triage_recommendation": None,
            "readmission_30d":      0.0,
            "deterioration":        0.0,
            "sepsis_risk":          0.0,
            "discharge_today":      0.0,
            "discharge_tomorrow":   0.0,
            "los_predicted_days":   3,
            "acuity_tier":          "MEDIUM",
            "risk_factors":         [],
            "risk_demo_mode":       True,
            "sepsis_recommendation": None,
            "sepsis_bundle_status":  None,
            "discharge_recommendation": None,
            "discharge_barriers":   [],
            "bed_recommendation":   None,
            "med_safety_flags":     [],
            "med_safety_recommendation": None,
            "rag_context":          "",
            "rag_sources":          [],
            "intervention_plan":    None,
            "agent_recommendations": [],
            "audit_actions":        [],
            "errors":               [],
            "pipeline_duration_ms": 0,
        }

        try:
            if self._graph is not None:
                final = await self._graph.ainvoke(initial_state)
            else:
                final = await self._run_sequential(initial_state)
        except Exception as e:
            log.error("pipeline.failed", session=session_id, error=str(e))
            final = initial_state
            final["errors"] = [str(e)]

        duration_ms = int((time.perf_counter() - t0) * 1000)
        final["pipeline_duration_ms"] = duration_ms

        if self._db_persist:
            await self._persist(final)

        log.info(
            "pipeline.complete",
            session=session_id,
            duration_ms=duration_ms,
            recs=len(final.get("agent_recommendations", [])),
        )
        return final

    async def _run_sequential(self, state: PipelineState) -> PipelineState:
        """Fallback sequential execution when langgraph is not installed."""
        state = await risk_node(state, scoring_engine=self._scoring)
        state = await triage_node(state)
        state = await sepsis_node(state)
        state = await discharge_node(state)
        state = await bed_node(state)
        state = await medsafety_node(state)
        state = await rag_node(state, rag=self._rag)
        state = await intervention_node(state, anthropic_client=self._client, model=self._model)
        return state

    async def _persist(self, state: PipelineState) -> None:
        try:
            from src.db.session import get_db_session
            from src.db.models import PipelineRun
            admission = state["admission"]
            async with get_db_session() as session:
                run = PipelineRun(
                    session_id=state["session_id"],
                    mrn=admission.patient.mrn,
                    encounter_id=admission.patient.encounter_id,
                    event_type=admission.event_type.value,
                    unit=admission.location,
                    diagnosis=admission.admitting_diagnosis,
                    acuity_tier=state.get("acuity_tier"),
                    readmission_30d=state.get("readmission_30d"),
                    deterioration=state.get("deterioration"),
                    sepsis_risk=state.get("sepsis_risk"),
                    discharge_today=state.get("discharge_today"),
                    los_predicted_days=state.get("los_predicted_days"),
                    triage_output=state.get("triage_recommendation"),
                    sepsis_output=state.get("sepsis_recommendation"),
                    discharge_output=state.get("discharge_recommendation"),
                    bed_output=state.get("bed_recommendation"),
                    medsafety_output=state.get("med_safety_recommendation"),
                    intervention_plan=state.get("intervention_plan"),
                    agent_recommendations=state.get("agent_recommendations"),
                    rag_sources=state.get("rag_sources"),
                    risk_factors=state.get("risk_factors"),
                    pipeline_duration_ms=state.get("pipeline_duration_ms"),
                    pipeline_success=len(state.get("errors", [])) == 0,
                    error_message="; ".join(state.get("errors", [])) or None,
                )
                session.add(run)
        except Exception as e:
            log.error("pipeline.persist_failed", error=str(e))
