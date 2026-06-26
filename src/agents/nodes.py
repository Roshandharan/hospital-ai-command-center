"""
LangGraph agent node implementations.
Each node receives PipelineState, adds its outputs, returns updated state.
"""
from __future__ import annotations

import random
from typing import Any

import structlog

from src.agents.state import PipelineState

log = structlog.get_logger(__name__)


# ── Triage ────────────────────────────────────────────────────────────────────

async def triage_node(state: PipelineState) -> PipelineState:
    admission = state["admission"]
    log.info("agent.triage.start", session=state["session_id"])

    pc  = admission.patient_class
    esi = (
        "ESI-1" if state.get("deterioration", 0) > 0.75 else
        "ESI-2" if pc == "E" or state.get("deterioration", 0) > 0.50 else
        "ESI-3" if pc in ("I", "E") else
        "ESI-4"
    )
    route = (
        "ICU"        if esi in ("ESI-1", "ESI-2") and pc != "O" else
        "Inpatient"  if pc == "I" else
        "ED"         if pc == "E" else
        "Outpatient"
    )

    rec = {
        "agent": "TriageAgent",
        "agent_name": "Triage Optimization Agent",
        "color": "#ea580c",
        "confidence": 0.88,
        "priority": "STAT" if esi == "ESI-1" else "URGENT" if esi == "ESI-2" else "ROUTINE",
        "recommendation": (
            f"ESI level {esi} assigned. Route: {route}. "
            + ("Immediate physician evaluation required." if esi in ("ESI-1", "ESI-2") else "Standard triage pathway.")
        ),
        "actions": [
            {"label": "Confirm ESI Level", "type": "primary"},
            {"label": "Assign to Fast Track", "type": "secondary"},
        ],
        "assign_to": "RN Martinez",
    }

    state["esi_level"] = esi
    state["triage_route"] = route
    state["triage_recommendation"] = rec
    state["agent_recommendations"] = state.get("agent_recommendations", []) + [rec]
    state["audit_actions"] = state.get("audit_actions", []) + ["triage_complete"]
    log.info("agent.triage.done", esi=esi, route=route)
    return state


# ── Risk Scoring ──────────────────────────────────────────────────────────────

async def risk_node(state: PipelineState, scoring_engine: Any) -> PipelineState:
    from src.ml.risk_model import PatientFeatures

    admission = state["admission"]
    log.info("agent.risk.start", session=state["session_id"])

    features = PatientFeatures(
        age=admission.age or 50,
        sex_encoded={"F": 0, "M": 1}.get(admission.sex or "", 2),
        patient_class_encoded={"O": 0, "I": 1, "E": 2}.get(admission.patient_class, 1),
        prior_admissions_12m=random.randint(0, 4),
        prior_ed_visits_12m=random.randint(0, 3),
        prior_no_shows_12m=random.randint(0, 2),
        active_problem_count=random.randint(1, 8),
        medication_count=random.randint(1, 10),
        days_since_last_visit=random.uniform(5, 180),
        appt_lead_time_days=random.uniform(1, 21),
        appt_hour=admission.admit_datetime.hour,
        appt_day_of_week=admission.admit_datetime.weekday(),
        is_new_patient=random.choice([0, 1]),
        payer_type_encoded=random.choices([0, 1, 2, 3], [0.40, 0.35, 0.15, 0.10])[0],
        distance_miles=random.uniform(0.5, 25),
        language_barrier=1 if (admission.language and admission.language not in ("ENG", "ENGLISH", "EN")) else 0,
        charlson_index=random.uniform(0, 6),
        num_chronic_conditions=random.randint(0, 8),
    )

    scores = scoring_engine.score(features, patient_id=admission.patient.mrn)

    rec = {
        "agent": "RiskAgent",
        "agent_name": "Risk Stratification Agent",
        "color": "#2563eb",
        "confidence": 0.91,
        "priority": "URGENT" if scores.readmission_30d > 0.5 else "ROUTINE",
        "recommendation": (
            f"30-day readmission: {scores.readmission_30d:.0%} | "
            f"Deterioration: {scores.deterioration:.0%} | "
            f"Predicted LOS: {scores.los_predicted_days} days. "
            + ("High readmission risk — care management referral indicated." if scores.readmission_30d > 0.5
               else "Standard monitoring.")
        ),
        "actions": [
            {"label": "Enroll in Follow-Up", "type": "primary"},
            {"label": "Chart Risk Score", "type": "secondary"},
        ],
        "assign_to": "Care Management" if scores.readmission_30d > 0.5 else "NP Rivera",
    }

    state.update({
        "readmission_30d":    scores.readmission_30d,
        "deterioration":      scores.deterioration,
        "sepsis_risk":        scores.sepsis_risk,
        "discharge_today":    scores.discharge_today,
        "discharge_tomorrow": scores.discharge_tomorrow,
        "los_predicted_days": scores.los_predicted_days,
        "acuity_tier":        scores.acuity_tier,
        "risk_factors":       scores.top_risk_factors,
        "risk_demo_mode":     scores.demo_mode,
    })
    state["agent_recommendations"] = state.get("agent_recommendations", []) + [rec]
    state["audit_actions"] = state.get("audit_actions", []) + ["risk_scored"]
    log.info("agent.risk.done", acuity=scores.acuity_tier, readmission=round(scores.readmission_30d, 3))
    return state


# ── Sepsis ────────────────────────────────────────────────────────────────────

async def sepsis_node(state: PipelineState) -> PipelineState:
    s = state.get("sepsis_risk", 0)
    log.info("agent.sepsis.start", sepsis_risk=round(s, 3))
    bundle_complete = random.random() > 0.45
    priority = "STAT" if s > 0.5 else "URGENT"
    rec = {
        "agent": "SepsisAgent",
        "agent_name": "Sepsis Surveillance Agent",
        "color": "#dc2626",
        "confidence": 0.93,
        "priority": priority,
        "recommendation": (
            f"Sepsis risk: {s:.0%}. "
            + ("SEP-1 bundle NOT complete — lactate and blood cultures pending. Initiate immediately."
               if not bundle_complete else
               "SEP-1 bundle initiated. Reassess in 3 hours.")
        ),
        "actions": [
            {"label": "Initiate SEP-1 Bundle", "type": "danger"} if not bundle_complete
            else {"label": "Reassess in 3h", "type": "warning"},
            {"label": "Order Lactate + Cultures", "type": "danger"} if not bundle_complete
            else {"label": "Bundle Complete", "type": "success"},
        ],
        "assign_to": "Rapid Response Team" if s > 0.5 else "Dr. Sarah Chen",
        "time_sensitive": True,
        "window_hours": 6,
    }
    state["sepsis_recommendation"] = rec
    state["sepsis_bundle_status"] = "complete" if bundle_complete else "incomplete"
    state["agent_recommendations"] = state.get("agent_recommendations", []) + [rec]
    state["audit_actions"] = state.get("audit_actions", []) + ["sepsis_assessed"]
    return state


# ── Discharge ─────────────────────────────────────────────────────────────────

async def discharge_node(state: PipelineState) -> PipelineState:
    d = state.get("discharge_today", 0)
    r = state.get("readmission_30d", 0)
    log.info("agent.discharge.start", discharge_today=round(d, 3))
    barriers = random.sample([
        "PT evaluation pending", "SNF placement needed",
        "Medication reconciliation incomplete", "Awaiting final labs",
        "Family meeting required", "Home health order pending",
        "Patient education not completed",
    ], k=random.randint(1, 2))
    target = random.choice(["Today 14:00", "Today 16:00", "Tomorrow 10:00", "Tomorrow 14:00"])
    rec = {
        "agent": "DischargeAgent",
        "agent_name": "Discharge Planning Agent",
        "color": "#16a34a",
        "confidence": 0.87,
        "priority": "URGENT" if d > 0.5 else "ROUTINE",
        "recommendation": (
            f"Discharge probability today: {d:.0%}, tomorrow: {state.get('discharge_tomorrow', 0):.0%}. "
            + (f"Target: {target}. Barriers: {', '.join(barriers)}." if d > 0.3
               else f"Estimated discharge in {state.get('los_predicted_days', 3)} days.")
            + (" High readmission risk — post-discharge follow-up within 7 days required." if r > 0.5 else "")
        ),
        "actions": [
            {"label": "Set Discharge Target", "type": "primary"},
            {"label": "Order Discharge Meds", "type": "secondary"},
        ],
        "assign_to": "Care Management" if r > 0.5 else "NP Rivera",
    }
    state["discharge_recommendation"] = rec
    state["discharge_barriers"] = barriers
    state["agent_recommendations"] = state.get("agent_recommendations", []) + [rec]
    state["audit_actions"] = state.get("audit_actions", []) + ["discharge_assessed"]
    return state


# ── Bed Management ────────────────────────────────────────────────────────────

async def bed_node(state: PipelineState) -> PipelineState:
    acuity = state.get("acuity_tier", "MEDIUM")
    log.info("agent.bed.start", acuity=acuity)
    capacity = random.randint(78, 96)
    rec_action = random.choice(["telemetry stepdown", "monitored bed", "MICU transfer", "isolation room"])
    rec = {
        "agent": "BedAgent",
        "agent_name": "Bed Management Agent",
        "color": "#7c3aed",
        "confidence": 0.85,
        "priority": "URGENT",
        "recommendation": (
            f"Unit capacity at {capacity}%. {acuity} acuity patient — "
            f"recommend {rec_action} based on trajectory."
        ),
        "actions": [
            {"label": "Request Bed Transfer", "type": "primary"},
            {"label": "Place Bed Request", "type": "secondary"},
        ],
        "assign_to": "RN Johnson",
    }
    state["bed_recommendation"] = rec
    state["agent_recommendations"] = state.get("agent_recommendations", []) + [rec]
    state["audit_actions"] = state.get("audit_actions", []) + ["bed_assessed"]
    return state


# ── Medication Safety ─────────────────────────────────────────────────────────

async def medsafety_node(state: PipelineState) -> PipelineState:
    log.info("agent.medsafety.start")
    flags_pool = [
        ("Potential interaction: Warfarin + Amoxicillin — monitor INR closely.", False),
        ("Renal dose adjustment needed: Metformin — eGFR 32.", False),
        ("HIGH-ALERT: Heparin drip requires dual-nurse verification.", True),
        ("ALLERGY CONFLICT: Penicillin ordered — PCN allergy documented. HOLD order.", True),
        ("QT prolongation risk: Azithromycin + Amiodarone combination.", False),
        ("No significant drug interactions detected in current medication profile.", False),
        ("Dose adjustment recommended: Vancomycin — monitor trough levels.", False),
    ]
    flag_text, is_critical = random.choice(flags_pool)
    rec = {
        "agent": "MedSafetyAgent",
        "agent_name": "Medication Safety Agent",
        "color": "#0891b2",
        "confidence": 0.96,
        "priority": "STAT" if is_critical else "ROUTINE",
        "recommendation": flag_text,
        "actions": [
            {"label": "Acknowledge & Hold Order", "type": "danger"} if is_critical
            else {"label": "Review Interaction", "type": "warning"},
            {"label": "Contact Pharmacy", "type": "secondary"},
        ],
        "assign_to": "Clinical Pharmacist",
    }
    state["med_safety_flags"] = [{"flag": flag_text, "critical": is_critical}]
    state["med_safety_recommendation"] = rec
    state["agent_recommendations"] = state.get("agent_recommendations", []) + [rec]
    state["audit_actions"] = state.get("audit_actions", []) + ["medsafety_assessed"]
    return state


# ── RAG Retrieval ─────────────────────────────────────────────────────────────

async def rag_node(state: PipelineState, rag: Any) -> PipelineState:
    admission = state["admission"]
    acuity    = state.get("acuity_tier", "MEDIUM")
    sepsis    = state.get("sepsis_risk", 0)
    readmit   = state.get("readmission_30d", 0)

    query_parts = [f"patient acuity {acuity}"]
    if sepsis > 0.4:   query_parts.append("sepsis SEP-1 bundle")
    if readmit > 0.4:  query_parts.append("readmission risk discharge planning")
    if admission.age and admission.age >= 65:
        query_parts.append("geriatric elderly readmission")
    query = ". ".join(query_parts)

    ctx = rag.retrieve(query, n=4)
    state["rag_context"] = ctx.as_prompt_text()
    state["rag_sources"] = ctx.sources
    state["audit_actions"] = state.get("audit_actions", []) + ["rag_retrieved"]
    log.info("agent.rag.done", sources=ctx.sources)
    return state


# ── LLM Intervention ──────────────────────────────────────────────────────────

async def intervention_node(state: PipelineState, anthropic_client: Any, model: str) -> PipelineState:
    """Call Claude to synthesize agent outputs into a unified intervention plan."""
    if not anthropic_client:
        state["intervention_plan"] = _fallback_intervention(state)
        return state

    admission  = state["admission"]
    recs       = state.get("agent_recommendations", [])
    rag_ctx    = state.get("rag_context", "")
    risk_facts = state.get("risk_factors", [])

    rec_summary = "\n".join(
        f"- [{r.get('agent','?')}] {r.get('priority','?')}: {r.get('recommendation','')}"
        for r in recs
    )
    risk_summary = ", ".join(
        f"{f['feature']} ({f['direction']} risk)"
        for f in risk_facts[:3]
    ) or "none identified"

    prompt = f"""You are a clinical decision support AI. Synthesize the following agent outputs into a prioritized intervention plan.

PATIENT: MRN {admission.patient.mrn}, age {admission.age or 'unknown'}, class {admission.patient_class}
ACUITY: {state.get('acuity_tier','?')} | Readmission: {state.get('readmission_30d',0):.0%} | Sepsis: {state.get('sepsis_risk',0):.0%}
KEY RISK FACTORS: {risk_summary}

AGENT RECOMMENDATIONS:
{rec_summary}

RELEVANT CLINICAL GUIDELINES:
{rag_ctx}

Respond with a JSON object containing:
{{
  "priority_actions": ["action1", "action2", "action3"],
  "care_pathway": "brief description",
  "follow_up_timeframe": "e.g. 7 days",
  "escalation_threshold": "specific clinical criteria",
  "guidelines_applied": ["guideline1", "guideline2"]
}}
Return only valid JSON, no markdown."""

    try:
        response = await anthropic_client.messages.create(
            model=model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        import json
        text = response.content[0].text.strip()
        plan = json.loads(text)
        state["intervention_plan"] = plan
        log.info("agent.intervention.done", actions=len(plan.get("priority_actions", [])))
    except Exception as e:
        log.error("agent.intervention.failed", error=str(e))
        state["intervention_plan"] = _fallback_intervention(state)
        state["errors"] = state.get("errors", []) + [f"intervention_llm: {e}"]

    state["audit_actions"] = state.get("audit_actions", []) + ["intervention_planned"]
    return state


def _fallback_intervention(state: PipelineState) -> dict:
    """Deterministic fallback when LLM is unavailable."""
    actions = []
    if state.get("sepsis_risk", 0) > 0.4:
        actions.append("Initiate SEP-1 bundle — lactate, blood cultures, broad-spectrum antibiotics")
    if state.get("readmission_30d", 0) > 0.5:
        actions.append("Enroll in care management program for post-discharge follow-up within 7 days")
    if state.get("discharge_today", 0) > 0.4:
        actions.append("Initiate discharge planning — medication reconciliation and home health order")
    if not actions:
        actions.append("Continue standard monitoring per admission protocol")
    return {
        "priority_actions": actions,
        "care_pathway": f"{state.get('acuity_tier','MEDIUM')} acuity standard pathway",
        "follow_up_timeframe": "7 days" if state.get("readmission_30d", 0) > 0.5 else "30 days",
        "escalation_threshold": "Deterioration score >0.75 or sepsis risk >0.6",
        "guidelines_applied": ["CMS SEP-1", "CMS HRRP", "CMS CoP 482.13"],
    }
