"""
Hospital AI Command Center — API routes.
Serves census, OR schedule, ADT events, operational metrics,
agent recommendations, and action accountability log.
Real LangGraph pipeline available via /api/v1/pipeline/run.
"""
from __future__ import annotations

import asyncio
import json
import random
import uuid
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Literal, Optional

import structlog
from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

log = structlog.get_logger(__name__)
router = APIRouter()

# Pipeline singletons — initialized lazily on first use
_pipeline_instance = None


def _get_pipeline():
    global _pipeline_instance
    if _pipeline_instance is None:
        try:
            from src.core import get_pipeline
            _pipeline_instance = get_pipeline()
        except Exception as e:
            log.warning("pipeline.init_failed", error=str(e))
    return _pipeline_instance

DATA_DIR   = Path(__file__).parent.parent.parent / "data"
STATIC_DIR = Path(__file__).parent.parent.parent / "static"

# ── In-memory state ───────────────────────────────────────────────────────────
_census: dict = {}
_or_schedule: dict = {}
_adt_events: list = []
_operational_metrics: dict = {}
_action_log: list = []          # accountability trail
_sse_clients: dict[int, asyncio.Queue] = {}  # id(queue) → queue for O(1) removal
_event_counter = 0

STAFF = [
    {"name": "Dr. Sarah Chen",      "role": "Attending — Internal Medicine"},
    {"name": "Dr. Marcus Williams", "role": "Attending — Cardiology"},
    {"name": "Dr. Priya Patel",     "role": "Attending — Oncology"},
    {"name": "Dr. James O'Brien",   "role": "Hospitalist"},
    {"name": "RN Martinez",         "role": "Charge Nurse — ED"},
    {"name": "RN Johnson",          "role": "Charge Nurse — MICU"},
    {"name": "NP Rivera",           "role": "Nurse Practitioner"},
    {"name": "PA Thompson",         "role": "Physician Assistant"},
    {"name": "Rapid Response Team", "role": "RRT"},
    {"name": "Care Management",     "role": "Care Coordinator"},
]

AGENTS = [
    {"id": "RiskAgent",      "name": "Risk Stratification Agent", "icon": "📊", "color": "#2563eb"},
    {"id": "SepsisAgent",    "name": "Sepsis Surveillance Agent",  "icon": "🦠", "color": "#dc2626"},
    {"id": "DischargeAgent", "name": "Discharge Planning Agent",   "icon": "🏠", "color": "#16a34a"},
    {"id": "BedAgent",       "name": "Bed Management Agent",       "icon": "🛏",  "color": "#7c3aed"},
    {"id": "TriageAgent",    "name": "Triage Optimization Agent",  "icon": "⚡", "color": "#ea580c"},
    {"id": "MedSafetyAgent", "name": "Medication Safety Agent",    "icon": "💊", "color": "#0891b2"},
]


def load_data():
    global _census, _or_schedule, _adt_events, _operational_metrics
    try:
        with open(DATA_DIR / "census.json")               as f: _census               = json.load(f)
        with open(DATA_DIR / "or_schedule.json")          as f: _or_schedule          = json.load(f)
        with open(DATA_DIR / "adt_events.json")           as f: _adt_events           = json.load(f)
        with open(DATA_DIR / "operational_metrics.json")  as f: _operational_metrics  = json.load(f)
        log.info("data.loaded",
                 patients=_census["summary"]["occupied"],
                 or_cases=_or_schedule["summary"]["total_cases"],
                 adt_events=len(_adt_events))
    except Exception as e:
        log.error("data.load_failed", error=str(e))


def _build_agent_recommendations(acuity: str, scores: dict, diagnosis: str) -> list[dict]:
    """Generate realistic agent recommendations for a patient event."""
    recs = []
    r = scores.get("readmission_30d", 0)
    d = scores.get("deterioration", 0)
    s = scores.get("sepsis_risk", 0)
    disch = scores.get("discharge_today", 0)
    los = scores.get("los_predicted_days", 3)

    # RiskAgent — always fires
    risk_level = "HIGH" if r > 0.6 else "MEDIUM" if r > 0.35 else "LOW"
    recs.append({
        "agent": "RiskAgent",
        "agent_name": "Risk Stratification Agent",
        "icon": "📊",
        "color": "#2563eb",
        "confidence": round(random.uniform(0.78, 0.95), 2),
        "priority": "URGENT" if r > 0.6 else "ROUTINE",
        "recommendation": f"30-day readmission risk: {int(r*100)}% ({risk_level}). "
                          f"Predicted LOS: {los} days. "
                          + ("Enroll in post-discharge follow-up program." if r > 0.5 else "Standard discharge planning."),
        "actions": [
            {"label": "Enroll in Follow-Up Program", "type": "primary"} if r > 0.5 else {"label": "Standard Protocol", "type": "secondary"},
            {"label": "Care Mgmt Referral", "type": "primary"} if r > 0.6 else {"label": "Chart Risk Score", "type": "secondary"},
        ],
        "assign_to": "Care Management" if r > 0.5 else "NP Rivera",
    })

    # SepsisAgent — fires if sepsis risk elevated
    if s > 0.25 or acuity in ("CRITICAL", "HIGH"):
        bundle_complete = random.random() > 0.5
        recs.append({
            "agent": "SepsisAgent",
            "agent_name": "Sepsis Surveillance Agent",
            "icon": "🦠",
            "color": "#dc2626",
            "confidence": round(random.uniform(0.82, 0.97), 2),
            "priority": "STAT" if s > 0.5 else "URGENT",
            "recommendation": f"Sepsis risk: {int(s*100)}%. "
                              + ("SEP-1 bundle NOT complete — lactate pending, blood cultures ordered." if not bundle_complete
                                 else "SEP-1 bundle initiated. Reassess in 3 hours."),
            "actions": [
                {"label": "Initiate SEP-1 Bundle", "type": "danger"} if not bundle_complete else {"label": "Reassess in 3h", "type": "warning"},
                {"label": "Order Lactate + Cultures", "type": "danger"} if not bundle_complete else {"label": "Bundle Complete ✓", "type": "success"},
            ],
            "assign_to": "Rapid Response Team" if s > 0.5 else "Dr. Sarah Chen",
            "time_sensitive": True,
            "window_hours": 6,
        })

    # DischargeAgent
    recs.append({
        "agent": "DischargeAgent",
        "agent_name": "Discharge Planning Agent",
        "icon": "🏠",
        "color": "#16a34a",
        "confidence": round(random.uniform(0.71, 0.91), 2),
        "priority": "ROUTINE" if disch < 0.4 else "URGENT",
        "recommendation": f"Discharge probability today: {int(disch*100)}%. "
                          + (f"Target discharge: {random.choice(['Today 14:00','Today 16:00','Tomorrow 10:00'])}. "
                             f"Barriers: {random.choice(['PT eval pending','SNF placement needed','Patient education incomplete','Awaiting final labs'])}.")
                          if disch > 0.3 else f"Estimated discharge in {los} days. No immediate barriers identified.",
        "actions": [
            {"label": "Set Discharge Target", "type": "primary"},
            {"label": "Order Discharge Meds", "type": "secondary"} if disch > 0.3 else {"label": "Initiate Planning", "type": "secondary"},
        ],
        "assign_to": "Care Management" if disch > 0.5 else "NP Rivera",
    })

    # BedAgent
    if acuity in ("CRITICAL", "HIGH"):
        recs.append({
            "agent": "BedAgent",
            "agent_name": "Bed Management Agent",
            "icon": "🛏",
            "color": "#7c3aed",
            "confidence": round(random.uniform(0.74, 0.92), 2),
            "priority": "URGENT",
            "recommendation": f"Current unit capacity at {random.randint(78, 95)}%. "
                              f"Recommend {random.choice(['telemetry stepdown','MICU transfer','dedicated isolation room','monitored bed'])} "
                              f"based on acuity trajectory.",
            "actions": [
                {"label": "Request Bed Transfer", "type": "primary"},
                {"label": "Place Bed Request", "type": "secondary"},
            ],
            "assign_to": "RN Johnson",
        })

    # MedSafetyAgent — fires for complex patients
    if random.random() > 0.4 or acuity in ("CRITICAL", "HIGH"):
        flags = random.choice([
            "Potential interaction: Warfarin + Amoxicillin. Monitor INR closely.",
            "Renal dose adjustment needed: Metformin — eGFR 32.",
            "High-alert medication: Heparin drip — requires dual-nurse verification.",
            "Allergy conflict: Ordered Penicillin — PCN allergy documented.",
            "QT prolongation risk: Azithromycin + Amiodarone combination.",
            "No drug interactions detected. Current regimen reviewed.",
        ])
        is_critical_flag = any(w in flags for w in ["Allergy conflict", "High-alert"])
        recs.append({
            "agent": "MedSafetyAgent",
            "agent_name": "Medication Safety Agent",
            "icon": "💊",
            "color": "#0891b2",
            "confidence": round(random.uniform(0.88, 0.99), 2),
            "priority": "STAT" if is_critical_flag else "ROUTINE",
            "recommendation": flags,
            "actions": [
                {"label": "Acknowledge Flag", "type": "primary" if is_critical_flag else "secondary"},
                {"label": "Modify Order", "type": "danger"} if is_critical_flag else {"label": "No Action Needed", "type": "secondary"},
            ],
            "assign_to": "Dr. Sarah Chen",
        })

    # TriageAgent — only for ED patients
    if "ED" in diagnosis.upper() or acuity == "CRITICAL" or random.random() > 0.7:
        recs.append({
            "agent": "TriageAgent",
            "agent_name": "Triage Optimization Agent",
            "icon": "⚡",
            "color": "#ea580c",
            "confidence": round(random.uniform(0.76, 0.93), 2),
            "priority": "URGENT" if acuity in ("CRITICAL","HIGH") else "ROUTINE",
            "recommendation": f"ESI level: {random.choice(['ESI-1','ESI-2','ESI-2','ESI-3'])} assigned. "
                              f"Door-to-physician: {random.randint(8,38)} min. "
                              f"Recommend {random.choice(['immediate physician evaluation','expedited triage','fast-track pathway','standard triage'])}.",
            "actions": [
                {"label": "Assign to Fast Track", "type": "primary"},
                {"label": "Flag for Rapid Assessment", "type": "warning"},
            ],
            "assign_to": "RN Martinez",
        })

    return recs


async def broadcast(event_type: str, data: dict):
    dead = []
    for qid, q in list(_sse_clients.items()):
        try:
            await q.put({"type": event_type, "data": data})
        except Exception:
            dead.append(qid)
    for qid in dead:
        _sse_clients.pop(qid, None)


async def live_feed_loop():
    global _event_counter
    log.info("live_feed.starting")

    UNITS = ["ED","MICU","CCU","SICU","5NORTH","3EAST","ORTHO","PACU"]
    DIAGNOSES_MAP = {
        "ED":     ["Chest pain","Dyspnea","Abdominal pain","Syncope","Sepsis","Stroke","GI bleed","Trauma"],
        "MICU":   ["Septic shock","ARDS","Respiratory failure","DKA","Hypertensive emergency"],
        "CCU":    ["STEMI","NSTEMI","CHF exacerbation","Atrial fibrillation","Cardiac arrest"],
        "SICU":   ["Post-op monitoring","Bowel resection","Trauma laparotomy"],
        "5NORTH": ["Neutropenic fever","Chemo side effects","Lymphoma","Lung cancer"],
        "3EAST":  ["Pneumonia","COPD exacerbation","UTI","Cellulitis","DVT","Hip fracture"],
        "ORTHO":  ["Hip replacement","Knee replacement","Fracture fixation","Spinal fusion"],
        "PACU":   ["Post-op monitoring","Pain management","Anesthesia recovery"],
    }
    NAMES_F = ["Mary","Jennifer","Linda","Patricia","Maria","Ana","Sarah","Jessica","Karen","Michelle"]
    NAMES_M = ["James","Michael","Robert","David","John","William","Carlos","Wei","Omar","Ahmed"]
    LAST    = ["Smith","Johnson","Williams","Brown","Garcia","Martinez","Chen","Kim","Patel","Rodriguez"]

    await asyncio.sleep(5)
    log.info("live_feed.started")

    while True:
        try:
            _event_counter += 1
            unit   = random.choice(UNITS)
            sex    = random.choice(["M","F"])
            age    = random.randint(18, 92)
            etype  = random.choices(["ADMIT","DISCHARGE","TRANSFER","UPDATE"], weights=[40,25,20,15])[0]
            acuity = random.choices(["CRITICAL","HIGH","MEDIUM","LOW"], weights=[15,30,35,20])[0]
            dx     = random.choice(DIAGNOSES_MAP.get(unit, DIAGNOSES_MAP["3EAST"]))
            fname  = random.choice(NAMES_F if sex=="F" else NAMES_M)
            name   = f"{fname} {random.choice(LAST)}"

            scores = {
                "readmission_30d":    round(random.triangular(0.05, 0.95, 0.25 if acuity=="LOW" else 0.7), 3),
                "deterioration":      round(random.triangular(0.02, 0.90, 0.15 if acuity=="LOW" else 0.65), 3),
                "discharge_today":    round(random.triangular(0.02, 0.90, 0.6 if acuity=="LOW" else 0.08), 3),
                "discharge_tomorrow": round(random.triangular(0.05, 0.90, 0.5 if acuity in ("LOW","MEDIUM") else 0.2), 3),
                "sepsis_risk":        round(random.triangular(0.01, 0.85, 0.05 if acuity=="LOW" else 0.45), 3),
                "los_predicted_days": random.randint(1,4) if acuity=="LOW" else random.randint(3,14),
            }

            agent_recs = _build_agent_recommendations(acuity, scores, dx)

            event = {
                "event_id":    str(uuid.uuid4())[:12],
                "event_type":  etype,
                "seq":         _event_counter,
                "timestamp":   datetime.now().isoformat(),
                "mrn":         f"MRN{random.randint(100000,999999)}",
                "patient_name": name,
                "age":         age,
                "sex":         sex,
                "unit":        unit,
                "bed":         f"{unit}-{random.randint(1,20):02d}",
                "diagnosis":   dx,
                "acuity":      acuity,
                "attending":   random.choice([s["name"] for s in STAFF if "Dr." in s["name"]]),
                "scores":      scores,
                "agent_recommendations": agent_recs,
                "prior_admissions": random.randint(0,4),
                "insurance":   random.choice(["Medicare","Medicaid","Blue Cross","United","Self-Pay","Aetna"]),
                "language":    random.choice(["English","English","English","Spanish","Mandarin","Tagalog"]),
                "allergies":   random.sample(["PCN","Sulfa","Codeine","Contrast","Latex"], random.randint(0,2)),
                "chief_complaint": dx,
                "vitals": {
                    "hr":   random.randint(55,130),
                    "sbp":  random.randint(80,190),
                    "dbp":  random.randint(45,110),
                    "spo2": round(random.uniform(88,100), 1),
                    "temp": round(random.uniform(36.0,39.8), 1),
                    "rr":   random.randint(10,28),
                },
                "ai_summary": _ai_summary(acuity, scores, agent_recs),
            }

            _adt_events.insert(0, event)
            if len(_adt_events) > 300: _adt_events.pop()

            await broadcast("adt_event", event)

            if _event_counter % 3 == 0 and _census:
                await broadcast("stats_update", _get_live_stats())

            log.info("live_feed.event_sent", seq=_event_counter, unit=unit, acuity=acuity)

        except Exception as e:
            log.error("live_feed.error", error=str(e))

        await asyncio.sleep(12)


def _ai_summary(acuity: str, scores: dict, recs: list) -> str:
    n_agents = len(recs)
    n_urgent = sum(1 for r in recs if r.get("priority") in ("STAT","URGENT"))
    n_actioned = sum(1 for r in recs if r.get("actioned"))
    s = scores.get("sepsis_risk",0)
    r = scores.get("readmission_30d",0)
    return (f"{n_agents} agents analyzed · {n_urgent} urgent recommendations · "
            f"{n_actioned}/{n_agents} actioned. "
            + (f"⚠ Sepsis risk {int(s*100)}% — bundle review needed. " if s > 0.4 else "")
            + (f"High readmission risk ({int(r*100)}%) — care management flagged." if r > 0.6 else ""))


def _get_live_stats() -> dict:
    if not _census: return {}
    s = _census["summary"]
    return {**s, "timestamp": datetime.now().isoformat(),
            "active_sse_clients": len(_sse_clients),
            "events_processed": _event_counter}


# ── Action endpoint ───────────────────────────────────────────────────────────

class ActionRequest(BaseModel):
    event_id:   str
    agent_id:   str
    decision:   Literal["accepted", "overridden", "escalated"]
    actor:      str
    actor_role: str
    note:       str = ""


@router.post("/api/v1/action")
async def log_action(req: ActionRequest):
    record = {
        "action_id":  str(uuid.uuid4())[:10],
        "event_id":   req.event_id,
        "agent_id":   req.agent_id,
        "decision":   req.decision,
        "actor":      req.actor,
        "actor_role": req.actor_role,
        "note":       req.note,
        "timestamp":  datetime.now().isoformat(),
    }
    _action_log.insert(0, record)
    if len(_action_log) > 500: _action_log.pop()

    # Broadcast accountability update to all clients
    await broadcast("action_logged", record)
    log.info("action.logged", agent=req.agent_id, decision=req.decision, actor=req.actor)
    return {"success": True, "action": record}


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    p = STATIC_DIR / "dashboard.html"
    return HTMLResponse(p.read_text() if p.exists() else "<h1>Dashboard not found</h1>", status_code=200 if p.exists() else 404)

@router.get("/api/v1/census")
async def get_census():
    if not _census: load_data()
    return _census

@router.get("/api/v1/census/unit/{unit_id}")
async def get_unit(unit_id: str):
    if not _census: load_data()
    return _census["units"].get(unit_id.upper(), {"error": "Unit not found"})

@router.get("/api/v1/patient/{mrn}")
async def get_patient(mrn: str):
    if not _census: load_data()
    for unit in _census["units"].values():
        for bed in unit["beds"]:
            if bed["patient"] and bed["patient"]["mrn"] == mrn:
                return bed["patient"]
    return {"error": "Patient not found"}

@router.get("/api/v1/or-schedule")
async def get_or_schedule():
    if not _or_schedule: load_data()
    return _or_schedule

@router.get("/api/v1/adt-events")
async def get_adt_events(limit: int = 50):
    return {"events": _adt_events[:limit], "total": len(_adt_events)}

@router.get("/api/v1/operational-metrics")
async def get_operational_metrics():
    if not _operational_metrics: load_data()
    return _operational_metrics

@router.get("/api/v1/stats")
async def get_stats():
    return _get_live_stats()

@router.get("/api/v1/agents")
async def get_agents():
    return {"agents": AGENTS, "staff": STAFF}

@router.get("/api/v1/actions")
async def get_actions(limit: int = 50):
    return {"actions": _action_log[:limit], "total": len(_action_log)}

@router.get("/api/v1/audit")
async def get_audit(limit: int = 100, agent_id: Optional[str] = None, decision: Optional[str] = None):
    records = _action_log
    if agent_id:
        records = [r for r in records if r.get("agent_id") == agent_id]
    if decision:
        records = [r for r in records if r.get("decision") == decision]
    return {
        "records": records[:limit],
        "total": len(records),
        "summary": {
            "accepted":  sum(1 for r in _action_log if r.get("decision") == "accepted"),
            "overridden": sum(1 for r in _action_log if r.get("decision") == "overridden"),
            "escalated":  sum(1 for r in _action_log if r.get("decision") == "escalated"),
        },
    }

@router.get("/api/v1/health")
async def health():
    pipeline_ready = _get_pipeline() is not None
    return {
        "status": "healthy",
        "version": "3.0.0",
        "system": "Hospital AI Command Center",
        "data_loaded": _census is not None,
        "patients": _census["summary"]["occupied"] if _census else 0,
        "sse_clients": len(_sse_clients),
        "events_processed": _event_counter,
        "actions_logged": len(_action_log),
        "pipeline_ready": pipeline_ready,
    }


# ── Pipeline endpoint ─────────────────────────────────────────────────────────

class PipelineRunRequest(BaseModel):
    mrn: str
    age: Optional[int] = None
    sex: Optional[str] = None
    patient_class: str = "I"
    location: str = ""
    diagnosis: Optional[str] = None
    language: Optional[str] = None
    event_type: str = "A01"


@router.post("/api/v1/pipeline/run")
async def run_pipeline(req: PipelineRunRequest):
    """Run the real LangGraph multi-agent pipeline for a given patient."""
    pipeline = _get_pipeline()
    if not pipeline:
        return {"error": "Pipeline not initialized", "success": False}

    from src.ingestion.hl7_parser import AdmissionData, ADTEventType, PatientIdentifiers
    admission = AdmissionData(
        patient=PatientIdentifiers(mrn=req.mrn, encounter_id=str(uuid.uuid4())[:12]),
        event_type=ADTEventType(req.event_type) if req.event_type in ("A01","A02","A03","A08") else ADTEventType.ADMIT,
        admit_datetime=datetime.now(),
        facility="Hospital AI Command Center",
        location=req.location,
        patient_class=req.patient_class,
        age=req.age,
        sex=req.sex,
        language=req.language,
        admitting_diagnosis=req.diagnosis,
    )

    try:
        result = await pipeline.run(admission)
        return {
            "success": True,
            "session_id": result["session_id"],
            "acuity_tier": result.get("acuity_tier"),
            "readmission_30d": result.get("readmission_30d"),
            "deterioration": result.get("deterioration"),
            "sepsis_risk": result.get("sepsis_risk"),
            "discharge_today": result.get("discharge_today"),
            "los_predicted_days": result.get("los_predicted_days"),
            "agent_recommendations": result.get("agent_recommendations", []),
            "intervention_plan": result.get("intervention_plan"),
            "rag_sources": result.get("rag_sources", []),
            "risk_factors": result.get("risk_factors", []),
            "pipeline_duration_ms": result.get("pipeline_duration_ms"),
            "demo_mode": result.get("risk_demo_mode", True),
        }
    except Exception as e:
        log.error("pipeline.run_failed", error=str(e))
        return {"success": False, "error": str(e)}

@router.get("/api/v1/stream")
async def stream(request: Request):
    q: asyncio.Queue = asyncio.Queue()
    qid = id(q)
    _sse_clients[qid] = q
    log.info("sse.connected", total=len(_sse_clients))

    async def gen() -> AsyncGenerator[str, None]:
        if _census:
            payload = json.dumps({"type": "snapshot", "data": {
                "summary": _census["summary"],
                "recent_events": _adt_events[:15],
            }})
            yield f"data: {payload}\n\n"
        try:
            while True:
                if await request.is_disconnected(): break
                try:
                    event = await asyncio.wait_for(q.get(), timeout=25.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'type':'heartbeat','ts':datetime.now().isoformat()})}\n\n"
        except Exception as e:
            log.warning("sse.error", error=str(e))
        finally:
            _sse_clients.pop(qid, None)
            log.info("sse.disconnected", total=len(_sse_clients))

    return StreamingResponse(gen(), media_type="text/event-stream",
        headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no","Connection":"keep-alive"})
