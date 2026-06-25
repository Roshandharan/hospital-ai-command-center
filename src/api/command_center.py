"""
Command center API routes.
Serves pre-generated hospital data and SSE live feed.
"""
from __future__ import annotations

import asyncio
import json
import random
import uuid
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, StreamingResponse

log = structlog.get_logger(__name__)
router = APIRouter()

DATA_DIR = Path(__file__).parent.parent.parent / "data"
STATIC_DIR = Path(__file__).parent.parent.parent / "static"

# In-memory state — loaded once, mutated by live feed
_census = None
_or_schedule = None
_adt_events = []
_operational_metrics = None
_sse_clients: list[asyncio.Queue] = []


def load_data():
    global _census, _or_schedule, _adt_events, _operational_metrics
    try:
        with open(DATA_DIR / "census.json") as f:
            _census = json.load(f)
        with open(DATA_DIR / "or_schedule.json") as f:
            _or_schedule = json.load(f)
        with open(DATA_DIR / "adt_events.json") as f:
            _adt_events = json.load(f)
        with open(DATA_DIR / "operational_metrics.json") as f:
            _operational_metrics = json.load(f)
        log.info("data.loaded",
                 patients=_census["summary"]["occupied"],
                 or_cases=_or_schedule["summary"]["total_cases"],
                 adt_events=len(_adt_events))
    except Exception as e:
        log.error("data.load_failed", error=str(e))


async def broadcast(event_type: str, data: dict):
    """Send SSE event to all connected clients."""
    dead = []
    for q in _sse_clients:
        try:
            await q.put({"type": event_type, "data": data})
        except Exception:
            dead.append(q)
    for q in dead:
        _sse_clients.remove(q)


async def live_feed_loop():
    """Background task: generate new ADT events and broadcast every 12 seconds."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    log.info("live_feed.starting")

    UNITS = ["ED", "MICU", "CCU", "SICU", "5NORTH", "3EAST", "ORTHO", "PACU"]
    DIAGNOSES = {
        "ED": ["Chest pain", "Dyspnea", "Abdominal pain", "Syncope", "Sepsis", "Stroke", "GI bleed"],
        "MICU": ["Septic shock", "ARDS", "Respiratory failure", "DKA"],
        "CCU": ["STEMI", "NSTEMI", "CHF exacerbation", "Atrial fibrillation"],
        "SICU": ["Post-op monitoring", "Bowel resection", "Trauma"],
        "5NORTH": ["Neutropenic fever", "Chemo side effects", "Lymphoma"],
        "3EAST": ["Pneumonia", "COPD exacerbation", "UTI", "Cellulitis"],
        "ORTHO": ["Hip replacement", "Knee replacement", "Fracture fixation"],
        "PACU": ["Post-op monitoring", "Pain management"],
    }
    FIRST_NAMES = ["James", "Mary", "Michael", "Jennifer", "Robert", "Linda",
                   "David", "Patricia", "Carlos", "Maria", "Wei", "Ana"]
    LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Garcia", "Martinez",
                  "Chen", "Kim", "Patel", "Rodriguez", "Wilson", "Anderson"]
    EVENT_TYPES = ["ADMIT", "DISCHARGE", "TRANSFER", "UPDATE"]
    EVENT_WEIGHTS = [40, 25, 20, 15]

    await asyncio.sleep(5)
    log.info("live_feed.started")

    while True:
        try:
            unit = random.choice(UNITS)
            sex = random.choice(["M", "F"])
            age = random.randint(18, 92)
            event_type = random.choices(EVENT_TYPES, weights=EVENT_WEIGHTS)[0]
            acuity = random.choices(
                ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
                weights=[15, 30, 35, 20]
            )[0]

            readmission = round(random.uniform(0.05, 0.92), 3)
            deterioration = round(random.uniform(0.02, 0.88), 3)
            discharge_today = round(random.uniform(0.02, 0.85), 3)
            sepsis_risk = round(random.uniform(0.01, 0.75), 3)

            if acuity == "CRITICAL":
                readmission = round(random.uniform(0.65, 0.95), 3)
                deterioration = round(random.uniform(0.60, 0.90), 3)
                discharge_today = round(random.uniform(0.02, 0.10), 3)
                sepsis_risk = round(random.uniform(0.40, 0.85), 3)
            elif acuity == "HIGH":
                readmission = round(random.uniform(0.45, 0.70), 3)
                deterioration = round(random.uniform(0.30, 0.60), 3)

            first = random.choice(FIRST_NAMES)
            last = random.choice(LAST_NAMES)

            event = {
                "event_id": str(uuid.uuid4())[:12],
                "event_type": event_type,
                "timestamp": datetime.now().isoformat(),
                "mrn": f"MRN{random.randint(100000, 999999)}",
                "patient_name": f"{first} {last}",
                "age": age,
                "sex": sex,
                "unit": unit,
                "bed": f"{unit}-{random.randint(1, 20):02d}",
                "diagnosis": random.choice(DIAGNOSES.get(unit, DIAGNOSES["3EAST"])),
                "acuity": acuity,
                "attending": random.choice(["Dr. Chen", "Dr. Williams", "Dr. Patel",
                                            "Dr. O'Brien", "Dr. Rodriguez", "Dr. Kim"]),
                "scores": {
                    "readmission_30d": readmission,
                    "deterioration": deterioration,
                    "discharge_today": discharge_today,
                    "discharge_tomorrow": round(random.uniform(0.15, 0.75), 3),
                    "sepsis_risk": sepsis_risk,
                    "los_predicted_days": random.randint(1, 14),
                },
                "top_risk_factors": random.sample([
                    "Age > 75", "Prior admissions", "High CCI",
                    "Medicaid payer", "Language barrier", "Distance > 15mi",
                    "Prior no-shows", "Multiple comorbidities",
                    "High medication count", "Recent ED visit",
                ], k=2),
                "intervention": _intervention(acuity, readmission, sepsis_risk),
            }

            # Add to in-memory ADT log
            _adt_events.insert(0, event)
            if len(_adt_events) > 200:
                _adt_events.pop()

            # Broadcast to SSE clients
            await broadcast("adt_event", event)

            # Also broadcast updated summary every 3 events
            if random.random() > 0.7 and _census:
                await broadcast("stats_update", _get_live_stats())

            log.info("live_feed.event_sent", event_type=event_type,
                     unit=unit, acuity=acuity, clients=len(_sse_clients))

        except Exception as e:
            log.error("live_feed.error", error=str(e))

        await asyncio.sleep(12)


def _intervention(acuity, readmission, sepsis_risk):
    actions = []
    if sepsis_risk > 0.4:
        actions.append("Sepsis bundle activation")
    if readmission > 0.6:
        actions.append("Care management referral")
    if acuity in ("CRITICAL", "HIGH"):
        actions.append("Increase monitoring frequency")
    if not actions:
        actions.append("Standard care protocol")
    return {
        "priority": "STAT" if acuity == "CRITICAL" else "URGENT" if acuity == "HIGH" else "ROUTINE",
        "actions": actions,
    }


def _get_live_stats():
    if not _census:
        return {}
    s = _census["summary"]
    return {
        "occupied": s["occupied"],
        "total_beds": s["total_beds"],
        "occupancy_pct": s["occupancy_pct"],
        "critical_count": s["critical_count"],
        "high_count": s["high_count"],
        "medium_count": s["medium_count"],
        "low_count": s["low_count"],
        "discharge_today": s["discharge_today"],
        "sepsis_alerts": s["sepsis_alerts"],
        "timestamp": datetime.now().isoformat(),
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    html_path = STATIC_DIR / "dashboard.html"
    if not html_path.exists():
        return HTMLResponse("<h1>Dashboard not found</h1>", status_code=404)
    return HTMLResponse(html_path.read_text())


@router.get("/api/v1/census")
async def get_census():
    if not _census:
        load_data()
    return _census


@router.get("/api/v1/census/unit/{unit_id}")
async def get_unit(unit_id: str):
    if not _census:
        load_data()
    unit = _census["units"].get(unit_id.upper())
    if not unit:
        return {"error": "Unit not found"}
    return unit


@router.get("/api/v1/patient/{mrn}")
async def get_patient(mrn: str):
    if not _census:
        load_data()
    for unit in _census["units"].values():
        for bed in unit["beds"]:
            if bed["patient"] and bed["patient"]["mrn"] == mrn:
                return bed["patient"]
    return {"error": "Patient not found"}


@router.get("/api/v1/or-schedule")
async def get_or_schedule():
    if not _or_schedule:
        load_data()
    return _or_schedule


@router.get("/api/v1/adt-events")
async def get_adt_events(limit: int = 50):
    return {"events": _adt_events[:limit], "total": len(_adt_events)}


@router.get("/api/v1/operational-metrics")
async def get_operational_metrics():
    if not _operational_metrics:
        load_data()
    return _operational_metrics


@router.get("/api/v1/stats")
async def get_stats():
    return _get_live_stats()


@router.get("/api/v1/health")
async def health():
    return {
        "status": "healthy",
        "version": "2.0.0",
        "environment": "production",
        "hospital": "Hospital AI Command Center",
        "data_loaded": _census is not None,
        "patients": _census["summary"]["occupied"] if _census else 0,
        "sse_clients": len(_sse_clients),
    }


@router.get("/api/v1/stream")
async def stream(request: Request):
    """Server-Sent Events endpoint — streams live ADT events to dashboard."""
    q: asyncio.Queue = asyncio.Queue()
    _sse_clients.append(q)
    log.info("sse.client_connected", total_clients=len(_sse_clients))

    async def event_generator() -> AsyncGenerator[str, None]:
        # Send initial snapshot immediately
        if _census:
            data = json.dumps({"type": "snapshot", "data": {
                "summary": _census["summary"],
                "recent_events": _adt_events[:10],
            }})
            yield f"data: {data}\n\n"

        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(q.get(), timeout=30.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    # Heartbeat to keep connection alive
                    yield f"data: {json.dumps({'type': 'heartbeat', 'ts': datetime.now().isoformat()})}\n\n"
        except Exception as e:
            log.warning("sse.client_error", error=str(e))
        finally:
            if q in _sse_clients:
                _sse_clients.remove(q)
            log.info("sse.client_disconnected", total_clients=len(_sse_clients))

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
