# Hospital AI Command Center

A real-time clinical operations platform that puts AI agents directly into the clinical workflow — not as a reporting layer on top of the EHR, but as active participants who surface recommendations, assign accountability, and track whether anyone actually acted on them.

**Live demo:** https://hospital-ai-command-center.up.railway.app/dashboard

---

## The problem this solves

Traditional EHR systems are data repositories. They store everything but initiate nothing. A patient's sepsis risk might be calculable from their vitals, labs, and history — but the system just shows you the vitals. It's up to a nurse to notice the trajectory, flag the physician, and hope something happens before the 6-hour SEP-1 window closes.

The gap isn't data. It's the connection between data and action, and the accountability that something was done about it.

This system is an attempt to close that gap. Six AI agents run continuously against the patient population, each looking for a different category of risk. When an agent fires a recommendation, it gets routed to a specific staff member with a decision required: accept, override, or escalate. That decision is logged. The next shift can see who reviewed it and what they decided.

---

## What's running

**The command center** — a full-screen clinical dashboard covering the entire hospital in real time. Every bed across all units is visible, color-coded by acuity. Incoming ADT events (admissions, discharges, transfers) stream in via Server-Sent Events. Click any event or any bed to drill into the patient.

**Six AI agents**, running on every patient event:

- **Risk Stratification Agent** — 30-day readmission probability, predicted length of stay, care management routing
- **Sepsis Surveillance Agent** — time-sensitive, fires when sepsis risk is elevated, tracks SEP-1 bundle completion, 6-hour window alert
- **Discharge Planning Agent** — predicts discharge today/tomorrow, identifies barriers (PT pending, SNF placement, labs outstanding), surfaces checklist
- **Bed Management Agent** — monitors unit capacity, recommends transfers and appropriate care levels as acuity evolves
- **Triage Optimization Agent** — ED-focused, ESI level validation, door-to-physician time tracking
- **Medication Safety Agent** — reviews active orders for interactions, allergy conflicts, renal dose adjustments, high-alert medications

Each agent attaches its recommendation to the patient event with a confidence score and a named staff member to action it. The recommendation doesn't disappear after it's surfaced — it stays open until someone makes a decision.

**Action accountability** — every agent recommendation can be accepted, overridden, or escalated. The system records who made the call, in what role, at what time, and optionally why they overrode it. This creates an audit trail that no EHR provides: not just what happened to the patient, but what the AI recommended, what the human decided, and when.

**OR Board** — all operating rooms, live case status, surgeon and anesthesiologist, delay tracking, and a timeline view of the full surgical day.

**Operational metrics** — door-to-physician time, nurse staffing vs. need, OR on-time starts, patient satisfaction, 30-day readmission rate, and real-time capacity trend.

---

## Architecture

The stack is deliberately simple. No message queues, no microservices, no Kubernetes. One Python process, one in-memory state model, Server-Sent Events for the live feed.

```
FastAPI (single process, uvicorn)
  ├── /dashboard              → serves dashboard.html
  ├── /api/v1/census          → current bed map with full patient records
  ├── /api/v1/or-schedule     → operating room schedule
  ├── /api/v1/adt-events      → recent ADT event log
  ├── /api/v1/operational-metrics → ED, staffing, throughput, quality
  ├── /api/v1/stream          → SSE live feed (new events every 12s)
  ├── /api/v1/action  (POST)  → log a clinical decision
  └── /api/v1/actions         → accountability log

Background task: live_feed_loop()
  Generates synthetic ADT events every 12 seconds
  Runs agent recommendations on each event
  Broadcasts via SSE to all connected clients
```

The data model generates a realistic hospital census on startup — 115 patients across 9 units (ED, MICU, CCU, SICU, Oncology, General Medicine, Orthopedics, PACU, OR) with full vitals histories, lab results, medication orders, imaging orders, and clinical notes. The OR schedule has 10 rooms with 2–5 cases each, realistic timing, and on-time/delay tracking.

Patient data includes everything needed for the patient detail view: 24-hour vitals trends for HR, SBP, and SpO2; lab results with abnormal flags; active medication orders; imaging with results; and clinical notes from physicians and nursing.

---

## Running it

**Prerequisites:** Python 3.12, Docker (optional)

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/hospital-ai-command-center.git
cd hospital-ai-command-center

# Install
pip install -r requirements.txt

# Generate hospital data
python scripts/generate_data.py

# Start
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# Open
open http://localhost:8000/dashboard
```

With Docker:

```bash
docker build -t hospital-ai-command-center .
docker run -p 8000:8000 hospital-ai-command-center
```

---

## Deploying to Railway

1. Fork this repo and connect it to a new Railway project
2. Railway will detect the Dockerfile and build automatically
3. No databases required — the app is stateless, all data is generated in-memory at startup
4. Generate a public domain in Railway → Settings → Networking → Generate Domain → Port 8000
5. Open `/dashboard` on your new domain

The SSE live feed works through Railway's load balancer without any special configuration — it uses standard HTTP chunked transfer encoding, not WebSockets.

---

## The accountability model

The piece that makes this different from a dashboard is the action layer.

When a sepsis agent fires on a patient, it doesn't just show a number. It creates an open recommendation assigned to a specific person — say, the rapid response team. That recommendation stays open until someone clicks "Accept," "Override," or "Escalate." If they override it, they log why. If they escalate, it gets routed up.

The system tracks:
- Which agent fired the recommendation
- What the recommendation said
- Who the recommendation was assigned to
- What decision was made
- Who made it, in what role
- When
- Any override notes

This is the accountability log. It answers the question that traditional EHRs cannot: *did anyone actually look at this, and what did they decide?*

In a real deployment, this log would feed into quality review, compliance audits, and root cause analysis. A sentinel event becomes: "the sepsis agent fired at 14:10, it was assigned to the charge nurse, it was overridden at 14:23 with note 'patient already on protocol,' and the patient deteriorated at 16:00." That's a different kind of investigation than "the data was there."

---

## Technical notes

**Why Server-Sent Events instead of WebSockets**

WebSockets require a persistent TCP connection that many load balancers and reverse proxies don't handle cleanly. SSE is HTTP — it works through Railway, Nginx, Cloudflare, and everything else without special configuration. It's also unidirectional, which is all we need: the server pushes events, the client doesn't need to send anything back through the stream.

**Why single-process**

The live feed uses an asyncio background task that writes to in-memory state, and the SSE clients read from that state via asyncio Queues. With multiple processes, each process would have its own memory and its own background task — the client connected to process 1 would miss events generated by process 2. Redis pub/sub would solve this for a production multi-replica deployment. For a portfolio demo, single-process keeps the architecture transparent.

**Why synthetic data**

Real patient data is PHI and can't live in a demo. The synthetic generator produces statistically realistic distributions — age, payer mix, diagnosis codes, vitals ranges by acuity level, lab abnormality rates, OR scheduling patterns — based on publicly available hospital operations benchmarks. The goal is a demo that reads as realistic to someone who works in a hospital, not one that passes a statistical test.

---

## What's next

A few things I'd build if this were going into production:

- Connect to a real Cerner Millennium HL7 feed via the existing ingestion layer from the companion [Clinical AI Command Center](https://github.com/YOUR_USERNAME/clinical-ai-command-center) project
- Train the XGBoost risk models on the Snowflake HealtheAnalytics schema instead of synthetic data
- Persist the action log to PostgreSQL with proper user authentication so accountability is tied to real staff identities
- Push high-priority agent recommendations to a mobile app or pager system so the recommendation reaches the clinician rather than waiting for them to look at a screen
- Add a conversational layer: "show me all patients where the sepsis agent fired but was overridden in the last 24 hours"

---

## Stack

FastAPI · Python 3.12 · Server-Sent Events · Chart.js · Docker · Railway

---

Built as a portfolio project demonstrating the intersection of healthcare operations and applied AI. The clinical domain knowledge comes from two years of building production Cerner Millennium pipelines at Oracle Health.
