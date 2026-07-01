# Hospital AI Command Center

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Railway-6366f1?style=flat-square&logo=railway)](https://hospital-ai-command-center-production.up.railway.app/dashboard)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-FF6B35?style=flat-square)](https://github.com/langchain-ai/langgraph)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-FF6600?style=flat-square)](https://xgboost.readthedocs.io)
[![Claude](https://img.shields.io/badge/Claude-Sonnet%204.6-CC785C?style=flat-square)](https://anthropic.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

**Live:** https://hospital-ai-command-center-production.up.railway.app/dashboard

---

Clinical AI platforms are usually bolted onto EHRs as reporting tools — they surface insights in dashboards that clinicians check once a day, long after the moment for intervention has passed. This project is built the other way: a real-time multi-agent system that runs the instant a patient event fires, scores risk across five dimensions, retrieves relevant clinical evidence from a vector knowledge base, synthesizes an intervention plan using Claude, and routes specific recommendations to specific people — with a permanent record of whether anyone acted on them.

The system handles the complete workflow from HL7 ADT message ingestion → LangGraph pipeline → risk scoring → clinical guideline retrieval → LLM synthesis → SSE push → interactive dashboard → accountability logging.

<video src="https://github.com/user-attachments/assets/ff3826e7-1216-4d55-9856-0faff9e6a950" controls width="100%"></video>

---

## Table of Contents

- [Architecture](#architecture)
- [Features](#features)
- [LangGraph Pipeline](#langgraph-pipeline)
- [ML Risk Models](#ml-risk-models)
- [Clinical Knowledge Retrieval (RAG)](#clinical-knowledge-retrieval-rag)
- [OR Board](#or-board)
- [Role-Based Access Control](#role-based-access-control)
- [Action Accountability](#action-accountability)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Local Setup](#local-setup)
- [Environment Variables](#environment-variables)
- [Docker](#docker)
- [Railway Deployment](#railway-deployment)
- [What's Real vs Synthetic](#whats-real-vs-synthetic)
- [Production Readiness](#production-readiness)
- [API Reference](#api-reference)
- [Research Foundation](#research-foundation)
- [References](#references)

---

## Architecture

```
                         ┌─────────────────────────────────────────────────┐
                         │            HL7 v2.x ADT Feed                    │
                         │   A01 Admit · A02 Transfer · A03 Discharge       │
                         │   A08 Update  (Cerner Millennium structure)      │
                         └───────────────────┬─────────────────────────────┘
                                             │ parse MSH · PID · PV1 · DG1
                                             ▼
                         ┌─────────────────────────────────────────────────┐
                         │         FastAPI  /api/v1/adt/ingest             │
                         │         Pydantic validation · structlog          │
                         └───────────────────┬─────────────────────────────┘
                                             │ AdmissionData
                                             ▼
                         ┌─────────────────────────────────────────────────┐
                         │        LangGraph StateGraph  (8 nodes)          │
                         │                                                  │
                         │  risk ──► triage ──► sepsis ──► discharge       │
                         │                  └──► bed ──► medsafety         │
                         │                              └──► rag           │
                         │                                   └──► intervention
                         └───────────────────┬─────────────────────────────┘
                                             │ PipelineState dict
                          ┌──────────────────┼──────────────────┐
                          ▼                  ▼                  ▼
                   XGBoost × 5         ChromaDB            Claude
                  + SHAP values      (cosine search)    claude-sonnet-4-6
                                                        JSON action plan
                                             │
                                             ▼
                         ┌─────────────────────────────────────────────────┐
                         │   PostgreSQL  pipeline_runs · action_records    │
                         │   (optional — in-memory fallback if no DB URL)  │
                         └───────────────────┬─────────────────────────────┘
                                             │
                                             ▼
                         ┌─────────────────────────────────────────────────┐
                         │        Server-Sent Events  /api/v1/stream       │
                         │        EventSource push on every pipeline run   │
                         └───────────────────┬─────────────────────────────┘
                                             │
                                             ▼
                         ┌─────────────────────────────────────────────────┐
                         │       Single-Page Dashboard  (vanilla JS)       │
                         │  Census · OR Board · Ops · Analytics · Agents   │
                         │  7-role RBAC · SHAP waterfall · dark mode       │
                         └─────────────────────────────────────────────────┘
```

---

## Features

| Feature | Detail |
|---|---|
| **Real-time ADT feed** | SSE stream processes HL7 events every 12 s; each event triggers the full 8-node pipeline |
| **5 XGBoost risk models** | Readmission 30d, deterioration, sepsis, discharge today, discharge tomorrow — all with SHAP |
| **8-node LangGraph pipeline** | Conditional routing; not every patient fires every agent — the dashboard shows which nodes ran |
| **Claude LLM synthesis** | Intervention node uses `claude-sonnet-4-6` to produce a prioritized, structured JSON action plan |
| **ChromaDB RAG** | 10 clinical guideline documents (CMS, SEP-1, HEDIS, NPSG) retrieved per-query via cosine search |
| **SHAP explainability** | Per-prediction waterfall charts in the dashboard; every score is grounded in contributing features |
| **OR Board** | Live surgical schedule across all OR rooms with 5-tab case detail modal (overview, patient, team, anesthesia, pre-op checklist) |
| **7-role RBAC** | Physician, Nurse, Case Manager, Pharmacist, OR Coordinator, Executive, Read-Only — each sees a different view |
| **Action accountability** | Accept / Override / Escalate decisions logged to `action_records` with actor, role, timestamp, and note |
| **PostgreSQL persistence** | Full pipeline run stored — all five scores, all agent outputs, SHAP values, RAG sources, intervention plan |
| **Zero-dependency demo mode** | Runs without PostgreSQL, Anthropic key, or pre-trained models — every component has a deterministic fallback |
| **Dark / light mode** | Full CSS variable theming with localStorage persistence; inline-style overrides via attribute selectors |
| **HL7 v2.x parser** | Handles Cerner Millennium message structure — MSH field indexing, PID CX identifier routing, PV1.19 encounter ID |

---

## LangGraph Pipeline

The pipeline is a `StateGraph` compiled into a runnable graph. All eight nodes receive and return a `PipelineState` TypedDict. Dependencies are injected with `functools.partial`.

```
  risk  ──────────────────────────────────────────────────────► triage
   │                                                               │
   │  XGBoost × 5 classifiers                                     │  ESI-1 → -2 → -3 → -4
   │  18 input features                                           │  ICU / Inpatient / ED / Outpatient
   │  SHAP TreeExplainer                                          │
   │  shap_by_model dict → dashboard waterfall charts            ▼
   │                                                           sepsis
   │                                                              │
   │                                                              │  SEP-1 bundle compliance
   │                                                              │  6-hour time window
   │                                                              ▼
   │                                                          discharge
   │                                                              │
   │                                                              │  barrier identification
   │                                                              │  target discharge time
   │                                                              ▼
   │                                                             bed
   │                                                              │
   │                                                              │  unit capacity check
   │                                                              │  transfer routing
   │                                                              ▼
   │                                                          medsafety
   │                                                              │
   │                                                              │  drug-drug interactions
   │                                                              │  renal dose adjustments
   │                                                              │  allergy conflict detection
   │                                                              ▼
   │                                                             rag
   │                                                              │
   │                                                              │  contextual query from risk profile
   │                                                              │  ChromaDB cosine retrieval (top 4)
   │                                                              │  all-MiniLM-L6-v2 embeddings
   │                                                              ▼
   └─────────────────────────────────────────────────────► intervention
                                                                  │
                                                                  │  Claude claude-sonnet-4-6
                                                                  │  all agent recs + RAG context
                                                                  │  → structured JSON action plan
                                                                  ▼
                                                              PostgreSQL
                                                         pipeline_runs + action_records
```

Falls back to sequential execution when LangGraph is not installed, with identical behavior.

### PipelineState fields

```python
session_id          str             # UUID per run
admission           AdmissionData   # parsed HL7 event
esi_level           str             # ESI-1 through ESI-4
triage_route        str             # ICU / Inpatient / ED / Outpatient
readmission_30d     float           # XGBoost probability
deterioration       float
sepsis_risk         float
discharge_today     float
discharge_tomorrow  float
los_predicted_days  int
acuity_tier         str             # CRITICAL / HIGH / MEDIUM / LOW
risk_factors        list[dict]      # top cross-model SHAP factors
shap_by_model       dict            # per-model top-5 SHAP values
agent_recommendations list[dict]   # all agent outputs concatenated
intervention_plan   dict            # Claude's structured action plan
rag_context         str             # retrieved guideline text
rag_sources         list[str]
audit_actions       list[str]       # node completion log
pipeline_duration_ms int
```

---

## ML Risk Models

Five XGBoost binary classifiers trained on 5,000 synthetic encounters with feature-label correlations calibrated to published benchmark rates. Training uses 80/20 split, balanced class weights, and AUCPR early stopping.

### Validation performance

| Model | AUC | AUPRC |
|---|---|---|
| Readmission 30d | **0.9748** | **0.9549** |
| Deterioration | **0.9988** | **0.9977** |
| Sepsis risk | **1.0000** | **1.0000** |
| Discharge today | **0.9413** | **0.9910** |
| Discharge tomorrow | **0.9983** | **1.0000** |

Artifacts stored as `.ubj` (XGBoost binary format) in `data/models/`. `RiskScoringEngine` loads them once at startup and holds SHAP `TreeExplainer` instances per model.

### 18 Input Features

| Feature | Description |
|---|---|
| `age` | Patient age |
| `sex_encoded` | 0=F, 1=M, 2=Other |
| `patient_class_encoded` | 0=Outpatient, 1=Inpatient, 2=ED |
| `prior_admissions_12m` | Prior admissions in last 12 months |
| `prior_ed_visits_12m` | Prior ED visits in last 12 months |
| `prior_no_shows_12m` | Prior no-show count |
| `active_problem_count` | Active diagnoses on problem list |
| `medication_count` | Active medication count |
| `days_since_last_visit` | Recency of care |
| `appt_lead_time_days` | Lead time before admission |
| `appt_hour` | Hour of admission (0–23) |
| `appt_day_of_week` | Day of week (0=Monday) |
| `is_new_patient` | New patient flag |
| `payer_type_encoded` | 0=Commercial, 1=Medicare, 2=Medicaid, 3=Self-pay |
| `distance_miles` | Distance from facility |
| `language_barrier` | 1 if non-English primary language |
| `charlson_index` | Charlson Comorbidity Index (0–10) |
| `num_chronic_conditions` | Count of chronic conditions |

In a production deployment, these features map to the Cerner Millennium HealtheAnalytics schema. The feature pipeline is designed for both the synthetic demo path and a Snowflake query path.

### SHAP Explainability

Every prediction generates a per-model SHAP value array via `shap.TreeExplainer`. Top 5 features by absolute SHAP value are stored in `shap_by_model` per model, surfaced as interactive waterfall charts in the patient detail modal. Each bar shows whether the feature increased or decreased that specific prediction, with the clinical display name rather than the raw feature identifier.

---

## Clinical Knowledge Retrieval (RAG)

ChromaDB runs in embedded persistent mode — no separate service. The collection is seeded at startup with 10 clinical reference documents and queried per-pipeline-run using contextually-built queries.

### Knowledge Base

| Source | Content |
|---|---|
| CMS HRRP | 30-day all-cause readmission thresholds, high-risk indicators, care management interventions |
| CMS SEP-1 | Severe Sepsis Bundle — 3-hour and 6-hour compliance windows, lactate/culture/antibiotic/vasopressor criteria |
| HEDIS PCR | Plan All-Cause Readmission measure — benchmark rates, risk stratification criteria |
| Joint Commission NPSG 2024 | Patient identifier requirements, critical result reporting, alarm management, suicide risk screening |
| CMS Language Access | Title VI LEP requirements, language barrier adverse event rates, interpreter trigger criteria |
| SDOH Framework | Transportation barrier statistics, distance/payer/no-show screening criteria, MTM/Lyft Health interventions |
| Charlson Comorbidity Index | 10-year mortality predictions by CCI score, high-readmission-risk threshold (CCI ≥3) |
| CMS CoP 482.13 | Discharge planning elements, BOOST toolkit, 72-hour post-discharge call requirements |
| AGS Geriatric Guidelines | HOSPITAL Score for age ≥65, delirium/fall/functional decline risk criteria |
| ACEP ESI Guidelines | 5-level triage criteria, door-to-physician targets, deterioration score thresholds |

### Retrieval Logic

The RAG node builds a contextual query from the patient's live risk profile rather than sending a generic request:

```python
query_parts = [f"patient acuity {acuity}"]
if sepsis > 0.4:   query_parts.append("sepsis SEP-1 bundle")
if readmit > 0.4:  query_parts.append("readmission risk discharge planning")
if age >= 65:      query_parts.append("geriatric elderly readmission")
```

Embeddings use `all-MiniLM-L6-v2` (sentence-transformers). A 256-entry LRU cache avoids re-encoding repeated queries. Cosine similarity (`hnsw:space: cosine`) returns top-4 documents with relevance scores, formatted as a labeled prompt block passed to Claude.

---

## OR Board

The surgical operations view renders a live grid of all OR rooms with their case schedules. Clicking any case opens a 5-tab modal populated with deterministic synthetic data derived from the case ID via a reproducible hash function — the same case always shows the same enriched data.

### Tabs

| Tab | Content |
|---|---|
| **Overview** | Timeline bar, case status, scheduled vs actual start, estimated end, primary diagnosis, estimated blood loss |
| **Patient** | Demographics, ASA class, BMI, blood type, allergies, comorbidities, current medications, IV access, monitoring |
| **Surgical Team** | Attending surgeon, resident, scrub tech, circulating RN, anesthesiologist — with specialty and credential details |
| **Anesthesia** | Anesthesia type (General ETT / Spinal-Regional / MAC), patient positioning, airway assessment, monitoring plan |
| **Pre-op Checklist** | Consent, NPO status, H&P, labs, imaging, blood products, anesthesia pre-eval, timeout completion — with color-coded status |

The OR Board renders identically on both the sidebar panel (Census page) and the full OR Board tab, both wired to `openOrCaseModal()`.

---

## Role-Based Access Control

Seven roles, each receiving a tailored view of the dashboard. Role is persisted to `localStorage` and applied via `applyRole()` which shows/hides views and adjusts available actions.

| Role | Primary View | Key Capability |
|---|---|---|
| **Physician** | Patient detail + AI recommendations | Accept / Override / Escalate agent recommendations |
| **Nurse** | Census + alert feed | Triage actions, medication safety flags |
| **Case Manager** | Discharge planning | Barrier resolution, SNF/home health coordination |
| **Pharmacist** | Medication safety panel | Drug interaction review, allergy conflict resolution |
| **OR Coordinator** | OR Board full view | Case scheduling, room status, team assignments |
| **Executive** | Operations metrics | Capacity utilization, throughput, quality indicators |
| **Read-Only** | Full dashboard | View-only, no action capabilities |

---

## Action Accountability

Every agent recommendation has three response paths: **Accept**, **Override**, or **Escalate**. Each decision writes an immutable record to `action_records`:

```sql
action_records
  id            UUID PRIMARY KEY
  run_id        UUID → pipeline_runs.id
  event_id      VARCHAR(64)
  agent_id      VARCHAR(64)
  decision      VARCHAR(32)   -- accepted | overridden | escalated
  actor         VARCHAR(128)
  actor_role    VARCHAR(128)
  note          TEXT
  timestamp     DATETIME
```

This answers questions traditional EHRs cannot answer: Did anyone review the sepsis alert? Who overrode the discharge recommendation, and what was their reason? How many Bed Management recommendations were escalated this week? The Accountability view surfaces the full log.

The complete pipeline run is stored in `pipeline_runs` alongside every decision record:

```sql
pipeline_runs
  session_id           VARCHAR(64) UNIQUE
  mrn / encounter_id   patient identifiers
  acuity_tier          CRITICAL | HIGH | MEDIUM | LOW
  readmission_30d      FLOAT     -- XGBoost probability
  deterioration        FLOAT
  sepsis_risk          FLOAT
  discharge_today      FLOAT
  los_predicted_days   INT
  triage_output        JSON      -- full agent rec
  sepsis_output        JSON
  discharge_output     JSON
  bed_output           JSON
  medsafety_output     JSON
  intervention_plan    JSON      -- Claude's action plan
  agent_recommendations JSON[]
  shap_by_model        JSON      -- per-model SHAP values
  rag_sources          JSON[]
  pipeline_duration_ms INT
  pipeline_success     BOOLEAN
```

---

## Tech Stack

### Backend

| Component | Library | Version |
|---|---|---|
| API framework | FastAPI | 0.111 |
| ASGI server | uvicorn | 0.29 |
| Data validation | Pydantic v2 | 2.7.1 |
| Pipeline orchestration | LangGraph | 0.2 |
| ML models | XGBoost | ≥2.0 |
| Explainability | SHAP | ≥0.45 |
| Feature engineering | scikit-learn, NumPy | 1.4, 1.26 |
| Vector store | ChromaDB | ≥0.5 (embedded) |
| Embeddings | sentence-transformers | ≥2.7 |
| LLM | Anthropic SDK | ≥0.30 |
| Database ORM | SQLAlchemy async | ≥2.0 |
| DB driver | asyncpg | ≥0.29 |
| Logging | structlog | 24.1 |

### Frontend

| Component | Approach |
|---|---|
| Framework | Vanilla JS — no build step, single HTML file |
| Charts | Chart.js (SHAP waterfall, trend sparklines) |
| Real-time updates | `EventSource` (SSE) |
| Theming | CSS custom properties, `data-theme="dark"` attribute |
| State | `_recStore` global dict for safe onclick handlers in dynamic HTML |
| Synthetic data | `_hashCode(caseId + seed)` — deterministic per-case enrichment |

### Infrastructure

| Component | Service |
|---|---|
| Hosting | Railway |
| Container | Docker (python:3.12-slim) |
| Database | PostgreSQL 16 (Railway plugin, optional) |
| CI/CD | GitHub → Railway auto-deploy |
| Health check | `GET /api/v1/health` (30s interval) |

---

## Project Structure

```
hospital-ai-command-center/
├── src/
│   ├── main.py                    # FastAPI app factory, lifespan startup
│   ├── config.py                  # Pydantic Settings (env vars)
│   ├── api/
│   │   └── command_center.py      # All API routes + SSE stream
│   ├── agents/
│   │   ├── pipeline.py            # LangGraph StateGraph builder + ClinicalPipeline
│   │   ├── nodes.py               # 8 agent node implementations
│   │   └── state.py               # PipelineState TypedDict
│   ├── ml/
│   │   └── risk_model.py          # RiskScoringEngine (XGBoost + SHAP)
│   ├── rag/
│   │   └── retriever.py           # ClinicalRAG (ChromaDB + sentence-transformers)
│   ├── ingestion/
│   │   └── hl7_parser.py          # HL7 v2.x ADT parser (Cerner Millennium)
│   └── db/
│       ├── models.py              # SQLAlchemy ORM (pipeline_runs, action_records)
│       └── session.py             # Async session factory
├── static/
│   └── dashboard.html             # Full SPA (~3,750 lines)
├── scripts/
│   ├── generate_data.py           # Synthetic census, OR schedule, ADT events
│   └── train_models.py            # XGBoost training pipeline
├── data/
│   ├── models/
│   │   ├── readmission.ubj        # XGBoost binary artifacts
│   │   ├── deterioration.ubj
│   │   ├── sepsis.ubj
│   │   ├── discharge_today.ubj
│   │   ├── discharge_tomorrow.ubj
│   │   └── metrics.json           # Validation AUC/AUPRC per model
│   ├── chroma/                    # Embedded ChromaDB (HNSW index + SQLite)
│   ├── census.json                # Synthetic hospital census
│   ├── or_schedule.json           # Synthetic OR schedule
│   ├── adt_events.json            # Synthetic ADT event queue
│   └── operational_metrics.json   # Ops dashboard data
├── Dockerfile
└── requirements.txt
```

---

## Local Setup

```bash
git clone https://github.com/Roshandharan/hospital-ai-command-center.git
cd hospital-ai-command-center

# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# Install core dependencies (always required)
pip install fastapi uvicorn pydantic pydantic-settings httpx structlog numpy

# Install ML stack (required for real XGBoost scoring; demo heuristics work without it)
pip install xgboost shap scikit-learn pandas

# Install RAG stack (ChromaDB + embeddings; keyword fallback works without it)
pip install chromadb sentence-transformers

# Install LangGraph (sequential fallback works without it)
pip install langgraph

# Install Anthropic SDK (deterministic fallback works without it)
pip install anthropic

# Generate synthetic hospital data
python scripts/generate_data.py

# Train XGBoost models (~30 seconds)
python scripts/train_models.py

# Start the server
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000/dashboard.

The application starts cleanly without PostgreSQL, an Anthropic API key, or pre-trained model artifacts. Every component has an explicit fallback path.

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | No | `""` | Enables Claude LLM synthesis; deterministic fallback if missing |
| `DATABASE_URL` | No | — | PostgreSQL connection string (`postgresql+asyncpg://...`); in-memory if missing |
| `ANTHROPIC_MODEL` | No | `claude-sonnet-4-6` | Claude model ID |
| `CHROMA_PERSIST_PATH` | No | `./data/chroma` | ChromaDB persistence directory |
| `MODEL_ARTIFACTS_DIR` | No | `./data/models` | Directory containing `.ubj` model files |
| `FEED_INTERVAL_SECONDS` | No | `12.0` | ADT feed simulation interval |
| `ENVIRONMENT` | No | `development` | `development` or `production` |
| `DEBUG` | No | `false` | Enable FastAPI debug mode |

Create a `.env` file in the project root. All variables are optional — the system starts and operates fully without any of them.

---

## Docker

```bash
# Build
docker build -t hospital-ai-command-center .

# Run without external services
docker run -p 8000:8000 hospital-ai-command-center

# Run with Anthropic and PostgreSQL
docker run -p 8000:8000 \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  -e DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/db \
  hospital-ai-command-center
```

The Dockerfile uses `python:3.12-slim` and installs ML, RAG, and LangGraph packages with `|| echo "optional"` fallbacks so the image builds even if optional packages fail.

Health check: `GET /api/v1/health` — 30-second interval, 40-second start period.

---

## Railway Deployment

1. Connect GitHub repo to a new Railway project
2. Railway detects the Dockerfile and builds automatically
3. Add the **PostgreSQL** plugin from the Railway dashboard (auto-injects `DATABASE_URL`)
4. Set `ANTHROPIC_API_KEY` in **Variables**
5. Generate a public domain under **Settings → Networking → port 8000**

First cold start takes 60–90 seconds: sentence-transformer model downloads (~90 MB), ChromaDB seeds the knowledge base, XGBoost artifacts load, LangGraph pipeline initializes.

```bash
# Deploy via Railway CLI
railway up --service hospital-ai-command-center

# If 502 after deploy, re-bind the domain to port 8080
railway domain update <domain-id> --port 8080 --service hospital-ai-command-center
```

---

## What's Real vs Synthetic

### Production-quality implementations

- **LangGraph pipeline** — real `StateGraph` with dependency injection, `ainvoke`, sequential fallback
- **XGBoost models** — trained artifacts stored as `.ubj`, loaded via `XGBClassifier.load_model()`, SHAP explainers attached
- **ChromaDB RAG** — real embedded vector store with HNSW index, real cosine retrieval, real `all-MiniLM-L6-v2` embeddings
- **FastAPI + SSE** — real `EventSourceResponse`, real async streaming, real structlog JSON logging
- **HL7 v2.x parser** — handles Cerner Millennium message structure with correct MSH field indexing, PID CX identifier routing (MR/AN types), PV1.19 visit number extraction, DG1 ICD-10 parsing
- **PostgreSQL schema** — production-grade schema with UUID primary keys, indexed foreign keys, JSON columns for agent outputs, immutable audit trail
- **SQLAlchemy 2.0 async** — `AsyncSession`, `mapped_column`, `Mapped` type annotations, proper connection pooling

### Synthetic

- **Patient data** — census, vitals, lab results, medication lists, OR schedules, and ADT events are generated procedurally by `scripts/generate_data.py`
- **Risk model training data** — 5,000 synthetic encounters with calibrated feature-label correlations; not derived from real patient records
- **OR case enrichment** — anesthesia type, ASA class, team assignments, pre-op checklist items derived deterministically from `_hashCode(caseId + seed)`

In a production deployment: the feature pipeline queries Snowflake directly, the HL7 parser processes live Cerner ADT feeds, and models are retrained quarterly on real encounter data. Both integration paths are implemented and documented in the codebase.

---

## Production Readiness

### Implemented, needs integration

- **HL7 live feed** — parser handles real Cerner message structure; needs socket/MLLP listener wired to hospital interface engine
- **Snowflake feature pipeline** — feature schema maps to Cerner Millennium HealtheAnalytics; needs connection credentials
- **Model retraining** — `train_models.py` pipeline is parameterized; needs scheduler + real encounter data source

### Not yet implemented

- Staff authentication against Active Directory (action records currently use role-switch simulation)
- Mobile push notifications for STAT recommendations
- Model drift monitoring (prediction distribution tracking over time)
- FHIR R4 adapter layer (HL7 v2.x parser complete; FHIR R4 resource mapping not yet written)
- Conversational census query agent (ad-hoc natural language operational queries)

### Scalability notes

The API uses a single uvicorn worker and in-memory state for the live feed and accountability log. For a production multi-instance deployment: move the ADT event queue to Redis Streams, replace in-memory action log with PostgreSQL reads, and run uvicorn with multiple workers behind a load balancer.

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/dashboard` | GET | Serve the SPA dashboard |
| `/api/v1/health` | GET | Health check (used by Railway + Docker) |
| `/api/v1/stream` | GET | SSE stream — one event per pipeline run |
| `/api/v1/adt/ingest` | POST | Ingest an HL7 ADT event, run pipeline, return full state |
| `/api/v1/census` | GET | Current synthetic hospital census |
| `/api/v1/or-schedule` | GET | Current OR schedule |
| `/api/v1/operational-metrics` | GET | Ops dashboard metrics |
| `/api/v1/actions` | POST | Log an Accept/Override/Escalate decision |
| `/api/v1/actions` | GET | Retrieve accountability log |

---

## Research Foundation

Every component of this system is grounded in peer-reviewed literature. The sections below map published research directly to the design decisions and implementation choices made in this codebase.

---

### Gradient Boosting and XGBoost for Clinical Risk Prediction

The five XGBoost models at the core of this system — 30-day readmission, in-stay deterioration, sepsis risk, discharge today, and discharge tomorrow — are built on the most widely adopted framework in tabular clinical ML.

**Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.** *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.* [DOI: 10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)

> The foundational paper introducing XGBoost — the gradient-boosted decision tree system that now dominates clinical risk prediction benchmarks. The paper's sparse-aware split-finding algorithm and weighted quantile sketch for approximate learning are the reason XGBoost outperforms deep learning on most structured EHR tabular tasks. Every `.ubj` model artifact in `data/models/` was produced by XGBoost's binary serializer.

**Zhang, Y., Xiang, T., et al. (2024). Explainable machine learning for predicting 30-day readmission in acute heart failure patients.** *iScience (Cell Press).* [DOI: 10.1016/j.isci.2024.110281](https://doi.org/10.1016/j.isci.2024.110281) | PubMed: 39040074

> Developed an XGBoost readmission model on 2,232 heart failure patients (AUC 0.763), combining it with SHAP waterfall charts to surface per-patient risk explanations. This is almost exactly the architecture of the `RiskScoringEngine` + dashboard waterfall chart combination in this system.

**Bates, D.W., Saria, S., Ohno-Machado, L., Shah, A., & Escobar, G. (2014). Big data in health care: using analytics to identify and manage high-risk and high-cost patients.** *Health Affairs, 33(7):1123–1131.* [DOI: 10.1377/hlthaff.2014.0041](https://doi.org/10.1377/hlthaff.2014.0041)

> The landmark Health Affairs paper that articulated the six use cases for clinical analytics — readmission reduction, high-cost patient identification, triage optimization, disease management, adverse event prevention, and operational efficiency — and provided the evidence base for investing in real-time predictive systems. This paper defined the problem space this system addresses.

**Systematic review, 2024. Machine learning–based 30-day readmission prediction models for patients with heart failure.** *European Journal of Cardiovascular Nursing.* [DOI: 10.1093/eurjcn/zvae031](https://doi.org/10.1093/eurjcn/zvae031)

> A systematic review comparing ML approaches for 30-day readmission prediction across dozens of studies, identifying gradient boosting as consistently top-performing. The 18 input features selected for `PatientFeatures` — prior admissions, Charlson index, payer type, language barrier, distance from facility — map directly to features identified as high-importance in this review.

---

### Sepsis Detection and SEP-1 Bundle Compliance

The `SepsisAgent` node fires when `sepsis_risk > 0.25`, checks SEP-1 bundle completion, and generates a time-sensitive 6-hour recommendation. This design is grounded in the clinical AI sepsis literature:

**Komorowski, M., Celi, L.A., Badawi, O., Gordon, A.C., & Faisal, A.A. (2018). The Artificial Intelligence Clinician learns optimal treatment strategies for sepsis in intensive care.** *Nature Medicine, 24:1716–1720.* [DOI: 10.1038/s41591-018-0213-5](https://doi.org/10.1038/s41591-018-0213-5) | PubMed: 30349085

> A reinforcement learning agent trained on 100,000+ ICU patients learned optimal fluid and vasopressor dosing for sepsis, finding that patients whose clinicians matched the AI's dosing had the lowest mortality. This paper established that sepsis management is a domain where AI can produce demonstrably better decision strategies than average clinical practice.

**Reyna, M.A., Josef, C.S., Jeter, R., et al. (2020). Early Prediction of Sepsis From Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019.** *Critical Care Medicine, 48(2):210–217.* [DOI: 10.1097/CCM.0000000000004145](https://doi.org/10.1097/CCM.0000000000004145) | PubMed: 31850926

> Defined the PhysioNet 2019 Sepsis Challenge benchmark across 40,000+ ICU patients, establishing that ML models can predict sepsis six hours before clinical onset with meaningful utility. The utility score formulation — rewarding early true positives and penalizing late or false positives — directly informed the 6-hour window logic in the `SepsisAgent` recommendation structure.

**Sendak, M.P., Ratliff, W., Sarro, D., et al. (2020). Real-World Integration of a Sepsis Deep Learning Technology Into Routine Clinical Care: Implementation Study.** *JMIR Medical Informatics, 8(7):e15182.* [DOI: 10.2196/15182](https://doi.org/10.2196/15182) | PMC: PMC7391165

> The implementation study of Duke University Hospital's "Sepsis Watch" — a deep learning platform integrated into the live EHR. The key findings on team design (dedicated rapid response nurses, not physician alerting), alert routing, and clinician adoption patterns shaped the `assign_to: "Rapid Response Team"` routing in the `SepsisAgent` and the accountability log design.

**Nemati, S., Holder, A., Razmi, F., et al. (2018). An Interpretable Machine Learning Model for Accurate Prediction of Sepsis in the ICU.** *Critical Care Medicine, 46(4):547–553.* [DOI: 10.1097/CCM.0000000000002936](https://doi.org/10.1097/CCM.0000000000002936)

> Used a modified Weibull-Cox model with 65 EHR features to predict sepsis while maintaining interpretability — one of the first papers to explicitly argue that sepsis AI must be interpretable to be trusted clinically. The SHAP waterfall charts in this system's patient modal directly address this interpretability requirement.

**Systematic review, 2021. Early Prediction of Sepsis in the ICU Using Machine Learning.** *Frontiers in Medicine.* [DOI: 10.3389/fmed.2021.607952](https://doi.org/10.3389/fmed.2021.607952) | PMC: PMC8193357

> Reviewed 38 sepsis ML studies, finding AUROC values between 0.68–0.99 and identifying vital signs and lab values as the most predictive features. The sepsis model in this system achieves AUC 1.00 and AUPRC 1.00 on synthetic data (by design), consistent with the upper end of the literature range on carefully curated training sets.

---

### SHAP Explainability in Clinical AI

Every risk score in this system comes with a SHAP waterfall chart. The decision to use TreeSHAP rather than simpler feature importance methods is directly motivated by the following foundational literature:

**Lundberg, S.M., & Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions.** *Advances in Neural Information Processing Systems (NeurIPS 2017), 30:4765–4774.* [PDF](https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf)

> The foundational SHAP paper introducing SHapley Additive exPlanations, grounding feature attribution in Shapley values from cooperative game theory. This guarantees consistency, local accuracy, and missingness properties that simpler attribution methods (LIME, permutation importance) do not satisfy. Every `shap.TreeExplainer` call in `src/ml/risk_model.py` implements this framework.

**Lundberg, S.M., Nair, B., Vavilala, M.S., et al. (2018). Explainable machine-learning predictions for the prevention of hypoxaemia during surgery.** *Nature Biomedical Engineering, 2:749–760.* [DOI: 10.1038/s41551-018-0304-0](https://doi.org/10.1038/s41551-018-0304-0)

> Applied real-time SHAP explanations to a perioperative AI system trained on 50,000+ surgeries, showing that per-patient feature attribution at the point of care changes clinician behavior. This is the closest analogue in the literature to the SHAP waterfall charts displayed in this system's patient detail modal — real-time, per-prediction, pointing to actionable clinical factors.

**Lundberg, S.M., Erion, G.G., Chen, H., et al. (2020). From local explanations to global understanding with explainable AI for trees.** *Nature Machine Intelligence, 2:56–67.* [DOI: 10.1038/s42256-019-0138-9](https://doi.org/10.1038/s42256-019-0138-9)

> Introduced TreeSHAP, an algorithm that computes exact SHAP values for tree ensembles in polynomial time rather than exponential time. The `shap.TreeExplainer` used in `RiskScoringEngine._score_real()` is the TreeSHAP implementation. This paper also showed how SHAP values enable global model understanding — not just individual predictions — through summary and dependence plots.

**Tonekaboni, S., Joshi, S., McCradden, M.D., & Goldenberg, A. (2019). What Clinicians Want: Contextualizing Explainable Machine Learning for Clinical End Use.** *Proceedings of the 4th Machine Learning for Healthcare Conference (MLHC), PMLR 106:359–380.* [URL](https://proceedings.mlr.press/v106/tonekaboni19a.html)

> Surveyed ICU and ED clinicians on what types of ML explanations they actually find useful in practice. Key finding: clinicians prefer contextual explanations aligned to their current decision context, with uncertainty quantification, over raw ranked feature lists. The `display_name` mapping in `_DISPLAY_NAMES` (e.g., `"charlson_index"` → `"Comorbidity burden (CCI)"`) directly addresses this — the waterfall charts show clinically meaningful labels, not Python identifiers.

---

### Multi-Agent LLM Systems in Healthcare

The LangGraph `StateGraph` with eight specialized agent nodes — each with a distinct clinical domain, confidence score, and assignee — is an implementation of the agentic AI architecture increasingly studied for clinical decision support:

**Kim, Y., Park, C., Jeong, H., et al. (2024). MDAgents: An Adaptive Collaboration of LLMs for Medical Decision-Making.** *NeurIPS 2024 (Oral).* [arXiv: 2404.15155](https://arxiv.org/abs/2404.15155) | [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/file/90d1fc07f46e31387978b88e7e057a31-Paper-Conference.pdf)

> Introduced a framework that dynamically assigns solo or multi-agent LLM collaboration structures based on medical task complexity, improving performance on MedQA, MedMCQA, and PubMedQA by mimicking real-world clinical team decision-making. The insight — that complex clinical cases benefit from multiple specialized agents synthesizing their outputs rather than a single generalist model — is the architectural motivation for this system's 8-node LangGraph pipeline.

**Survey, 2025. A Survey of LLM-based Agents in Medicine: How far are we from Baymax?** *arXiv: 2502.11211.* [PDF](https://arxiv.org/pdf/2502.11211)

> A comprehensive survey of LLM-based medical agents covering architectures, planning, memory, tool use, and clinical deployment pathways. Particularly relevant to this system: the survey identifies the RAG + LLM pattern as the dominant approach for grounding clinical agents in authoritative guideline knowledge, and notes that multi-step agent pipelines outperform single-step prompting on complex clinical reasoning tasks.

**Multiple authors, 2025. ClinicalAgents: Multi-Agent Orchestration for Clinical Decision Making with Dual-Memory.** *arXiv: 2603.26182.* [PDF](https://arxiv.org/pdf/2603.26182)

> Proposes a multi-agent orchestration system with dual memory (short-term patient context + long-term medical knowledge) for clinical decision support. The architecture maps closely to this system's design: the `PipelineState` TypedDict serves as the shared short-term patient context passed through each agent node, while ChromaDB serves as the long-term medical knowledge store.

---

### Retrieval-Augmented Generation for Clinical Guidelines

The `ClinicalRAG` class uses ChromaDB with `all-MiniLM-L6-v2` embeddings to retrieve relevant clinical guidelines per patient, which the `InterventionAgent` uses to ground Claude's synthesis in authoritative sources. This architecture is directly motivated by the RAG literature:

**Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.** *Advances in Neural Information Processing Systems (NeurIPS 2020), 33:9459–9474.* [arXiv: 2005.11401](https://arxiv.org/abs/2005.11401)

> The foundational RAG paper introducing the paradigm of combining dense retrieval (a learned embedding index) with a generative LLM. The key insight: LLMs have fixed parametric knowledge that degrades on domain-specific or time-sensitive content; retrieval provides non-parametric, updatable context. Every call to `rag.retrieve()` in this system implements the RAG retrieval step described here.

**Kresevic, S., Giuffrè, M., Ajčević, M., et al. (2024). Optimization of hepatological clinical guidelines interpretation by large language models: a retrieval augmented generation-based framework.** *npj Digital Medicine, 7:102.* [DOI: 10.1038/s41746-024-01091-y](https://doi.org/10.1038/s41746-024-01091-y) | PMC: PMC11039454

> Applied GPT-4 Turbo with RAG to interpret Hepatitis C clinical guidelines. RAG-Top10 achieved 91.7% accuracy on open-ended guideline questions versus 36.6% for the baseline model, and correct prescribing in 76% versus 24% of cases. This is direct evidence for the mechanism this system relies on: grounding Claude's intervention synthesis in retrieved CMS/HEDIS/NPSG guidelines substantially improves the clinical accuracy of the output.

**Systematic review and meta-analysis, 2024. Improving large language model applications in biomedicine with retrieval-augmented generation.** PMC: PMC12005634

> Meta-analysis of 20 RAG-in-biomedicine studies (from 335 screened) found a pooled effect size of 1.35 (95% CI: 1.19–1.53), confirming that RAG statistically and significantly improves LLM accuracy over non-RAG baselines across clinical question answering tasks. Provides quantitative justification for the RAG architecture in this system.

**Wornow, M., Xu, Y., Thapa, R., et al. (2023). The shaky foundations of large language models and foundation models for electronic health records.** *npj Digital Medicine, 6:135.* [DOI: 10.1038/s41746-023-00879-8](https://doi.org/10.1038/s41746-023-00879-8) | PubMed: 37516790

> Reviewed 80+ EHR foundation models and found that most are trained on narrow datasets and evaluated on tasks irrelevant to real health systems. A critical benchmarking paper that motivates the design choice in this system to anchor Claude's outputs to retrieved, source-identified guidelines rather than relying on parametric LLM knowledge alone — and to display RAG sources (`rag_sources`) alongside every intervention plan so clinicians can verify the evidence chain.

---

### Large Language Models in Clinical Settings

The `InterventionAgent` calls Claude claude-sonnet-4-6 to synthesize all upstream agent outputs into a structured JSON action plan. The decision to use a frontier LLM for synthesis rather than a domain-fine-tuned model reflects the current state of clinical LLM research:

**Singhal, K., Azizi, S., Tu, T., et al. (2023). Large language models encode clinical knowledge.** *Nature, 620(7972):172–180.* [DOI: 10.1038/s41586-023-06291-2](https://doi.org/10.1038/s41586-023-06291-2)

> Introduced the MultiMedQA benchmark and Med-PaLM — the first LLM to pass USMLE-style questions at a passing threshold. The key finding: frontier LLMs trained on general text have internalized substantial clinical knowledge sufficient to reason about patient scenarios. This provides the foundation for using Claude as a synthesis layer without clinical fine-tuning, relying on RAG-retrieved guidelines to supply the specific guideline context.

**Nori, H., King, N., McKinney, S.M., Carignan, D., & Horvitz, E. (2023). Capabilities of GPT-4 on Medical Challenge Problems.** *arXiv: 2303.13375.* [URL](https://arxiv.org/abs/2303.13375)

> Showed that GPT-4 achieves passing scores on USMLE Steps 1–3 without any medical fine-tuning, outperforming previously specialized medical AI models. This result — that general frontier LLMs match or exceed specialized models on structured clinical reasoning — supports the system's use of `claude-sonnet-4-6` as a generalist synthesis layer rather than a hospital-specific fine-tuned model.

**Multiple authors, 2024. The potential of GPT-4 to analyse medical notes in three different languages: a retrospective model-evaluation study.** *The Lancet Digital Health.* [DOI: 10.1016/S2589-7500(24)00246-2](https://doi.org/10.1016/S2589-7500(24)00246-2)

> Evaluated GPT-4's ability to extract structured clinical information from free-text medical notes across three languages at eight university hospitals. High accuracy demonstrated. The system's `intervention_node` uses a structured prompt requiring JSON output — `{"priority_actions": [...], "care_pathway": "...", ...}` — and validates the response structure, directly applying this finding to extract reliable structured plans from Claude.

**Davenport, T.H., & Kalakota, R. (2019). The potential for artificial intelligence in healthcare.** *Future Healthcare Journal, 6(2):94–98.* [DOI: 10.7861/futurehosp.6-2-94](https://doi.org/10.7861/futurehosp.6-2-94) | PubMed: 31363513

> A widely cited overview paper covering NLP, robotics, and ML applications across clinical care, establishing the framework that AI in healthcare divides into physical automation (surgical robots), diagnostic AI (imaging, pathology), and administrative/workflow AI (risk stratification, discharge planning). This system occupies the third category — workflow AI that augments clinical decision-making rather than replacing it.

---

### Real-Time Clinical Decision Support Systems

**Topol, E.J. (2019). High-performance medicine: the convergence of human and artificial intelligence.** *Nature Medicine, 25(1):44–56.* [DOI: 10.1038/s41591-018-0300-7](https://doi.org/10.1038/s41591-018-0300-7) | PubMed: 30617339

> The most-cited synthesis paper on AI in clinical medicine, reviewing how deep learning is enabling real-time decision support. Topol's central argument — that AI will "liberate" clinicians from routine cognitive tasks, allowing them to focus on the relational and complex aspects of care — is the design philosophy behind this system. The AI handles initial triage, risk stratification, guideline retrieval, and draft action plan; the clinician retains Accept/Override/Escalate authority.

**Rajkomar, A., Oren, E., Chen, K., et al. (2018). Scalable and accurate deep learning with electronic health records.** *npj Digital Medicine, 1:18.* [DOI: 10.1038/s41746-018-0029-1](https://doi.org/10.1038/s41746-018-0029-1) | PubMed: 31304302

> Google's landmark study demonstrating that a single deep learning model trained on raw FHIR-formatted EHR data could predict in-hospital mortality, 30-day readmission, prolonged LOS, and discharge diagnoses simultaneously. The multi-output prediction approach — training separate models per outcome rather than one multi-task model — reflects a design choice made in this system for interpretability and modularity, consistent with the findings that task-specific models are easier to explain and audit than multi-task architectures.

**Multiple authors, 2024. Toward a responsible future: recommendations for AI-enabled clinical decision support.** *Journal of the American Medical Informatics Association (JAMIA), 31(11):2730.* [DOI: 10.1093/jamia/ocae207](https://doi.org/10.1093/jamia/ocae207)

> JAMIA policy paper providing actionable governance recommendations for AI-enabled CDSS: mandatory performance monitoring, clinician override logging, bias auditing, and explainability requirements. The `action_records` PostgreSQL table and Accountability view in this system directly implement the override logging requirement. The SHAP waterfall charts directly implement the explainability requirement.

**Ash, J.S., Berg, M., & Coiera, E. (2004). Some unintended consequences of information technology in health care: the nature of patient care information system-related errors.** *Journal of the American Medical Informatics Association (JAMIA), 11(2):104–112.* PubMed: 14633935

> The seminal paper on unintended consequences of clinical IT systems — specifically, how CDSS alerts generate new classes of errors through alert fatigue, workarounds, and automation bias. This paper is the reason the system is designed to limit alert volume (agents fire conditionally based on risk thresholds, not on every event), route to specific named assignees (not broadcast), and require explicit Accept/Override/Escalate responses (not passive acknowledgment).

---

### Patient Deterioration Prediction and Early Warning Systems

The `deterioration` XGBoost model is one of the five core risk models. Its clinical motivation and design draw from the early warning score literature:

**Escobar, G.J., Liu, V.X., Schuler, A., et al. (2020). Automated Identification of Adults at Risk for In-Hospital Clinical Deterioration.** *New England Journal of Medicine, 383(20):1951–1960.* [DOI: 10.1056/NEJMsa2001090](https://doi.org/10.1056/NEJMsa2001090) | PubMed: 33176085

> A landmark NEJM study from Kaiser Permanente showing that automated EHR-based deterioration detection with dedicated remote nurse monitoring reduced in-hospital mortality across a large integrated health system. This is the closest published analogue to the deterioration model + notification system in this codebase — Kaiser's "Advanced Alert Monitor" architecture (automated scoring → nurse triage → physician escalation) maps directly to this system's pipeline (XGBoost scoring → Bed/Triage agents → intervention plan with named assignees).

**Multiple authors, 2020. MEWS++: Enhancing the Prediction of Clinical Deterioration in Admitted Patients through a Machine Learning Model.** PMC: PMC7073544

> Compared an ML model to traditional MEWS, finding the ML approach improved sensitivity 37%, specificity 11%, and AUC-ROC 14%, with ability to predict deterioration or death 6 hours prior to the event. The 18-feature `PatientFeatures` input vector — covering active problem count, medication burden, Charlson index, and care setting — is designed to capture the same signal domains that MEWS++ identified as most predictive.

**Escobar, G.J., LaGuardia, J.C., Turk, B.J., et al. (2012). Early detection of impending physiologic deterioration among patients who are not in intensive care.** *Journal of Hospital Medicine, 7(5):388–395.* [DOI: 10.1002/jhm.1929](https://doi.org/10.1002/jhm.1929) | PubMed: 22447632

> The foundational Kaiser Permanente study developing the first EMR-based algorithm for detecting impending floor patient deterioration — predecessor to the NEJM 2020 study. Established that structured EHR data (labs, vitals, nursing flowsheet entries) processed by a statistical model can predict deterioration hours before clinical recognition, providing the early evidence base for ML-based early warning systems.

---

### HL7 ADT Messaging and Healthcare Interoperability

The `HL7ADTParser` in `src/ingestion/hl7_parser.py` handles Cerner Millennium message structure — MSH field indexing, PID CX identifier routing, PV1.19 encounter ID extraction, DG1 ICD-10 parsing. The clinical value of ADT-based real-time processing is established in the literature:

**Mandel, J.C., Kreda, D.A., Mandl, K.D., Kohane, I.S., & Ramoni, R.B. (2016). SMART on FHIR: a standards-based, interoperable apps platform for electronic health records.** *Journal of the American Medical Informatics Association (JAMIA), 23(5):899–908.* [DOI: 10.1093/jamia/ocv189](https://doi.org/10.1093/jamia/ocv189) | PubMed: 26911829

> Introduced the SMART on FHIR platform — a standards-based "app store" for EHR-integrated applications. SMART on FHIR is the production integration mechanism for hospital AI command centers: the HL7 v2.x parser in this system handles the legacy ADT feed format (still the dominant hospital interface standard), and FHIR R4 adapter development is documented in the roadmap as the next interoperability layer.

**Multiple authors, 2017. Hospitalization event notifications and reductions in readmissions of Medicare fee-for-service beneficiaries in the Bronx, New York.** *Journal of the American Medical Informatics Association (JAMIA), 24(e1):e150.* [DOI: 10.1093/jamia/ocw139](https://doi.org/10.1093/jamia/ocw139)

> Demonstrated that HL7 ADT-based hospitalization event notifications sent to care teams reduced 30-day readmissions in Medicare patients — directly validating the core premise of this system. The ADT event (A01/A02/A03/A08) is the trigger that initiates the entire pipeline; this study showed that simply acting on ADT events improves outcomes, and an AI pipeline on top of those events extends that benefit.

**Systematic review, 2021. The Fast Health Interoperability Resources (FHIR) Standard: Systematic Literature Review of Implementations, Applications, Challenges and Opportunities.** *JMIR Medical Informatics.* [DOI: 10.2196/21929](https://doi.org/10.2196/21929) | PMC: PMC8408751

> Systematic review of 141 FHIR implementation studies. Relevant finding: ADT-based event feeds are the dominant real-time data source for hospital AI, but FHIR R4 `Encounter` resources are the emerging standard for new implementations. The HL7 v2.x ADT parser in this system handles the deployed installed base; FHIR R4 is the planned forward path.

**Multiple authors, 2024. HL7 Fast Healthcare Interoperability Resources (HL7 FHIR) in digital healthcare ecosystems for chronic disease management.** *International Journal of Medical Informatics.* [DOI: 10.1016/j.ijmedinf.2024.105500](https://doi.org/10.1016/j.ijmedinf.2024.105500)

> Scoping review of FHIR implementations for chronic disease management, covering real-time event feeds, care coordination use cases, and system integration challenges. The ADT A01/A08 events processed by this system correspond to FHIR `Encounter.status` transitions — this paper maps the conceptual alignment between the two standards.

---

### Clinical AI Accountability, Bias, and Human-in-the-Loop Design

The Accept/Override/Escalate accountability system is not just an audit feature — it is the implementation of a set of design principles the research community has identified as essential for safe clinical AI deployment:

**Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S. (2019). Dissecting racial bias in an algorithm used to manage the health of populations.** *Science, 366(6464):447–453.* [DOI: 10.1126/science.aax2342](https://doi.org/10.1126/science.aax2342) | PubMed: 31649194

> Demonstrated that a widely-deployed commercial health algorithm systematically underestimated illness burden in Black patients by over 50% by using cost as a proxy for health need. The most-cited clinical AI accountability paper in existence. This study motivates three design choices in this system: (1) `payer_type_encoded` is a direct input feature rather than proxied through cost, making payer bias explicit; (2) SHAP explanations surface any demographic feature's contribution to individual predictions; (3) the accountability log captures Override decisions with clinician notes, enabling retroactive bias auditing.

**Multiple authors, 2025. For trustworthy AI, keep the human in the loop.** *Nature Medicine.* [DOI: 10.1038/s41591-025-04033-7](https://doi.org/10.1038/s41591-025-04033-7)

> Argues that active human involvement in AI decision loops — not passive oversight — is essential for safe clinical AI, and proposes design principles for human-AI teaming that preserve clinician agency and accountability. The three-path decision model (Accept/Override/Escalate) directly implements the "active involvement" principle: clinicians are required to make an explicit decision on every recommendation, with their choice logged to an immutable record.

**Multiple authors, 2024. Why do users override alerts? Utilizing large language models to summarize comments and optimize clinical decision support.** PMC: PMC11105133

> Analyzed clinician-written free-text override comments using LLMs to identify patterns in why alerts are dismissed, finding that most overrides are clinically justified but poorly captured by existing systems. The `note` field in `action_records` — allowing clinicians to annotate every Override decision — directly enables the feedback analysis described in this paper. A production extension would apply the same LLM analysis to identify high-frequency override reasons and improve alert precision.

**Multiple authors, 2023. Clinician in the loop: a flawed solution for AI oversight.** ResearchGate. [URL](https://www.researchgate.net/publication/404514656)

> A critical counterpoint arguing that placing the clinician "in the loop" as a post-hoc check is insufficient for genuine accountability due to alert fatigue and automation bias. This paper motivates the role-based view design — different roles see different subsets of recommendations, reducing per-session alert volume — and the Escalate path that creates a visible, tracked escalation record rather than passive acknowledgment.

---

### Summary Table

| Domain | Key Papers | System Implementation |
|---|---|---|
| Gradient boosting | Chen & Guestrin (KDD 2016), Zhang et al. (iScience 2024) | `RiskScoringEngine` — 5 XGBoost `.ubj` artifacts |
| 30-day readmission | Bates et al. (Health Affairs 2014), EU Cardiovascular Nursing SR (2024) | `readmission_30d` model, 18-feature `PatientFeatures` |
| Sepsis detection | Komorowski et al. (Nature Medicine 2018), Reyna et al. (CCM 2020) | `sepsis_node` — 6-hour SEP-1 window, Rapid Response routing |
| Sepsis implementation | Sendak et al. (JMIR 2020) | `assign_to` routing, accountability log design |
| SHAP explainability | Lundberg & Lee (NeurIPS 2017), Lundberg et al. (Nat Biomed Eng 2018) | `shap.TreeExplainer` per model, waterfall charts in dashboard |
| Clinician-facing SHAP | Tonekaboni et al. (MLHC 2019) | `_DISPLAY_NAMES` mapping, per-patient waterfall |
| Multi-agent clinical AI | Kim et al. (NeurIPS 2024), Survey (arXiv 2025) | 8-node LangGraph `StateGraph`, specialized agent nodes |
| RAG architecture | Lewis et al. (NeurIPS 2020) | `ClinicalRAG`, ChromaDB, `all-MiniLM-L6-v2` |
| Clinical RAG | Kresevic et al. (npj DM 2024), Meta-analysis (PMC 2024) | 10 guideline documents, contextual query construction |
| LLM foundation | Singhal et al. (Nature 2023), Nori et al. (arXiv 2023) | `intervention_node` — Claude claude-sonnet-4-6 synthesis |
| Real-time CDSS | Topol (Nature Medicine 2019), Rajkomar et al. (npj DM 2018) | SSE stream, per-event pipeline execution |
| CDSS governance | JAMIA (2024), Ash et al. (JAMIA 2004) | Conditional agent routing, named assignees, override log |
| Deterioration | Escobar et al. (NEJM 2020), MEWS++ (PMC 2020) | `deterioration` model, `acuity_tier` → ICU routing |
| HL7 interoperability | Mandel et al. (JAMIA 2016), JAMIA ADT study (2017) | `HL7ADTParser` — Cerner Millennium structure |
| Clinical AI bias | Obermeyer et al. (Science 2019) | Explicit `payer_type_encoded` feature, SHAP audit trail |
| Human-in-the-loop | Nature Medicine (2025), PMC override study (2024) | Accept/Override/Escalate + `action_records` note field |

---

## References

### Machine Learning and Predictive Modeling

[1] Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. https://doi.org/10.1145/2939672.2939785

[2] Zhang, Y., Xiang, T., et al. (2024). Explainable machine learning for predicting 30-day readmission in acute heart failure patients. *iScience*, 27(8). https://doi.org/10.1016/j.isci.2024.110281

[3] Bates, D. W., Saria, S., Ohno-Machado, L., Shah, A., & Escobar, G. (2014). Big data in health care: using analytics to identify and manage high-risk and high-cost patients. *Health Affairs*, 33(7), 1123–1131. https://doi.org/10.1377/hlthaff.2014.0041

[4] Machine learning–based 30-day readmission prediction models for patients with heart failure: a systematic review. (2024). *European Journal of Cardiovascular Nursing*. https://doi.org/10.1093/eurjcn/zvae031

[5] Rajkomar, A., Oren, E., Chen, K., et al. (2018). Scalable and accurate deep learning with electronic health records. *npj Digital Medicine*, 1, 18. https://doi.org/10.1038/s41746-018-0029-1

[6] Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S. (2019). Dissecting racial bias in an algorithm used to manage the health of populations. *Science*, 366(6464), 447–453. https://doi.org/10.1126/science.aax2342

[7] Rajpurkar, P., Chen, E., Banerjee, O., & Topol, E. J. (2022). AI in health and medicine. *Nature Medicine*, 28, 31–38. https://doi.org/10.1038/s41591-021-01614-0

[8] Tomašev, N., Glorot, X., Rae, J. W., et al. (2019). A clinically applicable approach to continuous prediction of future acute kidney injury. *Nature*, 572, 116–119. https://doi.org/10.1038/s41586-019-1390-1

---

### Sepsis Detection and Critical Care AI

[9] Komorowski, M., Celi, L. A., Badawi, O., Gordon, A. C., & Faisal, A. A. (2018). The Artificial Intelligence Clinician learns optimal treatment strategies for sepsis in intensive care. *Nature Medicine*, 24, 1716–1720. https://doi.org/10.1038/s41591-018-0213-5

[10] Reyna, M. A., Josef, C. S., Jeter, R., et al. (2020). Early Prediction of Sepsis From Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019. *Critical Care Medicine*, 48(2), 210–217. https://doi.org/10.1097/CCM.0000000000004145

[11] Sendak, M. P., Ratliff, W., Sarro, D., et al. (2020). Real-World Integration of a Sepsis Deep Learning Technology Into Routine Clinical Care: Implementation Study. *JMIR Medical Informatics*, 8(7), e15182. https://doi.org/10.2196/15182

[12] Nemati, S., Holder, A., Razmi, F., Stanley, M. D., Clifford, G. D., & Buchman, T. G. (2018). An Interpretable Machine Learning Model for Accurate Prediction of Sepsis in the ICU. *Critical Care Medicine*, 46(4), 547–553. https://doi.org/10.1097/CCM.0000000000002936

[13] Early Prediction of Sepsis in the ICU Using Machine Learning: A Systematic Review. (2021). *Frontiers in Medicine*. https://doi.org/10.3389/fmed.2021.607952

[14] Singer, M., Deutschman, C. S., Seymour, C. W., et al. (2016). The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). *JAMA*, 315(8), 801–810. https://doi.org/10.1001/jama.2016.0287

[15] Fleuren, L. M., Klausch, T. L. T., Zwager, C. L., et al. (2020). Machine learning for the prediction of sepsis: a systematic review and meta-analysis of diagnostic test accuracy. *Intensive Care Medicine*, 46, 383–400. https://doi.org/10.1007/s00134-019-05872-y

---

### Explainability and Interpretable AI in Healthcare

[16] Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 4765–4774. https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf

[17] Lundberg, S. M., Nair, B., Vavilala, M. S., et al. (2018). Explainable machine-learning predictions for the prevention of hypoxaemia during surgery. *Nature Biomedical Engineering*, 2, 749–760. https://doi.org/10.1038/s41551-018-0304-0

[18] Lundberg, S. M., Erion, G. G., Chen, H., et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2, 56–67. https://doi.org/10.1038/s42256-019-0138-9

[19] Tonekaboni, S., Joshi, S., McCradden, M. D., & Goldenberg, A. (2019). What Clinicians Want: Contextualizing Explainable Machine Learning for Clinical End Use. *Proceedings of the 4th Machine Learning for Healthcare Conference (MLHC)*, PMLR 106, 359–380. https://proceedings.mlr.press/v106/tonekaboni19a.html

[20] Ghassemi, M., Oakden-Rayner, L., & Beam, A. L. (2021). The false hope of current approaches to explainable artificial intelligence in health care. *The Lancet Digital Health*, 3(11), e745–e750. https://doi.org/10.1016/S2589-7500(21)00208-9

[21] Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence*, 1, 206–215. https://doi.org/10.1038/s42256-019-0048-x

---

### Multi-Agent AI and LangGraph Orchestration

[22] Kim, Y., Park, C., Jeong, H., et al. (2024). MDAgents: An Adaptive Collaboration of LLMs for Medical Decision-Making. *Advances in Neural Information Processing Systems (NeurIPS 2024)*. https://arxiv.org/abs/2404.15155

[23] A Survey of LLM-based Agents in Medicine: How far are we from Baymax? (2025). *arXiv preprint*. https://arxiv.org/abs/2502.11211

[24] ClinicalAgents: Multi-Agent Orchestration for Clinical Decision Making with Dual-Memory. (2025). *arXiv preprint*. https://arxiv.org/abs/2603.26182

[25] Wang, L., Ma, C., Feng, X., et al. (2024). A Survey on Large Language Model based Autonomous Agents. *Frontiers of Computer Science*, 18, 186345. https://doi.org/10.1007/s11704-024-40231-1

[26] Park, J. S., O'Brien, J. C., Cai, C. J., et al. (2023). Generative Agents: Interactive Simulacra of Human Behavior. *Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology (UIST)*. https://doi.org/10.1145/3586183.3606763

---

### Retrieval-Augmented Generation in Clinical Settings

[27] Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *Advances in Neural Information Processing Systems (NeurIPS)*, 33, 9459–9474. https://arxiv.org/abs/2005.11401

[28] Kresevic, S., Giuffrè, M., Ajčević, M., et al. (2024). Optimization of hepatological clinical guidelines interpretation by large language models: a retrieval augmented generation-based framework. *npj Digital Medicine*, 7, 102. https://doi.org/10.1038/s41746-024-01091-y

[29] Improving large language model applications in biomedicine with retrieval-augmented generation: a systematic review, meta-analysis, and clinical development guidelines. (2024). PMC12005634.

[30] Wornow, M., Xu, Y., Thapa, R., et al. (2023). The shaky foundations of large language models and foundation models for electronic health records. *npj Digital Medicine*, 6, 135. https://doi.org/10.1038/s41746-023-00879-8

[31] Zakka, C., Shad, R., Chaurasia, A., et al. (2024). Almanac — Retrieval-Augmented Language Models for Clinical Medicine. *NEJM AI*, 1(2). https://doi.org/10.1056/AIoa2300068

---

### Large Language Models in Clinical Practice

[32] Singhal, K., Azizi, S., Tu, T., et al. (2023). Large language models encode clinical knowledge. *Nature*, 620(7972), 172–180. https://doi.org/10.1038/s41586-023-06291-2

[33] Nori, H., King, N., McKinney, S. M., Carignan, D., & Horvitz, E. (2023). Capabilities of GPT-4 on Medical Challenge Problems. *arXiv preprint*. https://arxiv.org/abs/2303.13375

[34] The potential of GPT-4 to analyse medical notes in three different languages: a retrospective model-evaluation study. (2024). *The Lancet Digital Health*. https://doi.org/10.1016/S2589-7500(24)00246-2

[35] Davenport, T. H., & Kalakota, R. (2019). The potential for artificial intelligence in healthcare. *Future Healthcare Journal*, 6(2), 94–98. https://doi.org/10.7861/futurehosp.6-2-94

[36] Thirunavukarasu, A. J., Ting, D. S. J., Elangovan, K., et al. (2023). Large language models in medicine. *Nature Medicine*, 29, 1930–1940. https://doi.org/10.1038/s41591-023-02448-8

[37] Moor, M., Banerjee, O., Abad, Z. S. H., et al. (2023). Foundation models for generalist medical artificial intelligence. *Nature*, 616, 259–265. https://doi.org/10.1038/s41586-023-05881-4

---

### Real-Time Clinical Decision Support

[38] Topol, E. J. (2019). High-performance medicine: the convergence of human and artificial intelligence. *Nature Medicine*, 25(1), 44–56. https://doi.org/10.1038/s41591-018-0300-7

[39] Toward a responsible future: recommendations for AI-enabled clinical decision support. (2024). *Journal of the American Medical Informatics Association (JAMIA)*, 31(11), 2730. https://doi.org/10.1093/jamia/ocae207

[40] Ash, J. S., Berg, M., & Coiera, E. (2004). Some unintended consequences of information technology in health care: the nature of patient care information system-related errors. *Journal of the American Medical Informatics Association (JAMIA)*, 11(2), 104–112. PMID: 14633935

[41] Kawamoto, K., Houlihan, C. A., Balas, E. A., & Lobach, D. F. (2005). Improving clinical practice using clinical decision support systems: a systematic review of trials to identify features critical to success. *BMJ*, 330(7494), 765. https://doi.org/10.1136/bmj.38398.500764.8F

[42] Sutton, R. T., Pincock, D., Baumgart, D. C., et al. (2020). An overview of clinical decision support systems: benefits, risks, and strategies for success. *npj Digital Medicine*, 3, 17. https://doi.org/10.1038/s41746-020-0221-y

---

### Patient Deterioration and Early Warning Systems

[43] Escobar, G. J., Liu, V. X., Schuler, A., Lawson, B., Greene, J. D., & Kipnis, P. (2020). Automated Identification of Adults at Risk for In-Hospital Clinical Deterioration. *New England Journal of Medicine*, 383(20), 1951–1960. https://doi.org/10.1056/NEJMsa2001090

[44] MEWS++: Enhancing the Prediction of Clinical Deterioration in Admitted Patients through a Machine Learning Model. (2020). PMC7073544.

[45] Escobar, G. J., LaGuardia, J. C., Turk, B. J., et al. (2012). Early detection of impending physiologic deterioration among patients who are not in intensive care. *Journal of Hospital Medicine*, 7(5), 388–395. https://doi.org/10.1002/jhm.1929

[46] Smith, G. B., Prytherch, D. R., Meredith, P., Schmidt, P. E., & Featherstone, P. I. (2013). The ability of the National Early Warning Score (NEWS) to discriminate patients at risk of early cardiac arrest, unanticipated intensive care unit admission, and death. *Resuscitation*, 84(4), 465–470. https://doi.org/10.1016/j.resuscitation.2012.12.016

[47] Churpek, M. M., Treml, A. N., et al. (2016). Multicenter comparison of machine learning methods and conventional regression for predicting clinical deterioration on the wards. *Critical Care Medicine*, 44(2), 368–374. https://doi.org/10.1097/CCM.0000000000001571

---

### HL7, FHIR, and Healthcare Interoperability

[48] Mandel, J. C., Kreda, D. A., Mandl, K. D., Kohane, I. S., & Ramoni, R. B. (2016). SMART on FHIR: a standards-based, interoperable apps platform for electronic health records. *Journal of the American Medical Informatics Association (JAMIA)*, 23(5), 899–908. https://doi.org/10.1093/jamia/ocv189

[49] Hospitalization event notifications and reductions in readmissions of Medicare fee-for-service beneficiaries in the Bronx, New York. (2017). *Journal of the American Medical Informatics Association (JAMIA)*, 24(e1), e150. https://doi.org/10.1093/jamia/ocw139

[50] The Fast Health Interoperability Resources (FHIR) Standard: Systematic Literature Review of Implementations, Applications, Challenges and Opportunities. (2021). *JMIR Medical Informatics*. https://doi.org/10.2196/21929

[51] HL7 Fast Healthcare Interoperability Resources (HL7 FHIR) in digital healthcare ecosystems for chronic disease management. (2024). *International Journal of Medical Informatics*. https://doi.org/10.1016/j.ijmedinf.2024.105500

[52] Lehne, M., Sass, J., Essenwanger, A., Schepers, J., & Thun, S. (2019). Why digital medicine depends on interoperability. *npj Digital Medicine*, 2, 79. https://doi.org/10.1038/s41746-019-0158-1

---

### Clinical AI Accountability, Safety, and Human-in-the-Loop

[53] For trustworthy AI, keep the human in the loop. (2025). *Nature Medicine*. https://doi.org/10.1038/s41591-025-04033-7

[54] Why do users override alerts? Utilizing large language models to summarize comments and optimize clinical decision support. (2024). PMC11105133.

[55] Clinician in the loop: a flawed solution for AI oversight. (2023). ResearchGate. https://www.researchgate.net/publication/404514656

[56] Char, D. S., Shah, N. H., & Magnus, D. (2018). Implementing Machine Learning in Health Care — Addressing Ethical Challenges. *New England Journal of Medicine*, 378(11), 981–983. https://doi.org/10.1056/NEJMp1714229

[57] Price, W. N., & Cohen, I. G. (2019). Privacy in the age of medical big data. *Nature Medicine*, 25, 37–43. https://doi.org/10.1038/s41591-018-0272-7

[58] Shortliffe, E. H., & Sepúlveda, M. J. (2018). Clinical Decision Support in the Era of Artificial Intelligence. *JAMA*, 320(21), 2199–2200. https://doi.org/10.1001/jama.2018.17163

---

### Hospital Operations and Care Management

[59] Krumholz, H. M. (2013). Post-Hospital Syndrome — An Acquired, Transient Condition of Generalized Risk. *New England Journal of Medicine*, 368(2), 100–102. https://doi.org/10.1056/NEJMp1212324

[60] Jencks, S. F., Williams, M. V., & Coleman, E. A. (2009). Rehospitalizations among Patients in the Medicare Fee-for-Service Program. *New England Journal of Medicine*, 360(14), 1418–1428. https://doi.org/10.1056/NEJMsa0803563

[61] Hansen, L. O., Young, R. S., Hinami, K., Leung, A., & Williams, M. V. (2011). Interventions to Reduce 30-Day Rehospitalization: A Systematic Review. *Annals of Internal Medicine*, 155(8), 520–528. https://doi.org/10.7326/0003-4819-155-8-201110180-00008

[62] Vest, J. R., Kern, L. M., Silver, M. D., & Kaushal, R. (2015). The potential for community-based health information exchange systems to reduce hospital readmissions. *Journal of the American Medical Informatics Association (JAMIA)*, 22(2), 435–442. https://doi.org/10.1136/amiajnl-2014-002760

---

### Emerging Directions: Agentic AI and Foundation Models in Healthcare

[63] Tu, T., Palepu, A., Schaekermann, M., et al. (2024). Towards Conversational Diagnostic AI. *arXiv preprint*. https://arxiv.org/abs/2401.05654

[64] McDuff, D., Schaekermann, M., Tu, T., et al. (2023). Towards Accurate Differential Diagnosis with Large Language Models. *arXiv preprint*. https://arxiv.org/abs/2312.00164

[65] Cascella, M., Montomoli, J., Bellini, V., & Bignami, E. (2023). Evaluating the Feasibility of ChatGPT in Healthcare: An Analysis of Multiple Clinical and Research Scenarios. *Journal of Medical Systems*, 47, 33. https://doi.org/10.1007/s10916-023-01925-4

[66] Nori, H., Lee, Y. T., Zhang, S., et al. (2023). Can Large Language Models be Used to Provide Medical Advice? *arXiv preprint*. https://arxiv.org/abs/2311.02396

[67] Sallam, M. (2023). ChatGPT Utility in Healthcare Education, Research, and Practice: Systematic Review on the Promising Perspectives and Valid Concerns. *Healthcare*, 11(6), 887. https://doi.org/10.3390/healthcare11060887

---

## License

MIT
