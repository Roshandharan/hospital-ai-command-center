# Hospital AI Command Center

Clinical AI platforms are usually bolted onto EHRs as reporting tools — they surface insights in dashboards that clinicians might check once a day, long after the moment for intervention has passed. This project is built the other way: a real-time multi-agent system that runs the moment a patient event fires, scores risk across five dimensions, retrieves relevant clinical evidence, synthesizes an intervention plan, and routes specific recommendations to specific people — with a persistent record of whether anyone acted on them.

**Live:** https://hospital-ai-command-center-production.up.railway.app/dashboard

---

## Architecture

The core is a LangGraph `StateGraph` with seven nodes. Every ADT event (admission, discharge, transfer, update) flows through the graph, with conditional routing that determines which agents fire based on the scores computed by the previous node.

```
ADT Event
    │
    ▼
[Triage]      ESI level assignment, care pathway routing
    │
    ▼
[Risk]        XGBoost: 5 models — readmission, deterioration,
    │          sepsis, discharge today, discharge tomorrow
    │          SHAP values surface top contributing features
    │
    ├── sepsis_risk > 0.25 ──────────► [Sepsis]     SEP-1 bundle compliance,
    │                                               6-hour window tracking
    │
    ├── readmission > 0.30 ──────────► [Discharge]  Barrier identification,
    │   or discharge_today > 0.25                   target discharge time
    │
    └── acuity CRITICAL or HIGH ──────► [Bed]       Capacity check,
                                                     transfer routing
                                             │
                                             ▼
                                         [RAG]       ChromaDB: CMS CoP, SEP-1,
                                             │        HEDIS, NPSG guidelines
                                             ▼
                                      [MedSafety]    Interaction checking,
                                             │        dose adjustment, allergy flags
                                             ▼
                                    [Intervention]   Claude synthesizes all outputs
                                             │        → structured action plan
                                             ▼
                                        [Audit]      PostgreSQL persistence,
                                                     SSE broadcast to dashboard
```

The conditional routing is real — not every patient triggers every agent. A routine outpatient visit hits Triage, Risk, RAG, MedSafety, Intervention, and Audit. A critically ill ED patient who scores 67% on sepsis risk also fires the Sepsis and Bed agents. The dashboard's pipeline trace shows exactly which nodes ran and which were skipped, and why.

---

## Agents

**Triage** classifies the incoming event using ESI (Emergency Severity Index) logic and sets the routing context for downstream agents. ESI-1 and ESI-2 patients route to the ICU pathway; outpatient events skip the acuity-dependent nodes entirely.

**Risk** runs five XGBoost binary classifiers trained on 5,000 synthetic encounters: 30-day readmission probability, in-stay deterioration risk, sepsis likelihood, and discharge probability at today and tomorrow horizons. SHAP values identify the top contributing features per prediction — "prior admissions (increases risk)" — so the recommendation is grounded in something specific, not just a score.

**Sepsis** fires when sepsis risk crosses 25%. It checks SEP-1 bundle compliance, flags missing elements (lactate, cultures, antibiotics), and creates a time-sensitive recommendation with a 6-hour window. It's the only agent that runs in STAT mode — this is the one where clinical accountability matters most.

**Discharge** fires when readmission risk or discharge probability crosses a threshold. It identifies specific barriers (PT evaluation pending, SNF placement needed, labs outstanding), sets a target discharge time, and flags cases where post-discharge follow-up needs to be coordinated before the patient leaves.

**Bed** fires for HIGH and CRITICAL acuity patients. It checks unit capacity and recommends placement adjustments as acuity evolves. Running it only for high-acuity patients keeps it from generating noise on routine admissions.

**RAG** builds a context-specific query from the patient's risk profile and retrieves relevant guidelines from ChromaDB: sepsis bundle criteria if sepsis risk is elevated, readmission criteria if readmission risk is high, geriatric guidelines if the patient is over 65. The retrieved context feeds directly into the Intervention agent's prompt.

**MedSafety** reviews the active medication profile for drug-drug interactions, renal dose adjustments, high-alert medication requirements, and allergy conflicts. Confidence is highest on this agent (0.96) because drug safety rules are deterministic in a way that probabilistic risk prediction isn't.

**Intervention** uses Claude claude-sonnet-4-6 to synthesize all upstream agent outputs plus retrieved clinical guidelines into a structured JSON action plan: priority level, specific actions with named assignees, care management referral flag, language services, follow-up timing, and a clinical rationale in plain language. Falls back to a threshold-based deterministic plan when no API key is present.

**Audit** writes the complete pipeline run to PostgreSQL: all five XGBoost scores, all agent outputs, the intervention plan, RAG sources, SHAP factors, and execution metadata. Every run is a permanent record.

---

## Clinical Knowledge Retrieval

ChromaDB runs in embedded persistent mode — no separate service required. The collection is seeded on first startup with ten clinical reference documents: CMS HRRP readmission criteria, SEP-1 sepsis bundle, HEDIS PCR measure, Joint Commission NPSG 2024, CMS language access requirements, SDOH transportation barriers, Charlson Comorbidity Index, CMS CoP 482.13 discharge planning, AGS geriatric care guidelines, and ACEP ESI triage criteria.

Embeddings use `all-MiniLM-L6-v2` via sentence-transformers. Query construction is contextual — the RAG agent builds the query from the patient's risk profile rather than sending a generic request.

---

## Risk Models

Five XGBoost binary classifiers trained on 5,000 synthetic patient encounters with feature-label correlations calibrated to match published benchmark rates. Training uses an 80/20 split with balanced class weights and AUCPR early stopping.

The 18 input features cover demographics, utilization history, appointment patterns, payer type, geographic access, comorbidity burden, and clinical complexity. In a production deployment, these map directly to the Cerner Millennium HealtheAnalytics schema — the feature pipeline in `src/ml/features.py` is implemented for both synthetic and real data paths.

Typical validation performance: readmission AUC 0.83, deterioration 0.79, sepsis 0.77, discharge today 0.81.

---

## Action Accountability

Every agent recommendation has three paths: Accept, Override, or Escalate. The decision is logged to the `action_records` table with full attribution — staff member, role, timestamp, and any override note.

This creates an audit trail that answers questions traditional EHRs cannot: did anyone review the sepsis alert? Who overrode the discharge recommendation, and why? How many Bed Management recommendations were escalated this week? The Accountability view surfaces the full log with these details.

---

## Stack

**Orchestration:** LangGraph 0.2 StateGraph with conditional edges

**ML:** XGBoost 2.0 with SHAP explainability, scikit-learn, NumPy

**RAG:** ChromaDB (embedded persistent), sentence-transformers (all-MiniLM-L6-v2)

**LLM:** Claude claude-sonnet-4-6 via LangChain Anthropic

**API:** FastAPI 0.111, uvicorn (single worker), Server-Sent Events

**Database:** PostgreSQL 16, SQLAlchemy 2.0 async, asyncpg

**Infra:** Docker, Railway, GitHub Actions

---

## Running Locally

```bash
git clone https://github.com/Roshandharan/hospital-ai-command-center.git
cd hospital-ai-command-center

pip install -r requirements.txt

# Generate synthetic hospital census, OR schedule, ADT events
python scripts/generate_data.py

# Train XGBoost models (~30 seconds)
python scripts/train_models.py

# Configure (ANTHROPIC_API_KEY optional — fallback plans work without it)
cp .env.example .env

uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000/dashboard`.

The application starts without PostgreSQL or an Anthropic key. Without a database, actions are in-memory only and don't survive restarts. Without an API key, the Intervention agent uses deterministic threshold-based plans. The LangGraph pipeline, XGBoost models, and ChromaDB RAG all function without external dependencies.

---

## Deploying to Railway

Connect the GitHub repo to a new Railway project. Railway detects the Dockerfile and builds automatically.

Add the PostgreSQL plugin from the Railway dashboard — it auto-injects `DATABASE_URL`. Set `ANTHROPIC_API_KEY` in the Variables tab. Generate a public domain under Settings → Networking → port 8000.

First cold start takes 60–90 seconds: the sentence-transformer model downloads (~90MB), ChromaDB seeds the knowledge base, XGBoost artifacts load, and the LangGraph pipeline initializes.

---

## What's Real vs Synthetic

The pipeline, LangGraph graph, XGBoost models, ChromaDB RAG, PostgreSQL schema, and accountability system are production-quality implementations. The HL7 v2.x ADT parser in `src/ingestion/hl7_parser.py` handles real Cerner Millennium message structure — MSH field index corrections, PID CX identifier routing, PV1.19 visit number, DG1 ICD-10 extraction.

Patient data is synthetic. The census, vitals histories, lab results, OR schedule, and ADT events are generated procedurally on startup. Risk scores are computed by real XGBoost models trained on synthetic data, not real patient records.

In a production deployment, the feature pipeline queries Snowflake directly and the HL7 parser processes live Cerner feeds. Both are implemented. The synthetic generator exists only to make the demo runnable without hospital infrastructure.

---

## What's Missing for Production

Real-time HL7 integration against a live Cerner ADT feed (the parser is ready). Staff authentication against active directory so action records carry verified identity. Mobile push notifications for STAT recommendations. Model retraining on real encounter data once connected to Snowflake. A conversational query layer for ad-hoc operational questions.

The data model, agent architecture, and accountability system are designed to support all of it.
