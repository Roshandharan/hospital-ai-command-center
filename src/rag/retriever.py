"""
ChromaDB-backed clinical knowledge retrieval.
Uses embedded ChromaDB (no separate service) seeded with CMS/HEDIS/NPSG guidelines.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import structlog

from src.config import get_settings

log = structlog.get_logger(__name__)


@dataclass
class RetrievedContext:
    documents: list[str]
    sources: list[str]
    scores: list[float]
    query: str

    def as_prompt_text(self) -> str:
        if not self.documents:
            return "No relevant clinical guidelines retrieved."
        return "\n\n---\n\n".join(
            f"[{src} | relevance: {score:.2f}]\n{doc}"
            for doc, src, score in zip(self.documents, self.sources, self.scores)
        )


CLINICAL_GUIDELINES = [
    {"id": "cms_readmission", "source": "CMS HRRP", "text": "Hospital Readmissions Reduction Program: 30-day all-cause readmission rates above national threshold trigger payment reduction. High-risk indicators: age ≥65, Charlson Comorbidity Index ≥3, prior admissions ≥2, CHF/COPD diagnosis, Medicaid/self-pay payer. Care management intervention reduces 30-day readmission by 20-25%."},
    {"id": "sep1_bundle", "source": "CMS SEP-1", "text": "SEP-1 Severe Sepsis Bundle. Within 3 hours: (1) Measure lactate. (2) Blood cultures before antibiotics. (3) Broad-spectrum antibiotics. (4) 30ml/kg crystalloid for hypotension or lactate ≥4. Within 6 hours: (5) Vasopressors for refractory hypotension. (6) Reassess lactate if initial ≥2. Time-sensitive — 6-hour compliance window."},
    {"id": "hedis_pcr", "source": "HEDIS PCR", "text": "HEDIS Plan All-Cause Readmission (PCR): admissions followed by readmission within 30 days. High risk: age ≥65, CCI ≥3, prior admissions ≥2, CHF/COPD, Medicaid/self-pay, no follow-up within 7 days post-discharge. Benchmark: top decile plans achieve <12% 30-day readmission."},
    {"id": "npsg_2024", "source": "Joint Commission NPSG 2024", "text": "NPSG 01.01.01: Two patient identifiers. NPSG 02.03.01: Critical result reporting within defined timeframe. NPSG 06.01.01: Clinical alarm management — respond to physiologic monitor alarms per policy. NPSG 15.01.01: Suicide risk screening in ED and behavioral health settings."},
    {"id": "language_access", "source": "CMS Language Access", "text": "Title VI Civil Rights Act: hospital must provide interpreter services at no cost for Limited English Proficiency (LEP) patients. Language barriers associated with 30% increase in adverse events and significantly higher no-show rates. Screen all patients at registration, document in EHR, auto-trigger interpreter for non-English primary language."},
    {"id": "transport_sdoh", "source": "SDOH Framework", "text": "Transportation barriers account for 3.6 million missed medical appointments annually. Patients >10 miles from facility are 2x more likely to miss appointments. Proactive transport assistance (MTM, Lyft Health) reduces no-show rates 35-40%. Screen: distance >10 miles, Medicaid payer, prior no-shows, language barrier."},
    {"id": "charlson_cci", "source": "Charlson Comorbidity Index", "text": "CCI score predicts 10-year mortality: 0=98% survival, 1-2=89%, 3-4=77%, ≥5=21%. CCI ≥3 indicates high readmission risk — trigger care management. Weighted conditions include MI, CHF, PVD, dementia, COPD, liver disease, diabetes, renal disease, malignancy."},
    {"id": "discharge_planning", "source": "CMS CoP 482.13", "text": "Discharge planning must begin at admission for high-risk patients. Elements: post-discharge services identification, follow-up within 7 days, community resource coordination, medication reconciliation, patient/family education (teach-back), post-discharge phone call within 72 hours. BOOST toolkit reduces 30-day readmission 20-30% in elderly."},
    {"id": "geriatric_care", "source": "AGS Geriatric Guidelines", "text": "Patients ≥65 at elevated risk for readmission, delirium, falls, functional decline. HOSPITAL Score predicts 30-day readmission: hemoglobin <12, discharge from oncology, sodium <135, procedure in-hospital, index admission type emergency, ≥1 prior admission, length of stay >5 days. Score ≥5 = high risk."},
    {"id": "ed_throughput", "source": "ACEP ESI Guidelines", "text": "Emergency Severity Index (ESI) 5-level triage. ESI-1: immediate life-saving intervention. ESI-2: high risk, confused, severe pain — door-to-physician ≤10 min. ESI-3: 2+ resources, stable vitals. ESI-4: 1 resource. ESI-5: no resources. Door-to-physician target ≤30 minutes. Patients with deterioration risk >0.6 should be upgraded to ESI-2."},
]


class ClinicalRAG:
    def __init__(self) -> None:
        self._available = False
        self._client = None
        self._collection = None
        self._encoder = None
        self._settings = get_settings()

    def initialize(self) -> None:
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer

            persist_path = Path(self._settings.chroma_persist_path)
            persist_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(persist_path))
            self._collection = self._client.get_or_create_collection(
                name=self._settings.chroma_collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
            if self._collection.count() == 0:
                self._seed()
            self._available = True
            log.info("rag.initialized", docs=self._collection.count())
        except Exception as e:
            log.warning("rag.unavailable", error=str(e))
            self._available = False

    def _seed(self) -> None:
        docs  = [g["text"]   for g in CLINICAL_GUIDELINES]
        ids   = [g["id"]     for g in CLINICAL_GUIDELINES]
        metas = [{"source": g["source"]} for g in CLINICAL_GUIDELINES]
        embeddings = self._encoder.encode(docs).tolist()
        self._collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
        log.info("rag.seeded", count=len(docs))

    def retrieve(self, query: str, n: int = 4) -> RetrievedContext:
        if not self._available:
            return RetrievedContext([], [], [], query)
        try:
            emb = self._encoder.encode(query).tolist()
            count = self._collection.count()
            results = self._collection.query(
                query_embeddings=[emb],
                n_results=min(n, count),
                include=["documents", "metadatas", "distances"],
            )
            docs  = results["documents"][0] if results["documents"] else []
            metas = results["metadatas"][0]  if results["metadatas"]  else []
            dists = results["distances"][0]  if results["distances"]  else []
            return RetrievedContext(
                documents=docs,
                sources=[m.get("source", "") for m in metas],
                scores=[round(1 - d, 4) for d in dists],
                query=query,
            )
        except Exception as e:
            log.error("rag.retrieve_failed", error=str(e))
            return RetrievedContext([], [], [], query)
