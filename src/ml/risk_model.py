"""
XGBoost multi-output risk scorer with SHAP explainability.
Models: readmission_30d, deterioration, sepsis_risk, discharge_today
Falls back to heuristic scoring when artifacts not present (demo mode).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import structlog

log = structlog.get_logger(__name__)


@dataclass
class PatientFeatures:
    age: int
    sex_encoded: int                   # 0=F 1=M 2=Other
    patient_class_encoded: int         # 0=Out 1=In 2=ED
    prior_admissions_12m: int
    prior_ed_visits_12m: int
    prior_no_shows_12m: int
    active_problem_count: int
    medication_count: int
    days_since_last_visit: float
    appt_lead_time_days: float
    appt_hour: int
    appt_day_of_week: int
    is_new_patient: int
    payer_type_encoded: int
    distance_miles: float
    language_barrier: int
    charlson_index: float
    num_chronic_conditions: int

    def to_array(self) -> np.ndarray:
        return np.array([[
            self.age, self.sex_encoded, self.patient_class_encoded,
            self.prior_admissions_12m, self.prior_ed_visits_12m, self.prior_no_shows_12m,
            self.active_problem_count, self.medication_count, self.days_since_last_visit,
            self.appt_lead_time_days, self.appt_hour, self.appt_day_of_week,
            self.is_new_patient, self.payer_type_encoded, self.distance_miles,
            self.language_barrier, self.charlson_index, self.num_chronic_conditions,
        ]], dtype=np.float32)

    @property
    def feature_names(self) -> list[str]:
        return [
            "age", "sex", "patient_class",
            "prior_admissions_12m", "prior_ed_visits_12m", "prior_no_shows_12m",
            "active_problems", "medications", "days_since_last_visit",
            "appt_lead_time_days", "appt_hour", "appt_day_of_week",
            "is_new_patient", "payer_type", "distance_miles",
            "language_barrier", "charlson_index", "chronic_conditions",
        ]


@dataclass
class RiskScores:
    patient_id: str
    readmission_30d: float
    deterioration: float
    sepsis_risk: float
    discharge_today: float
    discharge_tomorrow: float
    los_predicted_days: int
    acuity_tier: str
    top_risk_factors: list[dict] = field(default_factory=list)
    demo_mode: bool = False


class RiskScoringEngine:
    _TIERS = [("CRITICAL", 0.75), ("HIGH", 0.50), ("MEDIUM", 0.25), ("LOW", 0.0)]

    def __init__(self, model_dir: Path):
        self._model_dir = model_dir
        self._models: dict = {}
        self._explainers: dict = {}
        self._loaded = False

    def load(self) -> None:
        try:
            import xgboost as xgb
            import shap
            targets = ["readmission", "deterioration", "sepsis", "discharge_today", "discharge_tomorrow"]
            loaded = 0
            for t in targets:
                p = self._model_dir / f"{t}.ubj"
                if p.exists():
                    m = xgb.XGBClassifier()
                    m.load_model(str(p))
                    self._models[t] = m
                    self._explainers[t] = shap.TreeExplainer(m)
                    loaded += 1
            if loaded > 0:
                log.info("models.loaded", count=loaded)
            else:
                log.info("models.demo_mode", reason="no artifacts found")
        except ImportError:
            log.warning("models.xgboost_missing")
        self._loaded = True

    def score(self, features: PatientFeatures, patient_id: str) -> RiskScores:
        if not self._loaded:
            self.load()
        if self._models:
            return self._score_real(features, patient_id)
        return self._score_heuristic(features, patient_id)

    def _score_real(self, f: PatientFeatures, pid: str) -> RiskScores:
        X = f.to_array()
        scores = {}
        factors = []
        for name, model in self._models.items():
            prob = float(model.predict_proba(X)[0][1])
            scores[name] = prob
            shap_vals = self._explainers[name].shap_values(X)[0]
            top = sorted(enumerate(shap_vals), key=lambda x: abs(x[1]), reverse=True)[:3]
            for i, v in top:
                if abs(v) > 0.01:
                    factors.append({
                        "feature": f.feature_names[i],
                        "shap_value": round(float(v), 4),
                        "direction": "increases" if v > 0 else "decreases",
                        "model": name,
                    })
        factors.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        max_risk = max(scores.get("readmission", 0), scores.get("deterioration", 0))
        los = self._predict_los(f, scores.get("readmission", 0))
        return RiskScores(
            patient_id=pid,
            readmission_30d=scores.get("readmission", 0),
            deterioration=scores.get("deterioration", 0),
            sepsis_risk=scores.get("sepsis", 0),
            discharge_today=scores.get("discharge_today", 0),
            discharge_tomorrow=scores.get("discharge_tomorrow", 0),
            los_predicted_days=los,
            acuity_tier=self._tier(max_risk),
            top_risk_factors=factors[:5],
            demo_mode=False,
        )

    def _score_heuristic(self, f: PatientFeatures, pid: str) -> RiskScores:
        """Deterministic heuristic scoring when no models are available."""
        readmission = min(0.95, (
            (min(f.age, 90) / 90) * 0.25
            + (min(f.prior_admissions_12m, 5) / 5) * 0.30
            + (min(f.charlson_index, 8) / 8) * 0.25
            + (f.payer_type_encoded / 3) * 0.10
            + f.language_barrier * 0.05
            + (f.patient_class_encoded == 2) * 0.05
        ))
        deterioration = min(0.95, (
            (f.patient_class_encoded == 2) * 0.35
            + (min(f.active_problem_count, 10) / 10) * 0.25
            + (min(f.prior_ed_visits_12m, 5) / 5) * 0.20
            + (min(f.medication_count, 15) / 15) * 0.15
            + (f.charlson_index > 3) * 0.05
        ))
        sepsis = min(0.90, deterioration * 0.6 + readmission * 0.2 + f.language_barrier * 0.05)
        discharge = min(0.95, max(0, 0.8 - readmission * 0.5 - deterioration * 0.4))
        discharge_tmrw = min(0.95, max(0, 0.85 - readmission * 0.4 - deterioration * 0.3))
        los = max(1, int((readmission * 14) + (deterioration * 7) + 1))
        max_risk = max(readmission, deterioration)
        factors = [
            {"feature": "age", "shap_value": round(f.age / 90 * 0.25, 3), "direction": "increases", "model": "heuristic"},
            {"feature": "prior_admissions_12m", "shap_value": round(f.prior_admissions_12m / 5 * 0.30, 3), "direction": "increases", "model": "heuristic"},
            {"feature": "charlson_index", "shap_value": round(f.charlson_index / 8 * 0.25, 3), "direction": "increases", "model": "heuristic"},
        ]
        return RiskScores(
            patient_id=pid,
            readmission_30d=round(readmission, 3),
            deterioration=round(deterioration, 3),
            sepsis_risk=round(sepsis, 3),
            discharge_today=round(discharge, 3),
            discharge_tomorrow=round(discharge_tmrw, 3),
            los_predicted_days=los,
            acuity_tier=self._tier(max_risk),
            top_risk_factors=factors,
            demo_mode=True,
        )

    def _tier(self, v: float) -> str:
        for tier, threshold in self._TIERS:
            if v >= threshold:
                return tier
        return "LOW"

    def _predict_los(self, f: PatientFeatures, readmission: float) -> int:
        base = {0: 2, 1: 5, 2: 3}.get(f.patient_class_encoded, 3)
        return max(1, int(base + readmission * 8 + f.charlson_index * 0.5))
