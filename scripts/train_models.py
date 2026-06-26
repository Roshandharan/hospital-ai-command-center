"""
Train XGBoost models on synthetic hospital data.
Usage: python scripts/train_models.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

FEATURE_COLS = [
    "age", "sex_encoded", "patient_class_encoded",
    "prior_admissions_12m", "prior_ed_visits_12m", "prior_no_shows_12m",
    "active_problem_count", "medication_count", "days_since_last_visit",
    "appt_lead_time_days", "appt_hour", "appt_day_of_week",
    "is_new_patient", "payer_type_encoded", "distance_miles",
    "language_barrier", "charlson_index", "num_chronic_conditions",
]


def generate_training_data(n: int = 5000) -> pd.DataFrame:
    np.random.seed(42)
    df = pd.DataFrame({
        "age":                   np.clip(np.random.normal(62, 18, n).astype(int), 18, 95),
        "sex_encoded":           np.random.choice([0, 1, 2], n, p=[0.50, 0.48, 0.02]),
        "patient_class_encoded": np.random.choice([0, 1, 2], n, p=[0.25, 0.45, 0.30]),
        "prior_admissions_12m":  np.random.poisson(1.2, n).clip(0, 8),
        "prior_ed_visits_12m":   np.random.poisson(0.8, n).clip(0, 6),
        "prior_no_shows_12m":    np.random.poisson(0.5, n).clip(0, 5),
        "active_problem_count":  np.random.poisson(3, n).clip(0, 15),
        "medication_count":      np.random.poisson(4, n).clip(0, 20),
        "days_since_last_visit": np.random.exponential(60, n).clip(1, 365),
        "appt_lead_time_days":   np.random.exponential(10, n).clip(1, 60),
        "appt_hour":             np.random.randint(7, 18, n),
        "appt_day_of_week":      np.random.randint(0, 7, n),
        "is_new_patient":        np.random.choice([0, 1], n, p=[0.75, 0.25]),
        "payer_type_encoded":    np.random.choice([0, 1, 2, 3], n, p=[0.40, 0.35, 0.15, 0.10]),
        "distance_miles":        np.random.exponential(8, n).clip(0.5, 45),
        "language_barrier":      np.random.choice([0, 1], n, p=[0.78, 0.22]),
        "charlson_index":        np.random.exponential(1.5, n).clip(0, 10),
        "num_chronic_conditions": np.random.poisson(2.5, n).clip(0, 12),
    })

    # readmission: ~15-20% positive rate
    readmit_score = (
        (df.age / 90) * 0.20
        + (df.prior_admissions_12m / 5) * 0.30
        + (df.charlson_index / 8) * 0.25
        + (df.payer_type_encoded / 3) * 0.10
        + df.language_barrier * 0.05
        + (df.patient_class_encoded == 2) * 0.10
        + np.random.uniform(0, 0.10, n)
    ).clip(0, 1)
    df["readmitted_30d"] = (readmit_score > 0.42).astype(int)

    # deterioration: ~18-22% positive rate
    deterirate_score = (
        (df.patient_class_encoded == 2) * 0.30
        + (df.active_problem_count / 15) * 0.25
        + (df.prior_ed_visits_12m / 6) * 0.20
        + (df.medication_count / 20) * 0.15
        + (df.charlson_index / 10) * 0.10
        + np.random.uniform(0, 0.10, n)
    ).clip(0, 1)
    df["deterioration_flag"] = (deterirate_score > 0.30).astype(int)

    # sepsis: ~8-12% positive rate
    sepsis_score = (
        deterirate_score * 0.50
        + readmit_score * 0.20
        + (df.patient_class_encoded == 2) * 0.20
        + np.random.uniform(0, 0.10, n)
    ).clip(0, 1)
    df["sepsis_flag"] = (sepsis_score > 0.45).astype(int)

    # discharge: ~35-40% today, ~55-60% tomorrow
    discharge_score = (
        1.0
        - readmit_score * 0.45
        - deterirate_score * 0.35
        + np.random.uniform(-0.08, 0.08, n)
    ).clip(0, 1)
    df["discharge_today_flag"]    = (discharge_score > 0.62).astype(int)
    df["discharge_tomorrow_flag"] = (discharge_score > 0.45).astype(int)

    print(f"Generated {n} rows")
    print(f"Readmission rate:  {df.readmitted_30d.mean():.1%}")
    print(f"Deterioration rate:{df.deterioration_flag.mean():.1%}")
    print(f"Sepsis rate:       {df.sepsis_flag.mean():.1%}")
    return df


def train() -> None:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.utils.class_weight import compute_sample_weight

    output_dir = Path("data/models")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = generate_training_data(5000)
    X = df[FEATURE_COLS].values.astype(np.float32)

    targets = {
        "readmission":        "readmitted_30d",
        "deterioration":      "deterioration_flag",
        "sepsis":             "sepsis_flag",
        "discharge_today":    "discharge_today_flag",
        "discharge_tomorrow": "discharge_tomorrow_flag",
    }

    params = dict(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        tree_method="hist", eval_metric="aucpr",
        early_stopping_rounds=30, random_state=42,
    )

    metrics = []
    for name, col in targets.items():
        y = df[col].values
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        weights = compute_sample_weight("balanced", y_tr)
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, sample_weight=weights, eval_set=[(X_val, y_val)], verbose=0)
        y_prob = model.predict_proba(X_val)[:, 1]
        auc   = roc_auc_score(y_val, y_prob)
        auprc = average_precision_score(y_val, y_prob)
        print(f"{name:22s}  AUC={auc:.3f}  AUPRC={auprc:.3f}  best_iter={model.best_iteration}")
        path = output_dir / f"{name}.ubj"
        model.save_model(str(path))
        metrics.append({"model": name, "auc": round(auc, 4), "auprc": round(auprc, 4)})

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nModels saved to {output_dir}/")


if __name__ == "__main__":
    train()
