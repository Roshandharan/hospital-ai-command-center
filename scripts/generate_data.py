"""
Synthetic data generator for Hospital AI Command Center demo.
Generates realistic patient census, encounters, vitals, orders, labs,
OR schedule, and ADT events.
"""
import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

random.seed(42)
np.random.seed(42)

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ── Hospital Configuration ────────────────────────────────────────────────────

HOSPITAL_NAME = "Hospital AI Command Center"

UNITS = {
    "ED":     {"name": "Emergency Department", "beds": 24, "color": "#ef4444"},
    "MICU":   {"name": "Medical ICU",          "beds": 16, "color": "#f97316"},
    "CCU":    {"name": "Cardiac Care Unit",    "beds": 16, "color": "#f59e0b"},
    "SICU":   {"name": "Surgical ICU",         "beds": 12, "color": "#eab308"},
    "5NORTH": {"name": "Oncology",             "beds": 28, "color": "#22c55e"},
    "3EAST":  {"name": "General Medicine",     "beds": 32, "color": "#06b6d4"},
    "ORTHO":  {"name": "Orthopedics",          "beds": 20, "color": "#8b5cf6"},
    "OR":     {"name": "Operating Rooms",      "beds": 10, "color": "#ec4899"},
    "PACU":   {"name": "Post-Anesthesia Care", "beds": 12, "color": "#14b8a6"},
}

DIAGNOSES = {
    "ED":     ["Chest pain NOS", "Dyspnea", "Abdominal pain", "Syncope", "Altered mental status",
               "Sepsis", "Stroke", "Trauma", "GI bleed", "Allergic reaction"],
    "MICU":   ["Septic shock", "ARDS", "Respiratory failure", "DKA", "Hypertensive emergency",
               "Acute liver failure", "Drug overdose", "Pneumonia severe"],
    "CCU":    ["STEMI", "NSTEMI", "CHF exacerbation", "Atrial fibrillation", "Cardiac arrest",
               "Cardiogenic shock", "Aortic dissection", "Heart block"],
    "SICU":   ["Post-op monitoring", "Bowel resection", "Aortic aneurysm repair",
               "Hepatic resection", "Esophagectomy", "Trauma laparotomy"],
    "5NORTH": ["Lung cancer", "Colon cancer", "Breast cancer chemo", "Lymphoma",
               "Leukemia", "Neutropenic fever", "Bone marrow suppression"],
    "3EAST":  ["Pneumonia", "COPD exacerbation", "UTI", "Cellulitis", "DVT",
               "Anemia", "Hyponatremia", "Hip fracture", "Deconditioning"],
    "ORTHO":  ["Hip replacement", "Knee replacement", "Spine surgery", "Hip fracture",
               "Shoulder repair", "ACL reconstruction", "Fracture fixation"],
    "PACU":   ["Post-op monitoring", "Pain management", "Anesthesia recovery"],
}

OR_PROCEDURES = [
    "Total Hip Replacement", "Total Knee Replacement", "Laparoscopic Cholecystectomy",
    "Appendectomy", "Colectomy", "CABG", "Aortic Valve Replacement",
    "Spinal Fusion", "Craniotomy", "Breast Lumpectomy", "Hysterectomy",
    "Prostatectomy", "Nephrectomy", "Bowel Resection", "Hernia Repair",
    "Thyroidectomy", "Shoulder Arthroplasty", "Carpal Tunnel Release",
]

SURGEONS = [
    "Dr. Sarah Chen", "Dr. Marcus Williams", "Dr. Priya Patel",
    "Dr. James O'Brien", "Dr. Elena Rodriguez", "Dr. David Kim",
    "Dr. Rachel Thompson", "Dr. Michael Santos",
]

ANESTHESIOLOGISTS = [
    "Dr. Lisa Park", "Dr. Robert Nguyen", "Dr. Amanda Foster",
    "Dr. Christopher Lee",
]

FIRST_NAMES_M = ["James", "Michael", "Robert", "David", "John", "William", "Richard",
                  "Joseph", "Thomas", "Charles", "Carlos", "Miguel", "Wei", "Omar"]
FIRST_NAMES_F = ["Mary", "Patricia", "Jennifer", "Linda", "Barbara", "Elizabeth",
                  "Susan", "Jessica", "Sarah", "Karen", "Maria", "Ana", "Mei", "Fatima"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
              "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Chen", "Kim",
              "Patel", "Wilson", "Anderson", "Taylor", "Thomas", "Jackson"]

LAB_TESTS = ["CBC", "BMP", "CMP", "Troponin", "BNP", "Lactate", "Blood Culture",
             "UA", "Coagulation Panel", "ABG", "Lipase", "LFTs", "TSH"]

MEDICATIONS = ["Metoprolol", "Lisinopril", "Aspirin", "Heparin", "Vancomycin",
               "Piperacillin-Tazobactam", "Furosemide", "Morphine", "Ondansetron",
               "Pantoprazole", "Insulin", "Norepinephrine", "Amiodarone", "Warfarin"]

IMAGING = ["Chest X-Ray", "CT Head", "CT Chest", "CT Abdomen/Pelvis",
           "MRI Brain", "Echo", "Ultrasound Abdomen", "ECG"]

NOW = datetime.now()

# ── Helper Functions ──────────────────────────────────────────────────────────

def random_mrn():
    return f"MRN{random.randint(100000, 999999)}"

def random_name(sex):
    if sex == "M":
        return f"{random.choice(FIRST_NAMES_M)} {random.choice(LAST_NAMES)}"
    return f"{random.choice(FIRST_NAMES_F)} {random.choice(LAST_NAMES)}"

def random_age(unit):
    if unit in ("MICU", "CCU", "SICU"):
        return int(np.clip(np.random.normal(65, 15), 18, 95))
    if unit == "ED":
        return int(np.clip(np.random.normal(50, 20), 18, 95))
    if unit == "5NORTH":
        return int(np.clip(np.random.normal(60, 12), 30, 90))
    return int(np.clip(np.random.normal(62, 18), 18, 95))

def acuity_for_unit(unit):
    if unit in ("MICU", "CCU", "SICU"):
        return random.choices(["CRITICAL", "HIGH"], weights=[60, 40])[0]
    if unit == "ED":
        return random.choices(["CRITICAL", "HIGH", "MEDIUM"], weights=[20, 40, 40])[0]
    if unit == "5NORTH":
        return random.choices(["HIGH", "MEDIUM", "LOW"], weights=[30, 50, 20])[0]
    return random.choices(["HIGH", "MEDIUM", "LOW"], weights=[15, 45, 40])[0]

def risk_scores_for_acuity(acuity):
    if acuity == "CRITICAL":
        return {
            "readmission_30d": round(random.uniform(0.70, 0.95), 3),
            "deterioration":   round(random.uniform(0.65, 0.90), 3),
            "no_show":         round(random.uniform(0.10, 0.30), 3),
            "discharge_today": round(random.uniform(0.02, 0.08), 3),
            "discharge_tomorrow": round(random.uniform(0.05, 0.15), 3),
            "discharge_this_week": round(random.uniform(0.20, 0.40), 3),
            "sepsis_risk":     round(random.uniform(0.40, 0.80), 3),
            "los_predicted_days": random.randint(7, 21),
        }
    if acuity == "HIGH":
        return {
            "readmission_30d": round(random.uniform(0.45, 0.70), 3),
            "deterioration":   round(random.uniform(0.30, 0.60), 3),
            "no_show":         round(random.uniform(0.10, 0.25), 3),
            "discharge_today": round(random.uniform(0.05, 0.15), 3),
            "discharge_tomorrow": round(random.uniform(0.15, 0.35), 3),
            "discharge_this_week": round(random.uniform(0.40, 0.70), 3),
            "sepsis_risk":     round(random.uniform(0.15, 0.40), 3),
            "los_predicted_days": random.randint(3, 10),
        }
    if acuity == "MEDIUM":
        return {
            "readmission_30d": round(random.uniform(0.20, 0.45), 3),
            "deterioration":   round(random.uniform(0.10, 0.30), 3),
            "no_show":         round(random.uniform(0.15, 0.35), 3),
            "discharge_today": round(random.uniform(0.15, 0.35), 3),
            "discharge_tomorrow": round(random.uniform(0.30, 0.55), 3),
            "discharge_this_week": round(random.uniform(0.65, 0.90), 3),
            "sepsis_risk":     round(random.uniform(0.03, 0.15), 3),
            "los_predicted_days": random.randint(2, 6),
        }
    return {
        "readmission_30d": round(random.uniform(0.05, 0.20), 3),
        "deterioration":   round(random.uniform(0.02, 0.10), 3),
        "no_show":         round(random.uniform(0.20, 0.45), 3),
        "discharge_today": round(random.uniform(0.35, 0.65), 3),
        "discharge_tomorrow": round(random.uniform(0.55, 0.80), 3),
        "discharge_this_week": round(random.uniform(0.85, 0.98), 3),
        "sepsis_risk":     round(random.uniform(0.01, 0.05), 3),
        "los_predicted_days": random.randint(1, 3),
    }

def generate_vitals_history(acuity, hours=24):
    """Generate 24h of vitals at 1h intervals."""
    vitals = []
    if acuity == "CRITICAL":
        base = {"hr": 105, "sbp": 88, "dbp": 55, "spo2": 91, "rr": 24, "temp": 38.8, "map": 66}
    elif acuity == "HIGH":
        base = {"hr": 95,  "sbp": 105, "dbp": 65, "spo2": 94, "rr": 20, "temp": 38.2, "map": 78}
    elif acuity == "MEDIUM":
        base = {"hr": 82,  "sbp": 122, "dbp": 74, "spo2": 96, "rr": 17, "temp": 37.5, "map": 90}
    else:
        base = {"hr": 72,  "sbp": 128, "dbp": 78, "spo2": 98, "rr": 14, "temp": 37.0, "map": 95}

    for i in range(hours):
        t = NOW - timedelta(hours=hours - i)
        trend = i / hours
        vitals.append({
            "timestamp": t.isoformat(),
            "hr":   round(base["hr"]  + np.random.normal(0, 6) + (trend * 5 if acuity in ("CRITICAL","HIGH") else -trend * 3), 0),
            "sbp":  round(base["sbp"] + np.random.normal(0, 8), 0),
            "dbp":  round(base["dbp"] + np.random.normal(0, 5), 0),
            "spo2": round(min(100, base["spo2"] + np.random.normal(0, 1.5) + trend * 2), 1),
            "rr":   round(base["rr"]  + np.random.normal(0, 2), 0),
            "temp": round(base["temp"] + np.random.normal(0, 0.2) - trend * 0.3, 1),
            "map":  round(base["map"] + np.random.normal(0, 6), 0),
            "pain": random.randint(0, 10) if acuity in ("CRITICAL", "HIGH") else random.randint(0, 6),
        })
    return vitals

def generate_labs(unit, acuity):
    labs = []
    test_count = random.randint(3, 8)
    selected = random.sample(LAB_TESTS, min(test_count, len(LAB_TESTS)))
    for test in selected:
        ordered_ago = random.randint(1, 8)
        result_ago = ordered_ago - random.randint(0, 1)
        status = "resulted" if result_ago > 0 else "pending"
        abnormal = acuity in ("CRITICAL", "HIGH") and random.random() < 0.6
        labs.append({
            "id": str(uuid.uuid4())[:8],
            "test": test,
            "status": status,
            "ordered_at": (NOW - timedelta(hours=ordered_ago)).isoformat(),
            "resulted_at": (NOW - timedelta(hours=max(0, result_ago))).isoformat() if status == "resulted" else None,
            "value": _lab_value(test, abnormal),
            "abnormal": abnormal,
            "critical": abnormal and random.random() < 0.3,
        })
    return labs

def _lab_value(test, abnormal):
    ranges = {
        "CBC":        ("WBC 12.4, Hgb 8.2, Plt 142" if abnormal else "WBC 7.8, Hgb 13.5, Plt 220"),
        "BMP":        ("Na 128, K 5.8, Cr 2.4, BUN 42" if abnormal else "Na 139, K 4.1, Cr 0.9, BUN 18"),
        "CMP":        ("ALT 180, AST 210, Total Bili 3.2" if abnormal else "ALT 28, AST 24, Total Bili 0.8"),
        "Troponin":   ("2.84 ng/mL [H]" if abnormal else "< 0.01 ng/mL"),
        "BNP":        ("1840 pg/mL [H]" if abnormal else "45 pg/mL"),
        "Lactate":    ("4.2 mmol/L [H]" if abnormal else "1.1 mmol/L"),
        "Blood Culture": ("Gram+ cocci in clusters" if abnormal else "No growth x48h"),
        "UA":         ("WBC >50, bacteria >50, nitrite+" if abnormal else "Clear, no infection"),
        "Coagulation Panel": ("PT 22, INR 2.1, PTT 68" if abnormal else "PT 12, INR 1.0, PTT 28"),
        "ABG":        ("pH 7.28, pCO2 52, pO2 58, HCO3 18" if abnormal else "pH 7.41, pCO2 40, pO2 95"),
        "Lipase":     ("1840 U/L [H]" if abnormal else "42 U/L"),
        "LFTs":       ("ALT 340, AST 280, AlkPhos 180" if abnormal else "ALT 22, AST 18, AlkPhos 72"),
        "TSH":        ("0.02 mIU/L [L]" if abnormal else "2.1 mIU/L"),
    }
    return ranges.get(test, "Pending")

def generate_orders(unit, acuity):
    orders = []
    med_count = random.randint(3, 8)
    for med in random.sample(MEDICATIONS, min(med_count, len(MEDICATIONS))):
        orders.append({
            "id": str(uuid.uuid4())[:8],
            "type": "medication",
            "name": med,
            "dose": _med_dose(med),
            "frequency": random.choice(["Q4H", "Q6H", "Q8H", "Q12H", "Daily", "BID", "PRN"]),
            "route": random.choice(["IV", "PO", "SubQ", "IM"]),
            "status": random.choice(["active", "active", "active", "pending", "completed"]),
            "ordered_by": random.choice(["Dr. Chen", "Dr. Williams", "Dr. Patel", "Dr. O'Brien"]),
            "ordered_at": (NOW - timedelta(hours=random.randint(1, 12))).isoformat(),
        })
    img_count = random.randint(0, 3)
    for img in random.sample(IMAGING, min(img_count, len(IMAGING))):
        orders.append({
            "id": str(uuid.uuid4())[:8],
            "type": "imaging",
            "name": img,
            "priority": random.choice(["STAT", "ROUTINE", "URGENT"]),
            "status": random.choice(["completed", "pending", "in-progress"]),
            "ordered_by": random.choice(["Dr. Chen", "Dr. Williams", "Dr. Patel"]),
            "ordered_at": (NOW - timedelta(hours=random.randint(1, 6))).isoformat(),
            "result": "Completed — see radiology report" if random.random() > 0.4 else None,
        })
    return orders

def _med_dose(med):
    doses = {
        "Metoprolol": "25mg", "Lisinopril": "10mg", "Aspirin": "81mg",
        "Heparin": "5000 units", "Vancomycin": "1.25g", "Furosemide": "40mg",
        "Morphine": "2mg", "Ondansetron": "4mg", "Pantoprazole": "40mg",
        "Insulin": "per sliding scale", "Norepinephrine": "0.1 mcg/kg/min",
        "Amiodarone": "200mg", "Warfarin": "5mg",
        "Piperacillin-Tazobactam": "3.375g",
    }
    return doses.get(med, "Standard dose")

def generate_notes(acuity):
    templates = [
        "Patient admitted for {dx}. Hemodynamically {stability}. Continue current management.",
        "Vitals {stability} overnight. Patient {comfort}. Plan to {plan}.",
        "Discussed goals of care with patient and family. {goal}.",
        "Responding {response} to treatment. {next_step}.",
    ]
    return [{
        "id": str(uuid.uuid4())[:8],
        "type": random.choice(["Progress Note", "Nursing Note", "Consult Note"]),
        "author": random.choice(["Dr. Chen", "Dr. Williams", "RN Martinez", "RN Johnson"]),
        "timestamp": (NOW - timedelta(hours=random.randint(1, 8))).isoformat(),
        "text": random.choice(templates).format(
            dx="the above",
            stability="stable" if acuity in ("LOW", "MEDIUM") else "unstable",
            comfort="comfortable" if acuity == "LOW" else "in distress",
            plan="reassess in 4 hours",
            goal="comfort-focused care" if random.random() > 0.7 else "full resuscitation",
            response="well" if acuity in ("LOW", "MEDIUM") else "slowly",
            next_step="continue monitoring",
        ),
    } for _ in range(random.randint(2, 5))]

# ── Generate Census ───────────────────────────────────────────────────────────

def generate_census():
    census = {"units": {}, "summary": {}, "generated_at": NOW.isoformat()}
    total_beds = 0
    total_occupied = 0
    all_patients = []

    for unit_id, unit_cfg in UNITS.items():
        if unit_id == "OR":
            continue
        beds = []
        n_beds = unit_cfg["beds"]
        occupancy = 0.85 if unit_id in ("MICU", "CCU", "SICU") else 0.70
        n_occupied = int(n_beds * occupancy)

        for bed_num in range(1, n_beds + 1):
            bed_id = f"{unit_id}-{bed_num:02d}"
            if bed_num <= n_occupied:
                sex = random.choice(["M", "F"])
                age = random_age(unit_id)
                acuity = acuity_for_unit(unit_id)
                mrn = random_mrn()
                los = random.randint(1, 14)
                admit_dt = (NOW - timedelta(days=los, hours=random.randint(0, 23))).isoformat()
                dx = random.choice(DIAGNOSES.get(unit_id, DIAGNOSES["3EAST"]))
                scores = risk_scores_for_acuity(acuity)
                vitals_now = generate_vitals_history(acuity, 1)[0]

                patient = {
                    "mrn": mrn,
                    "name": random_name(sex),
                    "age": age,
                    "sex": sex,
                    "bed_id": bed_id,
                    "unit": unit_id,
                    "unit_name": unit_cfg["name"],
                    "room": f"{bed_num:02d}",
                    "acuity": acuity,
                    "diagnosis": dx,
                    "admit_datetime": admit_dt,
                    "los_days": los,
                    "attending": random.choice(SURGEONS),
                    "scores": scores,
                    "vitals_current": vitals_now,
                    "vitals_history": generate_vitals_history(acuity, 24),
                    "labs": generate_labs(unit_id, acuity),
                    "orders": generate_orders(unit_id, acuity),
                    "notes": generate_notes(acuity),
                    "alerts": _patient_alerts(acuity, dx, scores),
                    "code_status": random.choices(["Full Code", "DNR", "DNR/DNI"], weights=[70, 20, 10])[0],
                    "isolation": random.choices([None, "Contact", "Droplet", "Airborne"], weights=[70, 15, 10, 5])[0],
                    "diet": random.choice(["Regular", "Low Sodium", "Diabetic", "NPO", "Cardiac"]),
                    "allergies": random.sample(["PCN", "Sulfa", "Codeine", "Contrast", "Latex"], random.randint(0, 2)),
                }
                beds.append({"bed_id": bed_id, "status": "occupied", "patient": patient})
                all_patients.append(patient)
                total_occupied += 1
            else:
                beds.append({"bed_id": bed_id, "status": "empty", "patient": None})
            total_beds += 1

        census["units"][unit_id] = {
            "id": unit_id,
            "name": unit_cfg["name"],
            "color": unit_cfg["color"],
            "total_beds": n_beds,
            "occupied": n_occupied,
            "available": n_beds - n_occupied,
            "beds": beds,
        }

    census["summary"] = {
        "total_beds": total_beds,
        "occupied": total_occupied,
        "available": total_beds - total_occupied,
        "occupancy_pct": round(total_occupied / total_beds * 100, 1),
        "critical_count": sum(1 for p in all_patients if p["acuity"] == "CRITICAL"),
        "high_count": sum(1 for p in all_patients if p["acuity"] == "HIGH"),
        "medium_count": sum(1 for p in all_patients if p["acuity"] == "MEDIUM"),
        "low_count": sum(1 for p in all_patients if p["acuity"] == "LOW"),
        "discharge_today": sum(1 for p in all_patients if p["scores"]["discharge_today"] > 0.5),
        "discharge_tomorrow": sum(1 for p in all_patients if p["scores"]["discharge_tomorrow"] > 0.5),
        "sepsis_alerts": sum(1 for p in all_patients if p["scores"]["sepsis_risk"] > 0.4),
        "pending_labs": sum(1 for p in all_patients for l in p["labs"] if l["status"] == "pending"),
        "critical_labs": sum(1 for p in all_patients for l in p["labs"] if l.get("critical")),
        "avg_los": round(sum(p["los_days"] for p in all_patients) / max(len(all_patients), 1), 1),
    }

    return census

def _patient_alerts(acuity, dx, scores):
    alerts = []
    if scores["sepsis_risk"] > 0.4:
        alerts.append({"type": "SEPSIS", "severity": "CRITICAL", "message": "Sepsis bundle due", "time": NOW.isoformat()})
    if scores["deterioration"] > 0.6:
        alerts.append({"type": "DETERIORATION", "severity": "HIGH", "message": "Deterioration risk elevated", "time": NOW.isoformat()})
    if scores["discharge_today"] > 0.6:
        alerts.append({"type": "DISCHARGE", "severity": "INFO", "message": "Discharge predicted today", "time": NOW.isoformat()})
    if acuity == "CRITICAL" and random.random() > 0.5:
        alerts.append({"type": "CRITICAL_LAB", "severity": "CRITICAL", "message": "Critical lab value requires attention", "time": NOW.isoformat()})
    return alerts

# ── Generate OR Schedule ──────────────────────────────────────────────────────

def generate_or_schedule():
    schedule = {"date": NOW.strftime("%Y-%m-%d"), "rooms": [], "summary": {}}
    total_cases = 0
    on_time = 0
    in_progress = 0
    completed = 0

    for room_num in range(1, 11):
        room_id = f"OR-{room_num:02d}"
        n_cases = random.randint(2, 5)
        cases = []
        current_time = NOW.replace(hour=7, minute=0, second=0)

        for case_num in range(n_cases):
            duration = random.randint(60, 240)
            scheduled_start = current_time + timedelta(minutes=case_num * 30)
            actual_start = scheduled_start + timedelta(minutes=random.randint(-5, 35))
            actual_end = actual_start + timedelta(minutes=duration)

            if actual_end < NOW:
                status = "completed"
                completed += 1
            elif actual_start < NOW:
                status = "in-progress"
                in_progress += 1
            else:
                status = "scheduled"

            delay_min = max(0, int((actual_start - scheduled_start).total_seconds() / 60))
            if delay_min < 10:
                on_time += 1

            sex = random.choice(["M", "F"])
            age = int(np.clip(np.random.normal(58, 15), 18, 90))

            cases.append({
                "case_id": f"CASE-{room_num:02d}-{case_num+1:02d}",
                "procedure": random.choice(OR_PROCEDURES),
                "surgeon": random.choice(SURGEONS),
                "anesthesiologist": random.choice(ANESTHESIOLOGISTS),
                "patient_mrn": random_mrn(),
                "patient_name": random_name(sex),
                "patient_age": age,
                "scheduled_start": scheduled_start.isoformat(),
                "actual_start": actual_start.isoformat() if status != "scheduled" else None,
                "estimated_end": actual_end.isoformat(),
                "duration_min": duration,
                "status": status,
                "delay_min": delay_min,
                "case_class": random.choice(["Elective", "Urgent", "Emergency"]),
            })
            current_time = actual_end
            total_cases += 1

        room_status = "available"
        for c in cases:
            if c["status"] == "in-progress":
                room_status = "in-use"
                break
        if all(c["status"] == "completed" for c in cases):
            room_status = "turnover"

        schedule["rooms"].append({
            "room_id": room_id,
            "name": f"Operating Room {room_num}",
            "status": room_status,
            "cases": cases,
        })

    schedule["summary"] = {
        "total_cases": total_cases,
        "completed": completed,
        "in_progress": in_progress,
        "scheduled": total_cases - completed - in_progress,
        "on_time_pct": round(on_time / max(total_cases, 1) * 100, 1),
        "avg_delay_min": random.randint(8, 22),
        "rooms_in_use": in_progress,
        "rooms_available": 10 - in_progress,
    }

    return schedule

# ── Generate ADT Events ───────────────────────────────────────────────────────

def generate_adt_events(n=50):
    events = []
    event_types = ["ADMIT", "DISCHARGE", "TRANSFER", "UPDATE"]
    weights = [40, 25, 20, 15]

    for i in range(n):
        event_type = random.choices(event_types, weights=weights)[0]
        unit = random.choice(list(UNITS.keys()))
        if unit == "OR":
            unit = "3EAST"
        sex = random.choice(["M", "F"])
        age = random_age(unit)
        acuity = acuity_for_unit(unit)
        t = NOW - timedelta(minutes=random.randint(0, 480))
        events.append({
            "event_id": str(uuid.uuid4())[:12],
            "event_type": event_type,
            "timestamp": t.isoformat(),
            "mrn": random_mrn(),
            "patient_name": random_name(sex),
            "age": age,
            "sex": sex,
            "unit": unit,
            "unit_name": UNITS[unit]["name"],
            "bed": f"{unit}-{random.randint(1, UNITS[unit]['beds']):02d}",
            "diagnosis": random.choice(DIAGNOSES.get(unit, DIAGNOSES["3EAST"])),
            "acuity": acuity,
            "attending": random.choice(SURGEONS),
        })

    events.sort(key=lambda x: x["timestamp"], reverse=True)
    return events

# ── Generate Operational Metrics ──────────────────────────────────────────────

def generate_operational_metrics():
    return {
        "generated_at": NOW.isoformat(),
        "ed_metrics": {
            "door_to_doc_min": random.randint(18, 45),
            "door_to_doc_target": 30,
            "lwbs_count": random.randint(2, 8),
            "boarding_count": random.randint(3, 12),
            "avg_los_hours": round(random.uniform(3.5, 6.5), 1),
            "patients_waiting": random.randint(8, 24),
            "triage_level_counts": {
                "ESI-1": random.randint(1, 3),
                "ESI-2": random.randint(3, 8),
                "ESI-3": random.randint(8, 15),
                "ESI-4": random.randint(5, 12),
                "ESI-5": random.randint(2, 6),
            },
        },
        "staffing": {
            "nurses_on_duty": random.randint(42, 58),
            "nurses_needed": 55,
            "physicians_on_duty": random.randint(18, 28),
            "open_shifts": random.randint(2, 8),
            "nurse_patient_ratio": round(random.uniform(3.2, 4.8), 1),
        },
        "throughput": {
            "admissions_today": random.randint(18, 35),
            "discharges_today": random.randint(15, 30),
            "transfers_in_today": random.randint(3, 10),
            "transfers_out_today": random.randint(2, 8),
            "avg_discharge_time": f"{random.randint(11, 14)}:{random.choice(['00','15','30','45'])}",
        },
        "quality": {
            "hapi_rate": round(random.uniform(0.8, 2.1), 2),
            "fall_rate": round(random.uniform(1.2, 3.4), 2),
            "cauti_rate": round(random.uniform(0.5, 1.8), 2),
            "clabsi_rate": round(random.uniform(0.3, 1.2), 2),
            "readmission_30d_rate": round(random.uniform(12.5, 18.2), 1),
            "patient_satisfaction": random.randint(78, 94),
        },
        "hourly_admissions": [
            {"hour": f"{h:02d}:00", "count": random.randint(0, 8)}
            for h in range(24)
        ],
        "capacity_trend": [
            {
                "time": (NOW - timedelta(hours=23-h)).strftime("%H:%M"),
                "occupancy": random.randint(62, 92),
            }
            for h in range(24)
        ],
    }

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Generating hospital data...")

    print("  → Census...")
    census = generate_census()
    with open(DATA_DIR / "census.json", "w") as f:
        json.dump(census, f, indent=2, default=str)
    print(f"     {census['summary']['occupied']} patients across {len(census['units'])} units")

    print("  → OR schedule...")
    or_schedule = generate_or_schedule()
    with open(DATA_DIR / "or_schedule.json", "w") as f:
        json.dump(or_schedule, f, indent=2, default=str)
    print(f"     {or_schedule['summary']['total_cases']} cases, {or_schedule['summary']['in_progress']} in progress")

    print("  → ADT events...")
    adt_events = generate_adt_events(100)
    with open(DATA_DIR / "adt_events.json", "w") as f:
        json.dump(adt_events, f, indent=2, default=str)
    print(f"     {len(adt_events)} events")

    print("  → Operational metrics...")
    metrics = generate_operational_metrics()
    with open(DATA_DIR / "operational_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"\nData saved to {DATA_DIR}/")
    print(f"Summary:")
    s = census["summary"]
    print(f"  Beds: {s['occupied']}/{s['total_beds']} occupied ({s['occupancy_pct']}%)")
    print(f"  Critical: {s['critical_count']} | High: {s['high_count']} | Medium: {s['medium_count']} | Low: {s['low_count']}")
    print(f"  Discharge today: {s['discharge_today']} | Tomorrow: {s['discharge_tomorrow']}")
    print(f"  Sepsis alerts: {s['sepsis_alerts']}")


if __name__ == "__main__":
    main()
