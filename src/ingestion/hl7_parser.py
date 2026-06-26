"""
HL7 v2.x ADT parser targeting Cerner Millennium message structure.
MSH.9 (message type) → array index 8
MSH.4 (sending facility) → array index 3
MSH.7 (datetime) → array index 6
PID.3 (patient identifiers) → CX list with MR/AN type routing
PV1.19 (visit number / encounter ID) → array index 19
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import structlog

log = structlog.get_logger(__name__)


class ADTEventType(str, Enum):
    ADMIT    = "A01"
    TRANSFER = "A02"
    DISCHARGE = "A03"
    UPDATE   = "A08"


@dataclass
class PatientIdentifiers:
    mrn: str
    encounter_id: str
    account_number: Optional[str] = None


@dataclass
class AdmissionData:
    patient: PatientIdentifiers
    event_type: ADTEventType
    admit_datetime: datetime
    facility: str
    location: str
    patient_class: str = "I"
    age: Optional[int] = None
    sex: Optional[str] = None
    language: Optional[str] = None
    admitting_diagnosis: Optional[str] = None
    attending_npi: Optional[str] = None
    raw_message: str = field(default="", repr=False)


class HL7ADTParser:
    _FIELD_SEP = "|"
    _COMP_SEP  = "^"
    _REP_SEP   = "~"

    def parse(self, raw: str) -> AdmissionData:
        segments = [s for s in raw.replace("\n", "\r").replace("\r\r", "\r").split("\r") if s.strip()]
        seg_map  = {s[:3]: s.split(self._FIELD_SEP) for s in segments}

        msh = seg_map.get("MSH", [])
        pid = seg_map.get("PID", [])
        pv1 = seg_map.get("PV1", [])
        evn = seg_map.get("EVN", [])
        dg1 = seg_map.get("DG1", [])

        # Event type — MSH.9 at array[8]
        trigger = self._get(msh, 8).split(self._COMP_SEP)
        code = trigger[1] if len(trigger) > 1 else trigger[0]
        try:
            event_type = ADTEventType(code)
        except ValueError:
            event_type = ADTEventType.UPDATE

        # Patient identifiers
        cx_list = self._get(pid, 3).split(self._REP_SEP)
        mrn = account = ""
        for cx in cx_list:
            parts = cx.split(self._COMP_SEP)
            id_type = parts[4] if len(parts) > 4 else ""
            if id_type == "MR":
                mrn = parts[0]
            elif id_type in ("AN", "FIN"):
                account = parts[0]
        if not mrn and cx_list:
            mrn = cx_list[0].split(self._COMP_SEP)[0]
        encounter_id = self._get(pv1, 19).split(self._COMP_SEP)[0]

        # Demographics
        age = None
        dob_raw = self._get(pid, 7)
        if dob_raw:
            try:
                dob = datetime.strptime(dob_raw[:8], "%Y%m%d")
                age = (datetime.now() - dob).days // 365
            except ValueError:
                pass

        # Datetime — EVN.2 or MSH.7 (array[6])
        raw_dt = self._get(evn, 2) or self._get(msh, 6)
        try:
            admit_dt = datetime.strptime(raw_dt[:14], "%Y%m%d%H%M%S")
        except (ValueError, TypeError):
            admit_dt = datetime.now(timezone.utc).replace(tzinfo=None)

        # Diagnosis
        dx_code = self._get(dg1, 3).split(self._COMP_SEP)[0]
        dx_desc = self._get(dg1, 4)
        dx = f"{dx_code}: {dx_desc}" if dx_code else None

        # Attending NPI — PV1.7, component[8]
        att_parts = self._get(pv1, 7).split(self._COMP_SEP)
        npi = att_parts[8] if len(att_parts) > 8 else None

        return AdmissionData(
            patient=PatientIdentifiers(mrn=mrn, encounter_id=encounter_id, account_number=account or None),
            event_type=event_type,
            admit_datetime=admit_dt,
            facility=self._get(msh, 3).split(self._COMP_SEP)[0],
            location=self._get(pv1, 3),
            patient_class=self._get(pv1, 2) or "I",
            age=age,
            sex=self._get(pid, 8) or None,
            language=self._get(pid, 15).split(self._COMP_SEP)[0] or None,
            admitting_diagnosis=dx,
            attending_npi=npi,
            raw_message=raw,
        )

    @staticmethod
    def _get(seg: list, idx: int, default: str = "") -> str:
        try:
            return seg[idx] or default
        except (IndexError, TypeError):
            return default
