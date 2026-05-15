from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from neurosurgery_shared import GenAIChatClient, MODEL_CONFIGS


KB_PATH = Path(__file__).with_name("neurosurgery_kb.json")

DEFAULT_NEUROSURGERY_KB: dict[str, Any] = {
    "syndrome_catalog": [],
    "red_flag_rules": [],
    "perioperative_risk_rules": [],
    "differential_templates": [],
    "workup_templates": [],
    "complication_watchlist_rules": [],
    "external_tools": [],
}

NEUROSURGERY_TASK_MODEL_SELECTION = {
    "patient_structuring": "deepseek_v3_2",
    "entity_linking": "deepseek_v3_2",
    "retrieval_reasoning": "deepseek_v3_2",
    "final_decision": "deepseek_v3_2",
}


NEUROSURGERY_PATIENT_FIELD_ALIASES = {
    "age": ["age", "年龄", "岁数"],
    "sex": ["sex", "gender", "性别"],
    "weight_kg": ["weight", "weight_kg", "体重", "公斤"],
    "height_cm": ["height", "height_cm", "身高"],
    "chief_complaint": ["chief_complaint", "chief complaint", "主诉"],
    "presenting_illness": ["presenting_illness", "现病史", "病情经过"],
    "suspected_diagnoses": ["diagnoses", "suspected_diagnoses", "诊断", "初步诊断", "拟诊"],
    "planned_procedure": ["planned_procedure", "procedure", "手术", "拟行手术", "拟行操作"],
    "procedure_site": ["procedure_site", "surgical_site", "手术部位", "病变部位"],
    "asa_hint": ["asa", "asa_hint", "ASA", "ASA分级", "麻醉分级"],
    "symptoms": ["symptoms", "症状", "伴随症状"],
    "neurological_exam": ["neurological_exam", "神经系统查体", "神经查体"],
    "comorbidities": ["comorbidities", "基础疾病", "合并症"],
    "past_history": ["past_history", "既往病史", "既往史"],
    "surgical_history": ["surgical_history", "既往手术史", "手术史"],
    "family_history": ["family_history", "家族史", "遗传病史"],
    "medication_history": ["medication_history", "长期用药", "用药史"],
    "allergies": ["allergies", "drug_allergies", "过敏史", "药物过敏"],
    "functional_status": ["functional_status", "功能状态", "活动耐量", "生活自理能力"],
    "seizure_history": ["seizure_history", "癫痫史", "发作史"],
    "headache_features": ["headache_features", "头痛特点", "头痛特征"],
    "visual_symptoms": ["visual_symptoms", "视觉症状", "视物模糊", "视力下降"],
    "cognitive_symptoms": ["cognitive_symptoms", "认知症状", "记忆下降", "精神行为变化"],
    "trauma_history": ["trauma_history", "外伤史", "头部外伤史"],
    "oncology_history": ["oncology_history", "肿瘤史", "恶性肿瘤史"],
    "vascular_history": ["vascular_history", "脑血管病史", "血管病史"],
    "device_history": ["device_history", "器械史", "分流管史", "植入物史"],
    "labs": ["labs", "实验室检查", "检验"],
    "vitals": ["vitals", "生命体征"],
    "imaging": ["imaging", "影像学检查", "影像"],
    "pathology": ["pathology", "病理", "病理结果"]
}


class NeurosurgeryKnowledgeBase:
    def __init__(self, path: Path = KB_PATH) -> None:
        self.path = path
        self.data = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return dict(DEFAULT_NEUROSURGERY_KB)

        content = self.path.read_text(encoding="utf-8-sig").strip()
        if not content:
            return dict(DEFAULT_NEUROSURGERY_KB)

        return json.loads(content)

    def list_syndromes(self) -> list[dict[str, Any]]:
        return self.data.get("syndrome_catalog", [])

    def list_red_flags(self) -> list[dict[str, Any]]:
        return self.data.get("red_flag_rules", [])

    def list_risk_rules(self) -> list[dict[str, Any]]:
        return self.data.get("perioperative_risk_rules", [])

    def list_differentials(self) -> list[dict[str, Any]]:
        return self.data.get("differential_templates", [])

    def list_workups(self) -> list[dict[str, Any]]:
        return self.data.get("workup_templates", [])

    def list_complications(self) -> list[dict[str, Any]]:
        return self.data.get("complication_watchlist_rules", [])

    def list_external_tools(self) -> list[dict[str, Any]]:
        return self.data.get("external_tools", [])

    def get_workup(self, workup_id: str) -> dict[str, Any] | None:
        for item in self.list_workups():
            if item["id"] == workup_id:
                return item
        return None


class NeurosurgeryPatientProfileMapperAgent:
    def __init__(
        self,
        kb: NeurosurgeryKnowledgeBase,
        api_client: GenAIChatClient | None = None,
    ) -> None:
        self.kb = kb
        self.api_client = api_client or GenAIChatClient()

    def normalize_patient_input(self, raw_patient: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {
            "age": None,
            "sex": None,
            "weight_kg": None,
            "height_cm": None,
            "bmi": None,
            "chief_complaint": None,
            "presenting_illness": None,
            "suspected_diagnoses": [],
            "planned_procedure": None,
            "procedure_site": None,
            "asa_hint": None,
            "symptoms": [],
            "neurological_exam": [],
            "comorbidities": [],
            "past_history": [],
            "surgical_history": [],
            "family_history": [],
            "medication_history": [],
            "allergies": [],
            "functional_status": None,
            "seizure_history": None,
            "headache_features": [],
            "visual_symptoms": [],
            "cognitive_symptoms": [],
            "trauma_history": None,
            "oncology_history": [],
            "vascular_history": [],
            "device_history": [],
            "labs": {},
            "vitals": {},
            "imaging": {},
            "pathology": {},
            "raw_input": raw_patient,
        }

        for canonical_key, aliases in NEUROSURGERY_PATIENT_FIELD_ALIASES.items():
            value = self._find_first_matching_value(raw_patient, aliases)
            if value is None:
                continue
            normalized[canonical_key] = self._coerce_value(canonical_key, value)

        normalized["bmi"] = self._calculate_bmi(
            normalized.get("weight_kg"),
            normalized.get("height_cm"),
        )
        return normalized

    def build_case_record(self, raw_patient: dict[str, Any]) -> dict[str, Any]:
        normalized = self.normalize_patient_input(raw_patient)
        kb_matches = self.match_patient_to_kb(normalized)
        return {
            "normalized_profile": normalized,
            "kb_matches": kb_matches,
        }

    def build_case_record_from_text(self, raw_patient_text: str) -> dict[str, Any]:
        structured = self.call_patient_structuring_api(raw_patient_text)
        normalized = self.normalize_patient_input(structured)
        entity_linking = self.call_medical_entity_linking_api(normalized)
        kb_matches = self.match_patient_to_kb(normalized, entity_linking)
        return {
            "structured_patient": structured,
            "normalized_profile": normalized,
            "entity_linking": entity_linking,
            "kb_matches": kb_matches,
        }

    def match_patient_to_kb(
        self,
        normalized_patient: dict[str, Any],
        entity_linking: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        text = self._build_patient_text(normalized_patient, entity_linking)
        matched_syndromes: list[dict[str, Any]] = []
        for syndrome in self.kb.list_syndromes():
            matched_keywords = [
                keyword
                for keyword in syndrome.get("trigger_keywords", [])
                if keyword.lower() in text
            ]
            if matched_keywords:
                matched_syndromes.append(
                    {
                        "syndrome_id": syndrome["id"],
                        "matched_keywords": matched_keywords,
                    }
                )

        matched_red_flags: list[str] = []
        for rule in self.kb.list_red_flags():
            if any(keyword.lower() in text for keyword in rule.get("if_any_keywords", [])):
                matched_red_flags.append(rule["id"])

        matched_risks: list[str] = []
        for rule in self.kb.list_risk_rules():
            if any(keyword.lower() in text for keyword in rule.get("if_any_keywords", [])):
                matched_risks.append(rule["id"])

        return {
            "matched_syndromes": matched_syndromes,
            "matched_red_flag_ids": list(dict.fromkeys(matched_red_flags)),
            "matched_risk_rule_ids": list(dict.fromkeys(matched_risks)),
        }

    def call_patient_structuring_api(self, raw_patient_text: str) -> dict[str, Any]:
        prompt = (
            "You are a neurosurgery preoperative structuring assistant.\n"
            "Convert the patient description into a JSON object only.\n"
            "Use these keys when possible: age, sex, weight_kg, height_cm, chief_complaint, "
            "presenting_illness, suspected_diagnoses, planned_procedure, procedure_site, asa_hint, "
            "symptoms, neurological_exam, comorbidities, past_history, surgical_history, "
            "family_history, medication_history, allergies, functional_status, seizure_history, "
            "headache_features, visual_symptoms, cognitive_symptoms, trauma_history, "
            "oncology_history, vascular_history, device_history, labs, vitals, imaging, pathology.\n"
            "Rules:\n"
            "- Output valid JSON only.\n"
            "- suspected_diagnoses, symptoms, neurological_exam, comorbidities, past_history, "
            "surgical_history, family_history, medication_history, allergies, headache_features, "
            "visual_symptoms, cognitive_symptoms, oncology_history, vascular_history, device_history "
            "must be arrays.\n"
            "- labs, vitals, imaging, pathology must be objects.\n"
            "- If information is missing, use null, [] or {}.\n\n"
            f"Patient text:\n{raw_patient_text}"
        )
        config = MODEL_CONFIGS[NEUROSURGERY_TASK_MODEL_SELECTION["patient_structuring"]]
        response = self.api_client.chat_json(config, prompt)
        return self.api_client.extract_json(response)

    def call_medical_entity_linking_api(
        self,
        normalized_patient: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = (
            "You are a neurosurgery entity-normalization assistant.\n"
            "Normalize the patient data into neurosurgery-friendly concepts.\n"
            "Return JSON only with keys:\n"
            "normalized_terms, syndrome_hints, lesion_hints, risk_keywords, differential_hints, "
            "workup_hints.\n\n"
            f"Patient JSON:\n{json.dumps(normalized_patient, ensure_ascii=False, indent=2)}"
        )
        config = MODEL_CONFIGS[NEUROSURGERY_TASK_MODEL_SELECTION["entity_linking"]]
        response = self.api_client.chat_json(config, prompt)
        return self.api_client.extract_json(response)

    def _find_first_matching_value(
        self,
        raw_patient: dict[str, Any],
        aliases: list[str],
    ) -> Any | None:
        for alias in aliases:
            if alias in raw_patient:
                return raw_patient[alias]
        return None

    def _coerce_value(self, canonical_key: str, value: Any) -> Any:
        list_keys = {
            "suspected_diagnoses",
            "symptoms",
            "neurological_exam",
            "comorbidities",
            "past_history",
            "surgical_history",
            "family_history",
            "medication_history",
            "allergies",
            "headache_features",
            "visual_symptoms",
            "cognitive_symptoms",
            "oncology_history",
            "vascular_history",
            "device_history",
        }
        dict_keys = {"labs", "vitals", "imaging", "pathology"}

        if canonical_key in list_keys:
            return self._coerce_list(value)
        if canonical_key in dict_keys:
            return self._coerce_object(value)
        return value

    def _coerce_list(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            text = (
                value.replace("，", ",")
                .replace("；", ",")
                .replace("、", ",")
                .replace("\n", ",")
            )
            return [item.strip() for item in text.split(",") if item.strip()]
        if value is None:
            return []
        return [str(value).strip()]

    def _coerce_object(self, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if value is None:
            return {}
        return {"summary": value}

    def _calculate_bmi(self, weight_kg: Any, height_cm: Any) -> float | None:
        try:
            if weight_kg in (None, "") or height_cm in (None, ""):
                return None
            height_m = float(height_cm) / 100.0
            if height_m <= 0:
                return None
            return round(float(weight_kg) / (height_m * height_m), 1)
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    def _build_patient_text(
        self,
        normalized_patient: dict[str, Any],
        entity_linking: dict[str, Any] | None,
    ) -> str:
        segments: list[str] = []
        for key in [
            "chief_complaint",
            "presenting_illness",
            "planned_procedure",
            "procedure_site",
            "functional_status",
            "seizure_history",
            "trauma_history",
            "asa_hint",
        ]:
            value = normalized_patient.get(key)
            if value:
                segments.append(str(value))

        for key in [
            "suspected_diagnoses",
            "symptoms",
            "neurological_exam",
            "comorbidities",
            "past_history",
            "surgical_history",
            "family_history",
            "medication_history",
            "allergies",
            "headache_features",
            "visual_symptoms",
            "cognitive_symptoms",
            "oncology_history",
            "vascular_history",
            "device_history",
        ]:
            segments.extend([str(item) for item in normalized_patient.get(key, [])])

        for key in ["labs", "vitals", "imaging", "pathology"]:
            value = normalized_patient.get(key, {})
            if value:
                segments.append(json.dumps(value, ensure_ascii=False))

        if entity_linking:
            segments.append(json.dumps(entity_linking, ensure_ascii=False))

        return " ".join(segments).lower()
