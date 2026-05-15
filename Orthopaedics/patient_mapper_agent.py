from __future__ import annotations

# Branch migration note:
# This file is intentionally self-contained so it can be copied into an
# independent orthopedics branch and renamed to `patient_mapper_agent.py`.

import json
import os
import re
import socket
import ssl
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

from llm_runtime import get_api_key_for_model_key, get_gateway_url


KB_PATH = Path(__file__).with_name("orthopaedics_kb.json")
GENAI_API_URL = get_gateway_url()


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model: str


MODEL_CONFIGS = {
    "gpt_5_2": ModelConfig(
        name="deepseek-v3:671b",
        model="deepseek-v3:671b",
    ),
    "deepseek_v3_2": ModelConfig(
        name="deepseek-v3:671b",
        model="deepseek-v3:671b",
    ),
    "deepseek_r1": ModelConfig(
        name="deepseek-r1:671b",
        model="deepseek-r1:671b",
    ),
    "qwen3": ModelConfig(
        name="deepseek-v3:671b",
        model="deepseek-v3:671b",
    ),
    "qwen3_vl": ModelConfig(
        name="deepseek-v3:671b",
        model="deepseek-v3:671b",
    ),
}


TASK_MODEL_SELECTION = {
    "patient_structuring": "deepseek_v3_2",
    "entity_linking": "deepseek_v3_2",
    "differential_ranking": "deepseek_v3_2",
}

ORTHOPEDICS_TASK_MODEL_SELECTION = TASK_MODEL_SELECTION


PATIENT_FIELD_ALIASES = {
    "age": ["age", "年龄", "岁数"],
    "sex": ["sex", "gender", "性别"],
    "weight_kg": ["weight", "weight_kg", "体重", "体重kg", "公斤"],
    "height_cm": ["height", "height_cm", "身高", "身高cm"],
    "chief_complaint": ["chief_complaint", "complaint", "主诉"],
    "symptom_duration": ["symptom_duration", "病程", "症状时长", "持续时间"],
    "pain_site": ["pain_site", "疼痛部位", "部位"],
    "pain_severity": ["pain_severity", "疼痛评分", "疼痛程度"],
    "laterality": ["laterality", "侧别", "患侧"],
    "trauma_history": ["trauma_history", "受伤史", "外伤史"],
    "onset_mechanism": ["onset_mechanism", "受伤机制", "发病机制"],
    "functional_limitation": ["functional_limitation", "功能受限", "活动受限"],
    "associated_symptoms": ["associated_symptoms", "伴随症状"],
    "red_flag_symptoms": ["red_flag_symptoms", "红旗症状", "危险信号"],
    "neurologic_symptoms": ["neurologic_symptoms", "神经症状", "麻木无力"],
    "fever_or_infection_signs": ["fever_or_infection_signs", "感染征象", "发热情况"],
    "wound_status": ["wound_status", "伤口情况", "切口情况"],
    "family_history": ["family_history", "家族史"],
    "allergies": ["allergies", "drug_allergies", "过敏史", "药物过敏"],
    "comorbidities": ["comorbidities", "past_history", "既往史", "基础疾病", "合并症"],
    "medications": ["medications", "用药", "长期用药"],
    "anticoagulants": ["anticoagulants", "抗凝药", "抗血小板药"],
    "smoking_status": ["smoking_status", "吸烟史"],
    "alcohol_use": ["alcohol_use", "饮酒史"],
    "osteoporosis_history": ["osteoporosis_history", "骨质疏松史"],
    "prior_orthopedic_history": ["prior_orthopedic_history", "既往骨科史", "既往骨折手术史"],
    "prior_surgeries": ["prior_surgeries", "既往手术史"],
    "assistive_device": ["assistive_device", "助行器", "辅具使用"],
    "procedure_name": ["procedure", "surgery", "operation", "手术", "术式", "拟行手术"],
    "procedure_site": ["procedure_site", "surgical_site", "手术部位"],
    "asa_hint": ["asa", "asa_class", "asa_hint", "ASA", "ASA分级", "麻醉分级"],
    "urgency": ["urgency", "emergency", "急诊", "择期急诊"],
    "labs": ["labs", "检验", "实验室", "实验室检查"],
    "vitals": ["vitals", "生命体征"],
    "imaging": ["imaging", "影像", "影像学检查"]
}


class OrthopedicsKnowledgeBase:
    def __init__(self, path: Path = KB_PATH) -> None:
        self.path = path
        self.data = self._load()

    def _load(self) -> dict[str, Any]:
        with self.path.open("r", encoding="utf-8-sig") as f:
            return json.load(f)

    def get_asa(self, asa_id: str) -> dict[str, Any] | None:
        for item in self.data.get("asa_physical_status", []):
            if item["id"] == asa_id:
                return item
        return None

    def get_pathway(self, pathway_id: str) -> dict[str, Any] | None:
        for item in self.data.get("orthopedic_pathways", []):
            if item["id"] == pathway_id:
                return item
        return None

    def get_condition(self, condition_id: str) -> dict[str, Any] | None:
        for item in self.data.get("orthopedic_conditions", []):
            if item["id"] == condition_id:
                return item
        return None

    def list_perioperative_rules(self) -> list[dict[str, Any]]:
        return self.data.get("perioperative_rules", [])

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": self.data["metadata"]["schema_version"],
            "asa_count": len(self.data.get("asa_physical_status", [])),
            "pathway_count": len(self.data.get("orthopedic_pathways", [])),
            "condition_count": len(self.data.get("orthopedic_conditions", [])),
            "workup_bundle_count": len(self.data.get("workup_bundles", [])),
            "rule_count": len(self.data.get("perioperative_rules", [])),
            "complication_count": len(self.data.get("complication_catalog", [])),
        }


class GenAIChatClient:
    def __init__(self, api_url: str = GENAI_API_URL, api_key: str | None = None) -> None:
        self.api_url = api_url
        self.api_key = str(api_key or "").strip()

    def chat_json(
        self,
        config: ModelConfig,
        prompt: str,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        api_key = self.api_key or get_api_key_for_model_key(
            next((key for key, candidate in MODEL_CONFIGS.items() if candidate == config), "deepseek_v3_2")
        )
        if not api_key:
            raise ValueError("Missing DeepSeek API key environment variable for Orthopaedics.")
        payload = {
            "model": config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ],
            "temperature": temperature,
            "n": 1,
            "stream": False,
            "presence_penalty": 0,
            "frequency_penalty": 0,
        }
        req = request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "accept": "application/json",
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with request.urlopen(req, timeout=60) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(
                    f"Orthopaedics API HTTP {exc.code}: {exc.reason}. Response: {body[:1200]}"
                )
                if exc.code == 429 and attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                break
            except (error.URLError, ssl.SSLError, TimeoutError, socket.timeout) as exc:
                last_error = exc
                if attempt == 2:
                    break
                time.sleep(1.5 * (attempt + 1))

        raise RuntimeError(
            f"Orthopaedics API request failed after retries: {last_error}"
        ) from last_error

    def extract_text(self, response: dict[str, Any]) -> str:
        choices = response.get("choices", [])
        if not choices:
            error_payload = response.get("error")
            if isinstance(error_payload, dict):
                code = str(error_payload.get("code") or "").strip()
                message = str(error_payload.get("message") or error_payload).strip()
                raise ValueError(
                    f"API response does not contain choices. error_code={code or 'unknown'} message={message}"
                )
            raise ValueError(f"API response does not contain choices. keys={list(response.keys())[:8]}")
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ).strip()
        raise ValueError("Unsupported API response format.")

    def extract_json(self, response: dict[str, Any]) -> dict[str, Any]:
        return self._parse_json_text(self.extract_text(response))

    def _parse_json_text(self, text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char not in "[{":
                continue
            try:
                payload, _ = decoder.raw_decode(text[index:])
                if isinstance(payload, dict):
                    return payload
            except json.JSONDecodeError:
                continue

        raise ValueError(f"Model did not return valid JSON. Raw output excerpt: {text[:500]}")


class OrthopedicsPatientMapperAgent:
    def __init__(
        self,
        kb: OrthopedicsKnowledgeBase,
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
            "chief_complaint": None,
            "symptom_duration": None,
            "pain_site": None,
            "pain_severity": None,
            "laterality": None,
            "trauma_history": None,
            "onset_mechanism": None,
            "functional_limitation": None,
            "associated_symptoms": [],
            "red_flag_symptoms": [],
            "neurologic_symptoms": None,
            "fever_or_infection_signs": None,
            "wound_status": None,
            "family_history": [],
            "allergies": [],
            "comorbidities": [],
            "medications": [],
            "anticoagulants": [],
            "smoking_status": None,
            "alcohol_use": None,
            "osteoporosis_history": None,
            "prior_orthopedic_history": [],
            "prior_surgeries": [],
            "assistive_device": None,
            "procedure_name": None,
            "procedure_site": None,
            "asa_hint": None,
            "urgency": None,
            "labs": {},
            "vitals": {},
            "imaging": {},
            "raw_input": raw_patient,
        }
        for canonical_key, aliases in PATIENT_FIELD_ALIASES.items():
            value = self._find_first_matching_value(raw_patient, aliases)
            if value is None:
                continue
            normalized[canonical_key] = self._coerce_value(canonical_key, value)
        normalized["bmi"] = self._compute_bmi(
            normalized.get("height_cm"),
            normalized.get("weight_kg"),
        )
        return normalized

    def build_case_record(self, raw_patient: dict[str, Any]) -> dict[str, Any]:
        normalized = self.normalize_patient_input(raw_patient)
        kb_matches = self.match_patient_to_kb(normalized)
        return {"normalized_profile": normalized, "kb_matches": kb_matches}

    def build_case_record_from_text(self, raw_patient_text: str) -> dict[str, Any]:
        structured = self.call_patient_structuring_api(raw_patient_text)
        normalized = self.normalize_patient_input(structured)
        entity_linking = self.call_medical_entity_linking_api(normalized)
        kb_matches = self.match_patient_to_kb(normalized)
        ranking = self.call_differential_ranking_api(normalized, kb_matches)
        return {
            "structured_patient": structured,
            "normalized_profile": normalized,
            "entity_linking": entity_linking,
            "kb_matches": kb_matches,
            "ranking": ranking,
        }

    def match_patient_to_kb(self, normalized_patient: dict[str, Any]) -> dict[str, Any]:
        asa_match = self._map_asa_hint(normalized_patient.get("asa_hint"))
        pathway_ids = self._infer_candidate_pathway_ids(normalized_patient)
        condition_ids = self._infer_candidate_condition_ids(normalized_patient)
        rule_ids = self._infer_risk_rule_ids(normalized_patient)
        return {
            "asa_match": self.kb.get_asa(asa_match) if asa_match else None,
            "candidate_pathways": [
                self.kb.get_pathway(pathway_id)
                for pathway_id in pathway_ids
                if self.kb.get_pathway(pathway_id) is not None
            ],
            "suspected_conditions": [
                self.kb.get_condition(condition_id)
                for condition_id in condition_ids
                if self.kb.get_condition(condition_id) is not None
            ],
            "matched_perioperative_rules": [
                rule
                for rule in self.kb.list_perioperative_rules()
                if rule["id"] in rule_ids
            ],
        }

    def call_patient_structuring_api(self, raw_patient_text: str) -> dict[str, Any]:
        print("[API] patient_structuring start")
        prompt = (
            "You are a clinical orthopedics structuring assistant.\n"
            "Convert the patient description into a JSON object only.\n"
            "Use these keys when possible:\n"
            "age, sex, weight_kg, height_cm, chief_complaint, symptom_duration, pain_site, "
            "pain_severity, laterality, trauma_history, onset_mechanism, functional_limitation, "
            "associated_symptoms, red_flag_symptoms, neurologic_symptoms, fever_or_infection_signs, "
            "wound_status, family_history, allergies, comorbidities, medications, anticoagulants, "
            "smoking_status, alcohol_use, osteoporosis_history, prior_orthopedic_history, "
            "prior_surgeries, assistive_device, procedure_name, procedure_site, asa_hint, urgency, "
            "labs, vitals, imaging.\n"
            "Rules:\n"
            "- Output valid JSON only.\n"
            "- Do not hallucinate unavailable facts.\n"
            "- Arrays: associated_symptoms, red_flag_symptoms, family_history, allergies, "
            "comorbidities, medications, anticoagulants, prior_orthopedic_history, prior_surgeries.\n"
            "- Objects: labs, vitals, imaging.\n"
            "- Missing information must be null, [] or {}.\n\n"
            f"Patient text:\n{raw_patient_text}"
        )
        config = MODEL_CONFIGS[TASK_MODEL_SELECTION["patient_structuring"]]
        response = self.api_client.chat_json(config, prompt)
        return self.api_client.extract_json(response)

    def call_medical_entity_linking_api(self, normalized_patient: dict[str, Any]) -> dict[str, Any]:
        print("[API] entity_linking start")
        prompt = (
            "You are an orthopedic medical entity normalization assistant.\n"
            "Normalize the patient data into database-friendly concepts.\n"
            "Return JSON only with keys:\n"
            "normalized_terms, complaint_keywords, anatomical_keywords, red_flag_keywords, "
            "risk_keywords, suspected_condition_keywords, perioperative_keywords.\n\n"
            f"Patient JSON:\n{json.dumps(normalized_patient, ensure_ascii=False, indent=2)}"
        )
        config = MODEL_CONFIGS[TASK_MODEL_SELECTION["entity_linking"]]
        response = self.api_client.chat_json(config, prompt)
        return self._normalize_entity_linking_output(self.api_client.extract_json(response))

    def call_differential_ranking_api(
        self,
        normalized_patient: dict[str, Any],
        kb_matches: dict[str, Any],
    ) -> dict[str, Any]:
        print("[API] differential_ranking start")
        prompt = (
            "You are an orthopedic differential diagnosis ranking assistant.\n"
            "Given the normalized patient profile and KB matches, rank the most likely orthopedic "
            "conditions and pathways. Think like a senior orthopedic surgeon: consider trauma "
            "mechanism, weight-bearing ability, infection risk, neurovascular compromise, fragility "
            "fracture risk, tumor red flags, perioperative comorbidities and family history.\n"
            "Return JSON only with keys:\n"
            "ranked_condition_ids, ranked_pathway_ids, urgent_flags, reasons, missing_critical_information.\n\n"
            "Normalized patient:\n"
            f"{json.dumps(normalized_patient, ensure_ascii=False, indent=2)}\n\n"
            "KB matches:\n"
            f"{json.dumps(kb_matches, ensure_ascii=False, indent=2)}"
        )
        config = MODEL_CONFIGS[TASK_MODEL_SELECTION["differential_ranking"]]
        response = self.api_client.chat_json(config, prompt)
        return self.api_client.extract_json(response)

    def _find_first_matching_value(self, raw_patient: dict[str, Any], aliases: list[str]) -> Any | None:
        for alias in aliases:
            if alias in raw_patient:
                return raw_patient[alias]
        return None

    def _coerce_value(self, canonical_key: str, value: Any) -> Any:
        if canonical_key in {
            "associated_symptoms",
            "red_flag_symptoms",
            "family_history",
            "allergies",
            "comorbidities",
            "medications",
            "anticoagulants",
            "prior_orthopedic_history",
            "prior_surgeries",
        }:
            return self._coerce_list(value)
        if canonical_key in {"labs", "vitals", "imaging"}:
            return value if isinstance(value, dict) else {}
        return value

    def _coerce_list(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            text = value.replace("，", ",").replace("；", ",").replace("、", ",")
            return [item.strip() for item in text.split(",") if item.strip()]
        if value is None:
            return []
        return [str(value).strip()]

    def _normalize_entity_linking_output(self, payload: dict[str, Any]) -> dict[str, Any]:
        expected_list_keys = [
            "normalized_terms",
            "complaint_keywords",
            "anatomical_keywords",
            "red_flag_keywords",
            "risk_keywords",
            "suspected_condition_keywords",
            "perioperative_keywords",
        ]
        normalized = dict(payload)
        for key in expected_list_keys:
            normalized[key] = self._normalize_text_list(payload.get(key))
        return normalized

    def _normalize_text_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            text = value.replace("，", ",").replace("；", ",").replace("、", ",")
            return [item.strip() for item in text.split(",") if item.strip()]
        if isinstance(value, list):
            result: list[str] = []
            for item in value:
                result.extend(self._normalize_text_list(item))
            return result
        if isinstance(value, dict):
            result: list[str] = []
            for item in value.values():
                result.extend(self._normalize_text_list(item))
            return result
        return [str(value).strip()]

    def _compute_bmi(self, height_cm: Any, weight_kg: Any) -> float | None:
        try:
            if height_cm is None or weight_kg is None:
                return None
            height_m = float(height_cm) / 100.0
            if height_m <= 0:
                return None
            return round(float(weight_kg) / (height_m * height_m), 2)
        except (TypeError, ValueError):
            return None

    def _map_asa_hint(self, asa_hint: Any) -> str | None:
        if asa_hint is None:
            return None
        text = str(asa_hint).upper().replace(" ", "")
        if "ASA" not in text:
            text = f"ASA{text}"
        if "IV" in text:
            return "ASA_IV"
        if "III" in text:
            return "ASA_III"
        if "II" in text:
            return "ASA_II"
        if "I" in text:
            return "ASA_I"
        if "V" in text:
            return "ASA_V"
        return None

    def _combined_text(self, normalized_patient: dict[str, Any]) -> str:
        pieces = [
            normalized_patient.get("chief_complaint"),
            normalized_patient.get("symptom_duration"),
            normalized_patient.get("pain_site"),
            normalized_patient.get("laterality"),
            normalized_patient.get("trauma_history"),
            normalized_patient.get("onset_mechanism"),
            normalized_patient.get("functional_limitation"),
            normalized_patient.get("neurologic_symptoms"),
            normalized_patient.get("fever_or_infection_signs"),
            normalized_patient.get("wound_status"),
            normalized_patient.get("procedure_name"),
            normalized_patient.get("procedure_site"),
            normalized_patient.get("osteoporosis_history"),
            normalized_patient.get("smoking_status"),
            " ".join(normalized_patient.get("associated_symptoms", [])),
            " ".join(normalized_patient.get("red_flag_symptoms", [])),
            " ".join(normalized_patient.get("comorbidities", [])),
            " ".join(normalized_patient.get("family_history", [])),
            json.dumps(normalized_patient.get("imaging", {}), ensure_ascii=False),
            json.dumps(normalized_patient.get("labs", {}), ensure_ascii=False),
        ]
        return " ".join(str(piece) for piece in pieces if piece)

    def _infer_candidate_pathway_ids(self, normalized_patient: dict[str, Any]) -> list[str]:
        text = self._combined_text(normalized_patient)
        pathway_ids: list[str] = []
        if any(token in text for token in ["跌倒", "摔伤", "扭伤", "车祸", "外伤", "骨折", "脱位", "不能负重"]):
            pathway_ids.append("acute_trauma_fracture_dislocation")
        if any(token in text for token in ["骨关节炎", "退变", "慢性膝痛", "慢性髋痛", "关节置换"]):
            pathway_ids.append("degenerative_joint_disease")
        if any(token in text for token in ["腰痛", "腰腿痛", "麻木", "无力", "脊柱", "马尾", "大小便"]):
            pathway_ids.append("spine_neurologic_pathway")
        if any(token in text for token in ["发热", "感染", "渗液", "红肿热痛", "脓", "CRP升高", "ESR升高", "白细胞升高"]):
            pathway_ids.append("infection_inflammatory_pathway")
        if any(token in text for token in ["肿瘤", "癌", "夜间痛", "静息痛", "病理骨折", "消瘦"]):
            pathway_ids.append("tumor_pathologic_pathway")
        if not pathway_ids:
            if (normalized_patient.get("age") or 0) >= 50:
                pathway_ids.append("degenerative_joint_disease")
            else:
                pathway_ids.append("acute_trauma_fracture_dislocation")
        return list(dict.fromkeys(pathway_ids))

    def _infer_candidate_condition_ids(self, normalized_patient: dict[str, Any]) -> list[str]:
        text = self._combined_text(normalized_patient)
        age = normalized_patient.get("age") or 0
        condition_ids: list[str] = []
        if any(token in text for token in ["髋", "腹股沟", "股骨颈", "转子", "不能负重"]) and any(
            token in text for token in ["跌倒", "外伤", "X线阴性", "隐匿", "骨折"]
        ):
            condition_ids.append("hip_fracture_or_occult_hip_fracture")
        if any(token in text for token in ["膝骨关节炎", "膝OA", "膝痛", "上下楼", "晨僵"]):
            condition_ids.append("knee_osteoarthritis")
        if any(token in text for token in ["髋骨关节炎", "髋OA", "腹股沟痛", "内旋受限", "跛行"]):
            condition_ids.append("hip_osteoarthritis")
        if any(token in text for token in ["腰腿痛", "坐骨神经", "椎间盘", "神经根", "椎管狭窄"]):
            condition_ids.append("lumbar_disc_or_stenosis_with_radiculopathy")
        if any(token in text for token in ["尿潴留", "大小便", "鞍区", "双下肢无力", "马尾"]):
            condition_ids.append("cauda_equina_or_progressive_neurologic_compromise")
        if any(token in text for token in ["发热", "关节红肿热痛", "骨髓炎", "化脓", "脓毒", "CRP升高", "ESR升高", "白细胞升高"]):
            condition_ids.append("septic_arthritis_or_osteomyelitis")
        if any(token in text for token in ["假体", "内固定", "植入物", "窦道", "切口渗液", "松动"]):
            condition_ids.append("prosthetic_joint_or_implant_infection")
        if any(token in text for token in ["半月板", "前交叉", "后交叉", "交锁", "打软腿"]):
            condition_ids.append("ligament_or_meniscus_injury")
        if any(token in text for token in ["肿瘤", "病理骨折", "溶骨", "成骨", "消瘦", "转移"]):
            condition_ids.append("pathologic_fracture_or_bone_tumor")
        if any(token in text for token in ["筋膜室", "疼痛不成比例", "被动牵伸痛", "无脉", "肢体冰冷"]):
            condition_ids.append("compartment_syndrome_or_acute_limb_ischaemia")
        if not condition_ids:
            if age >= 65 and any(token in text for token in ["跌倒", "髋痛", "不能负重"]):
                condition_ids.append("hip_fracture_or_occult_hip_fracture")
            elif any(token in text for token in ["腰", "腿麻", "放射痛"]):
                condition_ids.append("lumbar_disc_or_stenosis_with_radiculopathy")
            else:
                condition_ids.append("knee_osteoarthritis")
        return list(dict.fromkeys(condition_ids))

    def _infer_risk_rule_ids(self, normalized_patient: dict[str, Any]) -> list[str]:
        text = self._combined_text(normalized_patient)
        age = normalized_patient.get("age") or 0
        rule_ids: list[str] = []
        if any(token in text for token in ["无脉", "冰冷", "被动牵伸痛", "麻木加重", "大小便", "鞍区", "进行性无力"]):
            rule_ids.append("urgent_neurovascular_review")
        if age >= 65 and any(token in text for token in ["跌倒", "髋痛", "不能负重", "X线阴性", "隐匿"]):
            rule_ids.append("occult_hip_fracture_imaging")
        if any(token in text for token in ["低能量", "骨质疏松", "脆性骨折", "家族髋部骨折史"]) or (
            age >= 50 and any(token in text for token in ["骨折", "跌倒"])
        ):
            rule_ids.append("fragility_fracture_secondary_prevention")
        if any(token in text for token in ["制动", "石膏", "支具", "髋关节置换", "膝关节置换", "骨折手术", "不能下地"]):
            rule_ids.append("vte_risk_review")
        if any(token in text for token in ["发热", "红肿", "渗液", "感染", "糖尿病", "切口问题"]):
            rule_ids.append("infection_prevention_review")
        if any(token in text for token in ["关节置换", "髋置换", "膝置换"]):
            rule_ids.append("joint_replacement_blood_loss_plan")
        if age >= 70 or any(token in text for token in ["衰弱", "跌倒", "认知", "谵妄", "营养不良", "助行器"]):
            rule_ids.append("geriatric_frailty_review")
        if normalized_patient.get("procedure_name") or normalized_patient.get("asa_hint"):
            rule_ids.extend(["preop_history_and_family_review", "preop_test_selection_review"])
        return list(dict.fromkeys(rule_ids))


OrthopaedicsKnowledgeBase = OrthopedicsKnowledgeBase
PatientProfileMapperAgent = OrthopedicsPatientMapperAgent


if __name__ == "__main__":
    kb = OrthopedicsKnowledgeBase()
    agent = OrthopedicsPatientMapperAgent(kb)
    raw_patient = {
        "年龄": 79,
        "性别": "女",
        "体重": 56,
        "身高": 158,
        "主诉": "跌倒后右髋疼痛、不能负重 6 小时",
        "疼痛部位": "右髋部",
        "受伤史": "家中跌倒",
        "受伤机制": "低能量跌倒",
        "功能受限": "无法站立和行走",
        "基础疾病": ["高血压", "2型糖尿病", "房颤"],
        "抗凝药": ["利伐沙班"],
        "骨质疏松史": "有",
        "家族史": ["母亲有髋部骨折史"],
        "拟行手术": "右侧股骨近端骨折手术评估",
        "ASA分级": "III",
        "影像": {"xray": "疑似股骨颈骨折，需进一步明确"},
        "急诊": "急诊",
    }
    print("Knowledge Base Summary")
    print(json.dumps(kb.summary(), ensure_ascii=False, indent=2))
    print("\nCase Mapping Result")
    print(json.dumps(agent.build_case_record(raw_patient), ensure_ascii=False, indent=2))
