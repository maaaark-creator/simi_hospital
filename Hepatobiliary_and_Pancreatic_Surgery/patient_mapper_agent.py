from __future__ import annotations

import json
import sys
import re
import socket
import ssl
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

_LLM_HELPER_ROOT = Path(__file__).resolve().parent.parent
if str(_LLM_HELPER_ROOT) not in sys.path:
    sys.path.insert(0, str(_LLM_HELPER_ROOT))

from llm_gateway import DEFAULT_LLM_GATEWAY_URL, build_auth_headers, get_llm_model_id


KB_PATH = Path(__file__).with_name("hepatobiliary_kb.json")
GENAI_API_URL = DEFAULT_LLM_GATEWAY_URL


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model: str


MODEL_CONFIGS = {
    "gpt_5_2": ModelConfig(
        name="gpt-4o",
        model=get_llm_model_id("gpt_5_2", "gpt-4o"),
    ),
    "deepseek_v3_2": ModelConfig(
        name="deepseek-chat",
        model=get_llm_model_id("deepseek_r1", "deepseek-chat"),
    ),
    "deepseek_r1": ModelConfig(
        name="deepseek-chat",
        model=get_llm_model_id("deepseek_r1", "deepseek-chat"),
    ),
    "qwen3": ModelConfig(
        name="qwen-max",
        model=get_llm_model_id("qwen3", "qwen-max"),
    ),
}


TASK_MODEL_SELECTION = {
    "patient_structuring": "gpt_5_2",
    "entity_linking": "gpt_5_2",
    "plan_ranking": "gpt_5_2",
}


PATIENT_FIELD_ALIASES = {
    "age": ["age", "年龄", "岁数"],
    "sex": ["sex", "gender", "性别"],
    "weight_kg": ["weight", "weight_kg", "体重", "公斤"],
    "height_cm": ["height", "height_cm", "身高", "身高cm"],
    "chief_complaint": ["chief_complaint", "cc", "主诉"],
    "symptoms": ["symptoms", "症状", "现病史症状"],
    "diagnosis_hint": ["diagnosis_hint", "诊断提示", "拟诊", "临床诊断", "初步诊断"],
    "procedure_name": ["procedure_name", "procedure", "手术", "术式", "拟行手术", "操作"],
    "procedure_site": ["procedure_site", "手术部位", "病变部位"],
    "comorbidities": ["comorbidities", "past_history", "既往史", "合并症", "基础疾病"],
    "medications": ["medications", "medication_history", "home_medications", "用药史", "长期用药", "常用药"],
    "allergies": ["allergies", "drug_allergies", "过敏史", "药物过敏"],
    "urgency": ["urgency", "emergency", "急诊", "紧急程度"],
    "vitals": ["vitals", "生命体征"],
    "labs": ["labs", "检验", "实验室", "化验"],
    "imaging_summary": ["imaging_summary", "imaging", "影像", "影像摘要", "CT", "MRI", "MRCP", "超声"],
    "pathology_summary": ["pathology_summary", "pathology", "病理", "病理摘要"],
    "drainage_status": ["drainage_status", "引流情况", "胆道引流", "腹腔引流"],
    "infection_status": ["infection_status", "感染情况", "发热感染", "脓毒症提示"],
    "bleeding_status": ["bleeding_status", "出血情况", "活动性出血"],
    "liver_function": ["liver_function", "肝功能摘要", "肝功能"],
    "postop_context": ["postop_context", "术后背景", "术后情况", "围手术期情况"],
}


class HepatobiliaryKnowledgeBase:
    def __init__(self, path: Path = KB_PATH) -> None:
        self.path = path
        self.data = self._load()

    def _load(self) -> dict[str, Any]:
        with self.path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def get_drug(self, drug_id: str) -> dict[str, Any] | None:
        for drug in self.data.get("drugs", []):
            if drug["id"] == drug_id:
                return drug
        return None

    def get_care_plan(self, plan_id: str) -> dict[str, Any] | None:
        for plan in self.data.get("care_plans", []):
            if plan["id"] == plan_id:
                return plan
        return None

    def get_medical_condition(self, condition_id: str) -> dict[str, Any] | None:
        for condition in self.data.get("medical_conditions", []):
            if condition["id"] == condition_id:
                return condition
        return None

    def get_surgical_condition(self, condition_id: str) -> dict[str, Any] | None:
        for condition in self.data.get("surgical_conditions", []):
            if condition["id"] == condition_id:
                return condition
        return None

    def list_safety_rules(self) -> list[dict[str, Any]]:
        return self.data.get("safety_rules", [])

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": self.data["metadata"]["schema_version"],
            "medical_condition_count": len(self.data.get("medical_conditions", [])),
            "surgical_condition_count": len(self.data.get("surgical_conditions", [])),
            "care_plan_count": len(self.data.get("care_plans", [])),
            "drug_count": len(self.data.get("drugs", [])),
            "rule_count": len(self.data.get("safety_rules", [])),
        }


class GenAIChatClient:
    def __init__(self, api_url: str = GENAI_API_URL) -> None:
        self.api_url = api_url

    def chat_json(
        self,
        config: ModelConfig,
        prompt: str,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
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
                **build_auth_headers(),
                "Content-Type": "application/json",
            },
            method="POST",
        )

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with request.urlopen(req, timeout=60) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except (error.URLError, ssl.SSLError, TimeoutError, socket.timeout) as exc:
                last_error = exc
                if attempt == 2:
                    break
                time.sleep(1.5 * (attempt + 1))

        raise RuntimeError(
            f"Hepatobiliary API request failed after retries: {last_error}"
        ) from last_error

    def extract_text(self, response: dict[str, Any]) -> str:
        choices = response.get("choices", [])
        if not choices:
            raise ValueError(f"API response does not contain choices. Response excerpt: {json.dumps(response, ensure_ascii=False)[:500]}")

        message = choices[0].get("message", {})
        content = message.get("content")

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
            return "".join(text_parts).strip()

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


class PatientProfileMapperAgent:
    def __init__(
        self,
        kb: HepatobiliaryKnowledgeBase,
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
            "symptoms": [],
            "diagnosis_hint": None,
            "procedure_name": None,
            "procedure_site": None,
            "comorbidities": [],
            "medications": [],
            "allergies": [],
            "urgency": None,
            "vitals": {},
            "labs": {},
            "imaging_summary": None,
            "pathology_summary": None,
            "drainage_status": None,
            "infection_status": None,
            "bleeding_status": None,
            "liver_function": None,
            "postop_context": None,
            "raw_input": raw_patient,
        }

        for canonical_key, aliases in PATIENT_FIELD_ALIASES.items():
            value = self._find_first_matching_value(raw_patient, aliases)
            if value is None:
                continue
            normalized[canonical_key] = self._coerce_value(canonical_key, value)

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
        ranking = self.call_plan_ranking_api(normalized, kb_matches)
        return {
            "structured_patient": structured,
            "normalized_profile": normalized,
            "entity_linking": entity_linking,
            "kb_matches": kb_matches,
            "ranking": ranking,
        }

    def match_patient_to_kb(self, normalized_patient: dict[str, Any]) -> dict[str, Any]:
        from hbp_kb_retriever import KnowledgeRetriever

        return KnowledgeRetriever(self.kb).retrieve(normalized_patient)

    def call_patient_structuring_api(self, raw_patient_text: str) -> dict[str, Any]:
        prompt = (
            "You are a hepatobiliary-pancreatic surgery clinical structuring assistant.\n"
            "Convert the patient description into a JSON object only.\n"
            "Use these keys when possible: "
            "age, sex, weight_kg, height_cm, chief_complaint, symptoms, diagnosis_hint, "
            "procedure_name, procedure_site, comorbidities, medications, allergies, urgency, "
            "vitals, labs, imaging_summary, pathology_summary, drainage_status, infection_status, "
            "bleeding_status, liver_function, postop_context.\n"
            "Rules:\n"
            "- Output valid JSON only.\n"
            "- symptoms, comorbidities, medications and allergies must be arrays.\n"
            "- vitals and labs must be objects.\n"
            "- If information is missing, use null, [] or {}.\n\n"
            f"Patient text:\n{raw_patient_text}"
        )
        config = MODEL_CONFIGS[TASK_MODEL_SELECTION["patient_structuring"]]
        return self.api_client.extract_json(self.api_client.chat_json(config, prompt))

    def call_medical_entity_linking_api(self, normalized_patient: dict[str, Any]) -> dict[str, Any]:
        prompt = (
            "You are a hepatobiliary-pancreatic surgery entity normalization assistant.\n"
            "Normalize the following patient data to database-friendly HBP concepts.\n"
            "Return JSON only with keys:\n"
            "normalized_terms, syndrome_keywords, symptom_keywords, imaging_keywords, "
            "procedure_keywords, risk_keywords, medication_keywords.\n\n"
            f"Patient JSON:\n{json.dumps(normalized_patient, ensure_ascii=False, indent=2)}"
        )
        config = MODEL_CONFIGS[TASK_MODEL_SELECTION["entity_linking"]]
        response = self.api_client.chat_json(config, prompt)
        return self._normalize_entity_linking_output(self.api_client.extract_json(response))

    def call_plan_ranking_api(self, normalized_patient: dict[str, Any], kb_matches: dict[str, Any]) -> dict[str, Any]:
        prompt = (
            "You are a hepatobiliary-pancreatic surgery decision-support ranking assistant.\n"
            "Given the normalized patient profile and candidate hepatobiliary plans, rank the plans.\n"
            "Return JSON only with keys:\n"
            "ranked_plan_ids, top_choice, reasons, caution_points.\n\n"
            "Normalized patient:\n"
            f"{json.dumps(normalized_patient, ensure_ascii=False, indent=2)}\n\n"
            "Candidate matches:\n"
            f"{json.dumps(kb_matches, ensure_ascii=False, indent=2)}"
        )
        config = MODEL_CONFIGS[TASK_MODEL_SELECTION["plan_ranking"]]
        return self.api_client.extract_json(self.api_client.chat_json(config, prompt))

    def _find_first_matching_value(self, raw_patient: dict[str, Any], aliases: list[str]) -> Any | None:
        for alias in aliases:
            if alias in raw_patient:
                return raw_patient[alias]
        return None

    def _coerce_value(self, canonical_key: str, value: Any) -> Any:
        if canonical_key in {"symptoms", "comorbidities", "medications", "allergies"}:
            return self._coerce_list(value)
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
        keys = [
            "normalized_terms",
            "syndrome_keywords",
            "symptom_keywords",
            "imaging_keywords",
            "procedure_keywords",
            "risk_keywords",
            "medication_keywords",
        ]
        normalized = dict(payload)
        for key in keys:
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


if __name__ == "__main__":
    kb = HepatobiliaryKnowledgeBase()
    agent = PatientProfileMapperAgent(kb)
    raw_patient = {
        "年龄": 63,
        "性别": "男",
        "主诉": "发热伴黄疸 2 天",
        "症状": ["右上腹痛", "寒战", "黄疸"],
        "诊断提示": "急性胆管炎，胆总管结石可能",
        "基础疾病": ["2 型糖尿病", "高血压"],
        "生命体征": {"BP": "88/54 mmHg", "HR": 118, "T": "39.1C"},
        "化验": {"TBil": "96 umol/L", "WBC": "18.2e9/L", "乳酸": "3.5 mmol/L"},
        "影像摘要": "MRCP 提示胆总管下段结石并胆道扩张",
        "急诊": "急诊",
    }
    print("Knowledge Base Summary")
    print(json.dumps(kb.summary(), ensure_ascii=False, indent=2))
    print("\nCase Mapping Result")
    print(json.dumps(agent.build_case_record(raw_patient), ensure_ascii=False, indent=2))


