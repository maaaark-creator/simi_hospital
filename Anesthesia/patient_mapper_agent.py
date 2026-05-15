from __future__ import annotations

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


KB_PATH = Path(__file__).with_name("anesthesia_kb.json")
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
    "plan_ranking": "deepseek_v3_2",
}


PATIENT_FIELD_ALIASES = {
    "age": ["age", "\u5e74\u9f84", "\u5c81\u6570"],
    "sex": ["sex", "gender", "\u6027\u522b"],
    "weight_kg": ["weight", "weight_kg", "\u4f53\u91cd", "\u4f53\u91cdkg", "\u516c\u65a4"],
    "height_cm": ["height", "height_cm", "\u8eab\u9ad8", "\u8eab\u9ad8cm"],
    "procedure_name": [
        "procedure_name",
        "procedure",
        "surgery",
        "operation",
        "\u624b\u672f",
        "\u672f\u5f0f",
        "\u62df\u884c\u624b\u672f",
    ],
    "procedure_site": ["procedure_site", "surgical_site", "\u624b\u672f\u90e8\u4f4d"],
    "asa_hint": ["asa", "asa_class", "asa_hint", "ASA", "ASA\u5206\u7ea7", "\u9ebb\u9189\u5206\u7ea7"],
    "allergies": ["allergies", "drug_allergies", "\u8fc7\u654f\u53f2", "\u836f\u7269\u8fc7\u654f"],
    "comorbidities": [
        "comorbidities",
        "past_history",
        "\u65e2\u5f80\u53f2",
        "\u5408\u5e76\u75c7",
        "\u57fa\u7840\u75be\u75c5",
    ],
    "medications": [
        "medications",
        "medication_history",
        "home_medications",
        "\u7528\u836f\u53f2",
        "\u957f\u671f\u7528\u836f",
        "\u5e38\u7528\u836f",
        "\u53e3\u670d\u836f",
    ],
    "fasting_status": ["fasting_status", "\u7981\u98df\u60c5\u51b5", "\u7981\u98df", "\u9971\u80c3\u60c5\u51b5"],
    "airway_notes": ["airway", "airway_notes", "\u6c14\u9053", "\u6c14\u9053\u8bc4\u4f30"],
    "urgency": ["urgency", "emergency", "\u6025\u8bca", "\u62e9\u671f\u6025\u8bca"],
    "labs": ["labs", "\u68c0\u9a8c", "\u5b9e\u9a8c\u5ba4", "\u5b9e\u9a8c\u5ba4\u68c0\u67e5"],
    "vitals": ["vitals", "\u751f\u547d\u4f53\u5f81"],
}


class AnesthesiaKnowledgeBase:
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

    def get_asa(self, asa_id: str) -> dict[str, Any] | None:
        for item in self.data.get("asa_physical_status", []):
            if item["id"] == asa_id:
                return item
        return None

    def get_plan(self, plan_id: str) -> dict[str, Any] | None:
        for plan in self.data.get("anesthesia_plans", []):
            if plan["id"] == plan_id:
                return plan
        return None

    def list_safety_rules(self) -> list[dict[str, Any]]:
        return self.data.get("safety_rules", [])

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": self.data["metadata"]["schema_version"],
            "drug_count": len(self.data.get("drugs", [])),
            "asa_count": len(self.data.get("asa_physical_status", [])),
            "plan_count": len(self.data.get("anesthesia_plans", [])),
            "rule_count": len(self.data.get("safety_rules", [])),
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
            raise ValueError("Missing DeepSeek API key environment variable for Anesthesia.")
        payload = {
            "model": config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ],
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
                with request.urlopen(req, timeout=120) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(
                    f"Anesthesia API HTTP {exc.code}: {exc.reason}. Response: {body[:1200]}"
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

        raise RuntimeError(f"Anesthesia API request failed after retries: {last_error}") from last_error

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
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
            return "".join(text_parts).strip()

        raise ValueError("Unsupported API response format.")

    def extract_json(self, response: dict[str, Any]) -> dict[str, Any]:
        text = self.extract_text(response)
        return self._parse_json_text(text)

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
        kb: AnesthesiaKnowledgeBase,
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
            "procedure_name": None,
            "procedure_site": None,
            "asa_hint": None,
            "allergies": [],
            "comorbidities": [],
            "medications": [],
            "fasting_status": None,
            "airway_notes": None,
            "urgency": None,
            "labs": {},
            "vitals": {},
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
        return {
            "normalized_profile": normalized,
            "kb_matches": kb_matches,
        }

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
        asa_match = self._map_asa_hint(normalized_patient.get("asa_hint"))
        candidate_plan_ids = self._infer_candidate_plan_ids(normalized_patient)
        safety_rule_ids = self._infer_safety_rule_ids(normalized_patient)

        candidate_plans = [
            self.kb.get_plan(plan_id)
            for plan_id in candidate_plan_ids
            if self.kb.get_plan(plan_id) is not None
        ]
        matched_rules = [
            rule
            for rule in self.kb.list_safety_rules()
            if rule["id"] in safety_rule_ids
        ]

        return {
            "asa_match": self.kb.get_asa(asa_match) if asa_match else None,
            "candidate_plans": candidate_plans,
            "matched_safety_rules": matched_rules,
            "recommended_drug_ids": self._collect_candidate_drugs(candidate_plans),
        }

    def call_patient_structuring_api(self, raw_patient_text: str) -> dict[str, Any]:
        prompt = (
            "You are a clinical structuring assistant.\n"
            "Convert the patient description into a JSON object only.\n"
            "Use these keys when possible: "
            "age, sex, weight_kg, height_cm, procedure_name, procedure_site, "
            "asa_hint, allergies, comorbidities, medications, fasting_status, airway_notes, "
            "urgency, labs, vitals.\n"
            "Rules:\n"
            "- Output valid JSON only.\n"
            "- allergies, comorbidities and medications must be arrays.\n"
            "- labs and vitals must be objects.\n"
            "- If information is missing, use null, [] or {}.\n\n"
            f"Patient text:\n{raw_patient_text}"
        )
        config = MODEL_CONFIGS[TASK_MODEL_SELECTION["patient_structuring"]]
        response = self.api_client.chat_json(config, prompt)
        return self.api_client.extract_json(response)

    def call_medical_entity_linking_api(
        self,
        normalized_patient: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = (
            "You are a medical entity normalization assistant.\n"
            "Normalize the following patient data to database-friendly concepts.\n"
            "Return JSON only with keys:\n"
            "normalized_terms, asa_interpretation, procedure_keywords, risk_keywords, medication_keywords.\n\n"
            f"Patient JSON:\n{json.dumps(normalized_patient, ensure_ascii=False, indent=2)}"
        )
        config = MODEL_CONFIGS[TASK_MODEL_SELECTION["entity_linking"]]
        response = self.api_client.chat_json(config, prompt)
        return self.api_client.extract_json(response)

    def call_plan_ranking_api(
        self,
        normalized_patient: dict[str, Any],
        kb_matches: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = (
            "You are an anesthesia planning ranking assistant.\n"
            "Given the normalized patient profile and candidate plans, rank the plans.\n"
            "Return JSON only with keys:\n"
            "ranked_plan_ids, top_choice, reasons, caution_points.\n\n"
            "Normalized patient:\n"
            f"{json.dumps(normalized_patient, ensure_ascii=False, indent=2)}\n\n"
            "Candidate matches:\n"
            f"{json.dumps(kb_matches, ensure_ascii=False, indent=2)}"
        )
        config = MODEL_CONFIGS[TASK_MODEL_SELECTION["plan_ranking"]]
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
        if canonical_key in {"allergies", "comorbidities", "medications"}:
            return self._coerce_list(value)
        return value

    def _coerce_list(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            text = value.replace("\uff0c", ",").replace("\uff1b", ",").replace("\u3001", ",")
            return [item.strip() for item in text.split(",") if item.strip()]
        if value is None:
            return []
        return [str(value).strip()]

    def _map_asa_hint(self, asa_hint: Any) -> str | None:
        if asa_hint is None:
            return None
        text = str(asa_hint).upper().replace(" ", "")
        if "ASA" not in text:
            text = f"ASA{text}"
        if "VI" in text:
            return "ASA_VI"
        if "IV" in text:
            return "ASA_IV"
        if "III" in text:
            return "ASA_III"
        if "II" in text:
            return "ASA_II"
        if "I" in text:
            return "ASA_I"
        return None

    def _infer_candidate_plan_ids(
        self,
        normalized_patient: dict[str, Any],
    ) -> list[str]:
        procedure_text = " ".join(
            filter(
                None,
                [
                    str(normalized_patient.get("procedure_name") or ""),
                    str(normalized_patient.get("procedure_site") or ""),
                ],
            )
        )
        fasting_text = str(normalized_patient.get("fasting_status") or "")
        airway_text = str(normalized_patient.get("airway_notes") or "")

        plan_ids: list[str] = []

        if any(token in procedure_text for token in ["\u4e0b\u80a2", "\u4e0b\u8179", "\u4f1a\u9634"]):
            plan_ids.append("spinal_anesthesia")

        if any(
            token in fasting_text + airway_text
            for token in ["\u672a\u7981\u98df", "\u9971\u80c3", "\u8bef\u5438", "\u56f0\u96be\u6c14\u9053"]
        ):
            plan_ids.append("general_anesthesia_ett")

        if any(token in procedure_text for token in ["\u77ed\u5c0f", "\u8868\u6d45", "\u5185\u955c", "\u5c40\u9ebb"]):
            plan_ids.append("monitored_anesthesia_care")

        if any(token in procedure_text for token in ["\u8179\u8154\u955c", "laparoscopy", "\u4e0a\u8179", "thoracic", "\u80f8\u79d1"]):
            plan_ids.append("general_anesthesia_ett")
            plan_ids.append("balanced_general_anesthesia")

        if any(token in procedure_text for token in ["\u5256\u5bab\u4ea7", "cesarean"]):
            plan_ids.append("spinal_anesthesia")
            plan_ids.append("general_anesthesia_ett")

        if not plan_ids:
            plan_ids.append("general_anesthesia_ett")

        return list(dict.fromkeys(plan_ids))

    def _infer_safety_rule_ids(
        self,
        normalized_patient: dict[str, Any],
    ) -> list[str]:
        comorbidity_text = " ".join(normalized_patient.get("comorbidities", []))
        medication_text = " ".join(normalized_patient.get("medications", []))
        fasting_text = str(normalized_patient.get("fasting_status") or "")
        airway_text = str(normalized_patient.get("airway_notes") or "")
        age = normalized_patient.get("age")
        candidate_plan_ids = self._infer_candidate_plan_ids(normalized_patient)
        rule_ids: list[str] = []

        if any(
            token in comorbidity_text
            for token in ["\u6076\u6027\u9ad8\u70ed", "malignant hyperthermia", "MH"]
        ):
            rule_ids.append("mh_trigger_avoidance")

        if any(
            token in fasting_text
            for token in ["\u672a\u7981\u98df", "\u9971\u80c3", "\u80c3\u6392\u7a7a\u5ef6\u8fdf", "\u80a0\u68d7\u963b"]
        ):
            rule_ids.append("aspiration_risk_review")

        if any(
            token in comorbidity_text + airway_text
            for token in ["OSA", "sleep apnea", "\u963b\u585e\u6027\u7761\u7720\u547c\u5438\u6682\u505c"]
        ):
            rule_ids.append("osa_monitoring_review")

        if any(
            token in medication_text.lower() + fasting_text.lower()
            for token in ["semaglutide", "tirzepatide", "liraglutide", "dulaglutide", "glp-1", "glp1"]
        ):
            rule_ids.append("glp1_delayed_gastric_emptying_review")

        if "spinal_anesthesia" in candidate_plan_ids and any(
            token in medication_text.lower() + comorbidity_text.lower()
            for token in ["warfarin", "heparin", "enoxaparin", "apixaban", "rivaroxaban", "dabigatran", "\u6297\u51dd", "\u6297\u8840\u5c0f\u677f"]
        ):
            rule_ids.append("neuraxial_antithrombotic_review")

        if any(
            token in airway_text
            for token in ["\u56f0\u96be\u6c14\u9053", "\u5c0f\u4e0b\u988c", "\u5f20\u53e3\u53d7\u9650", "\u9888\u9879\u6d3b\u52a8\u53d7\u9650"]
        ):
            rule_ids.append("difficult_airway_strategy_review")

        if isinstance(age, (int, float)) and age >= 65:
            rule_ids.append("older_adult_perioperative_review")

        return list(dict.fromkeys(rule_ids))

    def _collect_candidate_drugs(
        self,
        candidate_plans: list[dict[str, Any]],
    ) -> list[str]:
        drug_ids: list[str] = []
        for plan in candidate_plans:
            candidate_drugs = plan.get("candidate_drugs", {})
            for ids in candidate_drugs.values():
                drug_ids.extend(ids)
        return list(dict.fromkeys(drug_ids))


if __name__ == "__main__":
    kb = AnesthesiaKnowledgeBase()
    agent = PatientProfileMapperAgent(kb)

    raw_patient = {
        "\u5e74\u9f84": 67,
        "\u6027\u522b": "\u7537",
        "\u4f53\u91cd": 72,
        "\u62df\u884c\u624b\u672f": "\u4e0b\u80a2\u9aa8\u6298\u5185\u56fa\u5b9a",
        "ASA\u5206\u7ea7": "III",
        "\u8fc7\u654f\u53f2": "\u65e0",
        "\u57fa\u7840\u75be\u75c5": ["\u9ad8\u8840\u538b", "\u7cd6\u5c3f\u75c5"],
        "\u7981\u98df\u60c5\u51b5": "\u5df2\u7981\u98df 8 \u5c0f\u65f6",
        "\u6c14\u9053\u8bc4\u4f30": "\u5f20\u53e3\u5c1a\u53ef\uff0c\u6682\u672a\u63d0\u793a\u56f0\u96be\u6c14\u9053",
        "\u6025\u8bca": "\u62e9\u671f",
    }

    print("Knowledge Base Summary")
    print(json.dumps(kb.summary(), ensure_ascii=False, indent=2))

    print("\nCase Mapping Result")
    print(json.dumps(agent.build_case_record(raw_patient), ensure_ascii=False, indent=2))
