from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
import re
import socket
import ssl
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

from llm_runtime import get_api_key_for_model_key, get_default_model_key, get_gateway_url


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
MDT_CALL_DIR = PROJECT_ROOT / "mdt_call"


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
    "qwen3": ModelConfig(
        name="qwen-instruct",
        model="qwen-instruct",
    ),
    "deepseek_r1": ModelConfig(
        name="deepseek-r1:671b",
        model="deepseek-r1:671b",
    ),
    "deepseek_v3_2": ModelConfig(
        name="deepseek-v3:671b",
        model="deepseek-v3:671b",
    ),
}

HEAD_DOCTOR_REVIEW_MODELS = ("deepseek_v3_2", "qwen3")
HEAD_DOCTOR_REVIEW_MAX_WORKERS = 2
DEFAULT_MDT_AGENT_NAME = "head_doctor_mdt"
DEFAULT_MDT_SCHEMA_VERSION = "mdt.output.v1"
DEFAULT_MDT_INPUT_SCHEMA_VERSION = "mdt.input.v1"


SPECIALTY_SPECS = {
    "anesthesia": {
        "label": "麻醉科",
        "dir": PROJECT_ROOT / "Anesthesia",
        "mapper_module": "patient_mapper_agent.py",
        "mapper_class": "PatientProfileMapperAgent",
        "kb_class": "AnesthesiaKnowledgeBase",
        "retriever_module": "kb_retriever.py",
        "retriever_class": "KnowledgeRetriever",
        "decision_module": "anesthesia_decision_agent.py",
        "decision_class": "AnesthesiaDecisionAgent",
    },
    "cardiology": {
        "label": "心内科",
        "dir": PROJECT_ROOT / "Cardiology",
        "mapper_module": "patient_mapper_agent.py",
        "mapper_class": "PatientProfileMapperAgent",
        "kb_class": "CardiologyKnowledgeBase",
        "retriever_module": "kb_retriever.py",
        "retriever_class": "KnowledgeRetriever",
        "decision_module": "cardiology_decision_agent.py",
        "decision_class": "CardiologyDecisionAgent",
    },
    "hepatobiliary": {
        "label": "肝胆胰外科",
        "dir": PROJECT_ROOT / "Hepatobiliary and Pancreatic Surgery",
        "mapper_module": "patient_mapper_agent.py",
        "mapper_class": "PatientProfileMapperAgent",
        "kb_class": "HepatobiliaryKnowledgeBase",
        "retriever_module": "hbp_kb_retriever.py",
        "retriever_class": "KnowledgeRetriever",
        "decision_module": "hbp_decision_agent.py",
        "decision_class": "HepatobiliaryDecisionAgent",
    },
    "neurosurgery": {
        "label": "神经外科",
        "dir": PROJECT_ROOT / "Neurosurgery",
        "mapper_module": "neurosurgery_patient_mapper_agent.py",
        "mapper_class": "NeurosurgeryPatientProfileMapperAgent",
        "kb_class": "NeurosurgeryKnowledgeBase",
        "retriever_module": "neurosurgery_kb_retriever.py",
        "retriever_class": "NeurosurgeryKnowledgeRetriever",
        "decision_module": "neurosurgery_decision_agent.py",
        "decision_class": "NeurosurgeryDecisionAgent",
    },
    "orthopaedics": {
        "label": "骨科",
        "dir": PROJECT_ROOT / "Orthopaedics",
        "mapper_module": "patient_mapper_agent.py",
        "mapper_class": "PatientProfileMapperAgent",
        "kb_class": "OrthopaedicsKnowledgeBase",
        "retriever_module": "kb_retriever.py",
        "retriever_class": "KnowledgeRetriever",
        "decision_module": "orthopaedics_decision_agent.py",
        "decision_class": "OrthopaedicsDecisionAgent",
    },
}


class SharedGenAIChatClient:
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
            next((key for key, candidate in MODEL_CONFIGS.items() if candidate == config), get_default_model_key())
        )
        if not api_key:
            raise ValueError("Missing DeepSeek API key environment variable for the selected model.")
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
                with request.urlopen(req, timeout=90) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(
                    f"Head doctor API HTTP {exc.code}: {exc.reason}. Response: {body[:1200]}"
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

        raise RuntimeError(f"Head doctor API request failed after retries: {last_error}") from last_error

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


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_mdt_call_agent_class():
    if str(MDT_CALL_DIR) not in sys.path:
        sys.path.insert(0, str(MDT_CALL_DIR))
    module = _load_module(
        "mdt_call_agent_module",
        MDT_CALL_DIR / "mdt_call_agent.py",
    )
    return getattr(module, "MDTCallAgent")


def _load_specialty_modules(specialty_id: str, spec: dict[str, Any]) -> tuple[Any, Any, Any]:
    specialty_dir = Path(spec["dir"])
    if str(specialty_dir) not in sys.path:
        sys.path.insert(0, str(specialty_dir))

    mapper_module = _load_module(
        f"{specialty_id}_mapper_module",
        specialty_dir / str(spec["mapper_module"]),
    )
    sys.modules["patient_mapper_agent"] = mapper_module

    retriever_module = _load_module(
        f"{specialty_id}_retriever_module",
        specialty_dir / str(spec["retriever_module"]),
    )
    decision_module = _load_module(
        f"{specialty_id}_decision_module",
        specialty_dir / str(spec["decision_module"]),
    )
    return mapper_module, retriever_module, decision_module


class SpecialtyAgentAdapter:
    def __init__(self, specialty_id: str, spec: dict[str, Any], shared_api_client: SharedGenAIChatClient) -> None:
        self.specialty_id = specialty_id
        self.label = str(spec["label"])
        mapper_module, retriever_module, decision_module = _load_specialty_modules(
            specialty_id,
            spec,
        )

        kb_class = getattr(mapper_module, str(spec["kb_class"]))
        mapper_class = getattr(mapper_module, str(spec["mapper_class"]))
        retriever_class = getattr(retriever_module, str(spec["retriever_class"]))
        decision_class = getattr(decision_module, str(spec["decision_class"]))

        self.api_client = shared_api_client
        self.kb = kb_class()
        self.mapper = mapper_class(self.kb)
        self.retriever = retriever_class(self.kb)
        self.decision_agent = decision_class()

    def evaluate_case(
        self,
        patient_input: dict[str, Any] | str,
        use_api_for_structuring: bool,
        use_api_for_entity_linking: bool,
        use_api_for_retrieval: bool,
        use_api_for_decision: bool,
    ) -> dict[str, Any]:
        warnings: list[str] = []
        structured_patient, normalized_patient = self._prepare_patient(
            patient_input,
            use_api_for_structuring=use_api_for_structuring,
            warnings=warnings,
        )
        entity_linking = self._call_entity_linking_with_fallback(
            normalized_patient,
            use_api_for_entity_linking=use_api_for_entity_linking,
            warnings=warnings,
        )
        retrieval_result = self._retrieve_case(
            normalized_patient,
            entity_linking,
            use_api_for_retrieval=use_api_for_retrieval,
            warnings=warnings,
        )
        decision_result = self._decide_case(
            normalized_patient,
            retrieval_result,
            use_api_for_decision=use_api_for_decision,
            warnings=warnings,
        )
        return {
            "specialty": self.specialty_id,
            "specialty_label": self.label,
            "structured_patient": structured_patient,
            "normalized_patient": normalized_patient,
            "entity_linking": entity_linking,
            "retrieval_result": retrieval_result,
            "decision_result": decision_result,
            "warnings": warnings,
        }

    def clarify_case(
        self,
        patient_input: dict[str, Any] | str,
        specialty_evaluation: dict[str, Any],
        question: str,
        use_api_for_clarification: bool,
    ) -> dict[str, Any]:
        if use_api_for_clarification:
            prompt = (
                f"你现在扮演{self.label}专科医生。\n"
                "请基于患者信息、你之前的专科结论，以及当前追问问题，补充更细化的意见。\n"
                "只返回 JSON，字段固定为：specialty, question, answer, action_items, remaining_uncertainties, confidence。\n"
                "- answer 必须是一个简短字符串（不超过120字），不要返回数组。\n"
                "- action_items 和 remaining_uncertainties 必须是字符串数组。\n"
                "- action_items 最多 3 条，remaining_uncertainties 最多 3 条。\n"
                "- 若仍不能确认，请明确写入 remaining_uncertainties。\n\n"
                f"患者信息：\n{json.dumps(patient_input, ensure_ascii=False, indent=2) if isinstance(patient_input, dict) else patient_input}\n\n"
                f"既有专科结论：\n{json.dumps(specialty_evaluation, ensure_ascii=False, indent=2)}\n\n"
                f"追问问题：\n{question}"
            )
            try:
                response = self.api_client.chat_json(MODEL_CONFIGS["gpt_5_2"], prompt)
                payload = self.api_client.extract_json(response)

                answer_value = payload.get("answer")
                if isinstance(answer_value, list):
                    answer_text = "；".join(str(item).strip() for item in answer_value if str(item).strip())
                    payload["answer"] = answer_text[:240]
                elif isinstance(answer_value, str):
                    payload["answer"] = answer_value.strip()[:240]
                elif answer_value is None:
                    payload["answer"] = ""
                else:
                    payload["answer"] = str(answer_value).strip()[:240]

                payload.setdefault("specialty", self.specialty_id)
                payload.setdefault("question", question)
                payload["action_items"] = self._string_list(payload.get("action_items"))[:3]
                payload["remaining_uncertainties"] = self._string_list(payload.get("remaining_uncertainties"))[:3]
                payload["confidence"] = str(payload.get("confidence") or "medium")
                return payload
            except Exception:
                # Fall back to deterministic local clarification to avoid aborting the whole workflow.
                pass

        decision_result = specialty_evaluation.get("decision_result", {})
        retrieval_result = specialty_evaluation.get("retrieval_result", {})
        answer_parts = []
        if decision_result.get("primary_plan"):
            primary_plan = decision_result["primary_plan"]
            answer_parts.append(f"当前首选方案仍倾向 {primary_plan.get('name_zh') or primary_plan.get('id')}")
        elif decision_result.get("primary_plan_id"):
            answer_parts.append(f"当前首选方案仍倾向 {decision_result.get('primary_plan_id')}")
        if decision_result.get("primary_condition"):
            primary_condition = decision_result["primary_condition"]
            answer_parts.append(f"核心问题仍聚焦于 {primary_condition.get('name_zh') or primary_condition.get('id')}")
        elif decision_result.get("primary_condition_id"):
            answer_parts.append(f"核心问题仍聚焦于 {decision_result.get('primary_condition_id')}")
        if retrieval_result.get("matched_safety_rules"):
            answer_parts.append(
                "需要重点参照安全规则："
                + "、".join(
                    str(rule.get("name_zh") or rule.get("id"))
                    for rule in retrieval_result["matched_safety_rules"]
                )
            )
        remaining = self._string_list(decision_result.get("need_more_info"))
        return {
            "specialty": self.specialty_id,
            "question": question,
            "answer": "；".join(answer_parts) if answer_parts else f"{self.label}建议补充关键信息后再细化判断。",
            "action_items": self._string_list(decision_result.get("risk_flags")),
            "remaining_uncertainties": remaining,
            "confidence": "medium" if answer_parts else "low",
        }

    def _retrieve_case(
        self,
        normalized_patient: dict[str, Any],
        entity_linking: dict[str, Any],
        use_api_for_retrieval: bool,
        warnings: list[str] | None = None,
    ) -> dict[str, Any]:
        if use_api_for_retrieval and hasattr(self.retriever, "retrieve_with_api"):
            try:
                return self.retriever.retrieve_with_api(normalized_patient, entity_linking)
            except Exception as exc:
                if warnings is not None:
                    warnings.append(f"{self.specialty_id} retrieval_api_fallback: {exc}")
        return self.retriever.retrieve(normalized_patient, entity_linking)

    def _call_entity_linking_with_fallback(
        self,
        normalized_patient: dict[str, Any],
        *,
        use_api_for_entity_linking: bool,
        warnings: list[str] | None = None,
    ) -> dict[str, Any]:
        if not use_api_for_entity_linking:
            return {}
        try:
            return self.mapper.call_medical_entity_linking_api(normalized_patient)
        except Exception as exc:
            if warnings is not None:
                warnings.append(f"{self.specialty_id} entity_linking_api_fallback: {exc}")
            return {}

    def _decide_case(
        self,
        normalized_patient: dict[str, Any],
        retrieval_result: dict[str, Any],
        *,
        use_api_for_decision: bool,
        warnings: list[str] | None = None,
    ) -> dict[str, Any]:
        if use_api_for_decision:
            try:
                return self.decision_agent.decide_with_api(normalized_patient, retrieval_result)
            except Exception as exc:
                if warnings is not None:
                    warnings.append(f"{self.specialty_id} decision_api_fallback: {exc}")
        return self.decision_agent.decide(normalized_patient, retrieval_result)

    def _prepare_patient(
        self,
        patient_input: dict[str, Any] | str,
        use_api_for_structuring: bool,
        warnings: list[str] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if isinstance(patient_input, dict):
            structured_patient = patient_input
        elif use_api_for_structuring:
            try:
                structured_patient = self.mapper.call_patient_structuring_api(patient_input)
            except Exception as exc:
                if warnings is not None:
                    warnings.append(f"{self.specialty_id} structuring_api_fallback: {exc}")
                structured_patient = {"raw_text": patient_input}
        else:
            structured_patient = {"raw_text": patient_input}
        normalized = self.mapper.normalize_patient_input(structured_patient)
        return structured_patient, normalized

    def _string_list(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []


class HeadDoctorAgent:
    def __init__(self, api_client: SharedGenAIChatClient | None = None) -> None:
        self.api_client = api_client or SharedGenAIChatClient()
        self.specialists = {
            specialty_id: SpecialtyAgentAdapter(specialty_id, spec, self.api_client)
            for specialty_id, spec in SPECIALTY_SPECS.items()
        }
        mdt_call_agent_class = _load_mdt_call_agent_class()
        self.mdt_call_agent = mdt_call_agent_class(api_client=self.api_client)

    def evaluate_case(
        self,
        patient_input: dict[str, Any] | str,
        use_api_for_structuring: bool = True,
        use_api_for_entity_linking: bool = True,
        use_api_for_specialists: bool = True,
        use_mdt_for_uncertainty: bool = True,
        use_api_for_mdt: bool = True,
        use_api_for_clarification: bool = True,
        use_api_for_final: bool = True,
    ) -> dict[str, Any]:
        envelope_context = self._prepare_mdt_envelope_context(patient_input)
        normalized_patient_input = envelope_context["normalized_patient_input"]
        initial_triage = self.mdt_call_agent.coordinate_initial_triage(
            patient_input=normalized_patient_input,
            specialist_callback=lambda specialty_id, case_input: self._dispatch_initial_specialty_eval(
                specialty_id=specialty_id,
                patient_input=case_input,
                use_api_for_structuring=use_api_for_structuring,
                use_api_for_entity_linking=use_api_for_entity_linking,
                use_api_for_specialists=use_api_for_specialists,
            ),
            use_api_for_planning=use_api_for_mdt,
        )
        specialty_opinions = initial_triage.get("specialty_opinions", {})

        uncertainty_review = self._collect_uncertainties(specialty_opinions)
        mdt_follow_up = None
        if use_mdt_for_uncertainty and uncertainty_review["unresolved_items"]:
            mdt_follow_up = self.mdt_call_agent.coordinate_follow_up(
                patient_input=normalized_patient_input,
                specialty_opinions=specialty_opinions,
                unresolved_items=uncertainty_review["unresolved_items"],
                specialist_callback=self._dispatch_clarification,
                use_api_for_planning=use_api_for_mdt,
                use_api_for_clarification=use_api_for_clarification,
            )

        final_recommendation = (
            self._integrate_with_api(
                patient_input=normalized_patient_input,
                specialty_opinions=specialty_opinions,
                uncertainty_review=uncertainty_review,
                mdt_follow_up=mdt_follow_up,
            )
            if use_api_for_final
            else self._integrate_locally(
                patient_input=normalized_patient_input,
                specialty_opinions=specialty_opinions,
                uncertainty_review=uncertainty_review,
                mdt_follow_up=mdt_follow_up,
            )
        )

        workflow_result = {
            "patient_input": normalized_patient_input,
            "patient_input_original": envelope_context["original_patient_input"],
            "patient_input_payload": envelope_context["input_payload"],
            "mdt_input_envelope": envelope_context["input_envelope"],
            "initial_triage": initial_triage,
            "specialty_opinions": specialty_opinions,
            "uncertainty_review": uncertainty_review,
            "mdt_follow_up": mdt_follow_up,
            "head_doctor_recommendation": final_recommendation,
        }
        workflow_result["mdt_output_envelope"] = self._build_mdt_output_envelope(
            input_envelope=envelope_context["input_envelope"],
            final_recommendation=final_recommendation,
        )
        return workflow_result

    def _prepare_mdt_envelope_context(
        self,
        patient_input: dict[str, Any] | str,
    ) -> dict[str, Any]:
        if self._looks_like_agent_envelope(patient_input):
            input_envelope = dict(patient_input)
            payload = input_envelope.get("payload")
            normalized_patient_input = self._coerce_patient_input_for_mdt(payload)
            return {
                "original_patient_input": patient_input,
                "input_envelope": self._normalize_input_envelope(input_envelope),
                "input_payload": payload,
                "normalized_patient_input": normalized_patient_input,
            }

        payload = self._normalize_payload_for_mdt(patient_input)
        envelope = self._normalize_input_envelope(
            {
                "payload": payload,
                "output_type": "mdt_case_intake",
                "agent_name": "mdt_intake",
                "schema_version": DEFAULT_MDT_INPUT_SCHEMA_VERSION,
            }
        )
        return {
            "original_patient_input": patient_input,
            "input_envelope": envelope,
            "input_payload": payload,
            "normalized_patient_input": self._coerce_patient_input_for_mdt(payload),
        }

    def _looks_like_agent_envelope(self, value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        required_keys = {
            "output_id",
            "admission_id",
            "patient_id",
            "bed_id",
            "agent_name",
            "schema_version",
            "output_type",
            "generated_at",
            "payload",
        }
        return required_keys.issubset(set(value.keys()))

    def _normalize_input_envelope(self, envelope: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(envelope)
        payload = self._normalize_payload_for_mdt(normalized.get("payload"))
        payload_dict = payload if isinstance(payload, dict) else {}
        normalized["output_id"] = str(normalized.get("output_id") or self._new_mdt_id("in"))
        normalized["admission_id"] = str(
            normalized.get("admission_id") or payload_dict.get("admission_id") or self._new_mdt_id("adm")
        )
        normalized["patient_id"] = str(
            normalized.get("patient_id") or payload_dict.get("patient_id") or self._new_mdt_id("pat")
        )
        normalized["bed_id"] = str(
            normalized.get("bed_id") or payload_dict.get("bed_id") or self._new_mdt_id("bed")
        )
        normalized["agent_name"] = str(normalized.get("agent_name") or "mdt_intake")
        normalized["schema_version"] = str(normalized.get("schema_version") or DEFAULT_MDT_INPUT_SCHEMA_VERSION)
        normalized["output_type"] = str(normalized.get("output_type") or "mdt_case_intake")
        normalized["generated_at"] = self._normalize_timestamp(
            normalized.get("generated_at") or payload_dict.get("updated_at")
        )
        normalized["payload"] = payload
        return normalized

    def _normalize_payload_for_mdt(self, payload: Any) -> dict[str, Any] | str:
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, str):
            return payload
        if payload is None:
            return {}
        return {"raw_input": payload}

    def _looks_like_icu_state_payload(self, payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        required_keys = {
            "admission_id",
            "patient_id",
            "bed_id",
            "current_vitals",
            "active_problems",
            "active_risks",
            "latest_interventions",
            "care_phase",
        }
        return required_keys.issubset(set(payload.keys()))

    def _coerce_patient_input_for_mdt(self, payload: Any) -> dict[str, Any] | str:
        normalized = self._normalize_payload_for_mdt(payload)
        if self._looks_like_icu_state_payload(normalized):
            return self._build_case_text_from_icu_payload(normalized)
        return normalized

    def _build_case_text_from_icu_payload(self, payload: dict[str, Any]) -> str:
        vitals_text = self._format_named_mapping(payload.get("current_vitals"))
        problems = self._string_list_from_any(payload.get("active_problems"))
        risks = self._string_list_from_any(payload.get("active_risks"))
        interventions = self._string_list_from_any(payload.get("latest_interventions"))
        segments = [
            f"ICU患者交接：admission_id={payload.get('admission_id')}，patient_id={payload.get('patient_id')}，bed_id={payload.get('bed_id')}。",
            f"当前状态更新时间：{payload.get('updated_at') or '未提供'}。",
            f"当前 care_phase：{payload.get('care_phase') or 'unstable'}。",
        ]
        if vitals_text:
            segments.append(f"当前生命体征：{vitals_text}。")
        if problems:
            segments.append("当前活动问题：" + "；".join(problems) + "。")
        if risks:
            segments.append("当前活动风险：" + "；".join(risks) + "。")
        if interventions:
            segments.append("最新干预与处理：" + "；".join(interventions) + "。")
        segments.append("请基于以上ICU当前状态信息进行MDT评估，并给出围术期会诊建议。")
        return "".join(segments)

    def _build_mdt_output_envelope(
        self,
        input_envelope: dict[str, Any],
        final_recommendation: dict[str, Any],
    ) -> dict[str, Any]:
        standardized_payload = self._build_standardized_mdt_payload(
            input_envelope=input_envelope,
            final_recommendation=final_recommendation,
        )
        return {
            "output_id": self._new_mdt_id("out"),
            "admission_id": str(input_envelope.get("admission_id") or self._new_mdt_id("adm")),
            "patient_id": str(input_envelope.get("patient_id") or self._new_mdt_id("pat")),
            "bed_id": str(input_envelope.get("bed_id") or self._new_mdt_id("bed")),
            "agent_name": DEFAULT_MDT_AGENT_NAME,
            "schema_version": DEFAULT_MDT_SCHEMA_VERSION,
            "output_type": "mdt_recommendation_ready",
            "generated_at": self._utc_now_iso(),
            "payload": standardized_payload,
        }

    def _build_standardized_mdt_payload(
        self,
        input_envelope: dict[str, Any],
        final_recommendation: dict[str, Any],
    ) -> dict[str, Any]:
        input_payload = input_envelope.get("payload")
        current_vitals = self._extract_current_vitals_from_input(input_payload)
        input_active_problems = self._extract_problem_items_from_input(input_payload)
        input_active_risks = self._extract_risk_items_from_input(input_payload)
        input_latest_interventions = self._extract_intervention_items_from_input(input_payload)
        active_problems = self._merge_problem_items(
            input_active_problems,
            self._build_problem_items_from_recommendation(final_recommendation),
        )
        active_risks = self._merge_risk_items(
            input_active_risks,
            self._build_risk_items_from_recommendation(final_recommendation),
        )
        latest_interventions = self._merge_intervention_items(
            input_latest_interventions,
            self._build_intervention_items_from_recommendation(final_recommendation),
        )
        active_problems = self._filter_icu_problem_objects(active_problems)
        active_risks = self._filter_icu_risk_objects(active_risks)
        latest_interventions = self._filter_icu_intervention_objects(latest_interventions)
        care_phase = self._merge_care_phase(
            self._extract_input_care_phase(input_payload),
            self._map_current_status_to_care_phase(
            final_recommendation.get("current_status_level"),
            active_risks=[self._risk_item_to_text(item) for item in active_risks],
        )
        )
        return {
            "admission_id": str(input_envelope.get("admission_id") or self._new_mdt_id("adm")),
            "patient_id": str(input_envelope.get("patient_id") or self._new_mdt_id("pat")),
            "bed_id": str(input_envelope.get("bed_id") or self._new_mdt_id("bed")),
            "updated_at": self._utc_now_iso(),
            "current_vitals": current_vitals,
            "active_problems": active_problems,
            "active_risks": active_risks,
            "latest_interventions": latest_interventions,
            "care_phase": care_phase,
        }

    def _extract_current_vitals_from_input(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        for key in ("current_vitals", "vitals", "生命体征"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        return {}

    def _extract_input_care_phase(self, payload: Any) -> str | None:
        if not isinstance(payload, dict):
            return None
        value = str(payload.get("care_phase") or "").strip().lower()
        return value or None

    def _extract_string_list_field(self, payload: Any, field_name: str) -> list[str]:
        if not isinstance(payload, dict):
            return []
        return self._string_list_from_any(payload.get(field_name))

    def _extract_problem_items_from_input(self, payload: Any) -> list[dict[str, Any]]:
        if not isinstance(payload, dict):
            return []
        return self._coerce_problem_items(payload.get("active_problems"))

    def _extract_risk_items_from_input(self, payload: Any) -> list[dict[str, Any]]:
        if not isinstance(payload, dict):
            return []
        return self._coerce_risk_items(payload.get("active_risks"))

    def _extract_intervention_items_from_input(self, payload: Any) -> list[dict[str, Any]]:
        if not isinstance(payload, dict):
            return []
        return self._coerce_intervention_items(payload.get("latest_interventions"))

    def _string_list_from_any(self, value: Any) -> list[str]:
        if isinstance(value, list):
            items: list[str] = []
            for item in value:
                if isinstance(item, str):
                    text = item.strip()
                    if text:
                        items.append(text)
                elif isinstance(item, dict):
                    items.extend(self._dict_item_to_strings(item))
                elif item is not None:
                    items.append(str(item).strip())
            return self._merge_string_lists(items)
        if isinstance(value, dict):
            return self._dict_item_to_strings(value)
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    def _dict_item_to_strings(self, item: dict[str, Any]) -> list[str]:
        candidates = []
        for keys in (
            ("problem", "status"),
            ("risk_type", "severity"),
            ("intervention_type", "description"),
            ("title", "detail"),
            ("name", "value"),
        ):
            primary = str(item.get(keys[0]) or "").strip()
            secondary = str(item.get(keys[1]) or "").strip()
            if primary and secondary:
                candidates.append(f"{primary}: {secondary}")
            elif primary:
                candidates.append(primary)
        if candidates:
            return candidates
        return [json.dumps(item, ensure_ascii=False, sort_keys=True)]

    def _format_named_mapping(self, value: Any) -> str:
        if not isinstance(value, dict):
            return ""
        parts = [f"{k}={v}" for k, v in value.items() if v is not None and str(v).strip()]
        return "，".join(parts)

    def _merge_string_lists(self, *groups: list[str]) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for group in groups:
            for item in group:
                text = str(item).strip()
                if not text:
                    continue
                key = text.casefold()
                if key in seen:
                    continue
                seen.add(key)
                merged.append(text)
        return merged

    def _coerce_risk_items(self, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        items: list[dict[str, Any]] = []
        for raw in value:
            if isinstance(raw, dict):
                risk_type = str(raw.get("risk_type") or raw.get("type") or "").strip()
                severity = str(raw.get("severity") or "").strip().lower() or "warning"
                if not risk_type:
                    continue
                items.append(
                    {
                        "risk_type": risk_type,
                        "severity": severity if severity in ("low", "warning", "critical") else "warning",
                    }
                )
            elif isinstance(raw, str) and raw.strip():
                items.append(
                    {
                        "risk_type": raw.strip(),
                        "severity": self._infer_risk_severity(raw),
                    }
                )
        return items

    def _coerce_problem_items(self, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        items: list[dict[str, Any]] = []
        for raw in value:
            if isinstance(raw, dict):
                problem = str(
                    raw.get("problem")
                    or raw.get("name")
                    or raw.get("title")
                    or raw.get("problem_type")
                    or ""
                ).strip()
                status = str(raw.get("status") or "active").strip().lower() or "active"
                if not problem:
                    continue
                items.append(
                    {
                        "problem": problem,
                        "status": status,
                    }
                )
            elif isinstance(raw, str) and raw.strip():
                items.append(
                    {
                        "problem": raw.strip(),
                        "status": "active",
                    }
                )
        return items

    def _coerce_intervention_items(self, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        items: list[dict[str, Any]] = []
        for raw in value:
            if isinstance(raw, dict):
                intervention_type = str(raw.get("intervention_type") or raw.get("type") or "").strip()
                description = str(raw.get("description") or raw.get("detail") or "").strip()
                timestamp = str(raw.get("timestamp") or raw.get("updated_at") or "").strip()
                if not intervention_type and not description:
                    continue
                items.append(
                    {
                        "id": str(raw.get("id") or self._new_mdt_id("intv")),
                        "intervention_type": intervention_type or "mdt_action",
                        "description": description or intervention_type,
                        "timestamp": timestamp or self._utc_now_iso(),
                    }
                )
            elif isinstance(raw, str) and raw.strip():
                items.append(
                    {
                        "id": self._new_mdt_id("intv"),
                        "intervention_type": "mdt_action",
                        "description": raw.strip(),
                        "timestamp": self._utc_now_iso(),
                    }
                )
        return items

    def _build_risk_items_from_recommendation(
        self,
        final_recommendation: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return [
            {
                "risk_type": text,
                "severity": self._infer_risk_severity(text),
            }
            for text in self._string_list(final_recommendation.get("key_risks"))
            if text.strip()
        ]

    def _build_problem_items_from_recommendation(
        self,
        final_recommendation: dict[str, Any],
    ) -> list[dict[str, Any]]:
        merged_problems = self._merge_string_lists(
            self._string_list(final_recommendation.get("core_constraints")),
            self._string_list(final_recommendation.get("unresolved_issues")),
        )
        return [
            {
                "problem": text,
                "status": "active",
            }
            for text in merged_problems
            if text.strip()
        ]

    def _build_intervention_items_from_recommendation(
        self,
        final_recommendation: dict[str, Any],
    ) -> list[dict[str, Any]]:
        merged_steps = self._merge_string_lists(
            self._string_list(final_recommendation.get("specific_surgical_plan")),
            self._string_list(final_recommendation.get("final_plan")),
            self._string_list(final_recommendation.get("next_steps")),
        )
        return [
            {
                "id": self._new_mdt_id("intv"),
                "intervention_type": "mdt_recommendation",
                "description": text,
                "timestamp": self._utc_now_iso(),
            }
            for text in merged_steps
            if text.strip()
        ]

    def _merge_problem_items(self, *groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for group in groups:
            for item in group:
                problem = str(item.get("problem") or "").strip()
                status = str(item.get("status") or "active").strip().lower() or "active"
                if not problem:
                    continue
                key = (problem.casefold(), status)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(
                    {
                        "problem": problem,
                        "status": status,
                    }
                )
        return merged

    def _merge_risk_items(self, *groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for group in groups:
            for item in group:
                risk_type = str(item.get("risk_type") or "").strip()
                severity = str(item.get("severity") or "warning").strip().lower()
                if not risk_type:
                    continue
                key = (risk_type.casefold(), severity)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(
                    {
                        "risk_type": risk_type,
                        "severity": severity if severity in ("low", "warning", "critical") else "warning",
                    }
                )
        return merged

    def _merge_intervention_items(self, *groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for group in groups:
            for item in group:
                intervention_type = str(item.get("intervention_type") or "").strip()
                description = str(item.get("description") or "").strip()
                if not intervention_type and not description:
                    continue
                key = f"{intervention_type.casefold()}|{description.casefold()}"
                if key in seen:
                    continue
                seen.add(key)
                merged.append(
                    {
                        "id": str(item.get("id") or self._new_mdt_id("intv")),
                        "intervention_type": intervention_type or "mdt_action",
                        "description": description or intervention_type,
                        "timestamp": str(item.get("timestamp") or self._utc_now_iso()),
                    }
                )
        return merged[:10]

    def _filter_icu_problem_objects(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cleaned: list[dict[str, Any]] = []
        for item in items:
            problem = str(item.get("problem") or "").strip()
            if not problem or not self._is_valid_icu_problem_item(problem):
                continue
            cleaned.append(
                {
                    "problem": problem,
                    "status": str(item.get("status") or "active").strip().lower() or "active",
                }
            )
        return self._merge_problem_items(cleaned)

    def _filter_icu_risk_objects(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cleaned: list[dict[str, Any]] = []
        for item in items:
            normalized = self._normalize_icu_risk_item(str(item.get("risk_type") or "").strip())
            if not normalized or not self._is_valid_icu_risk_item(normalized):
                continue
            cleaned.append(
                {
                    "risk_type": normalized,
                    "severity": str(item.get("severity") or "warning").strip().lower() or "warning",
                }
            )
        return self._merge_risk_items(cleaned)

    def _filter_icu_intervention_objects(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cleaned: list[dict[str, Any]] = []
        for item in items:
            description = self._normalize_icu_intervention_item(str(item.get("description") or "").strip())
            if not description or not self._is_valid_icu_intervention_item(description):
                continue
            cleaned.append(
                {
                    "id": str(item.get("id") or self._new_mdt_id("intv")),
                    "intervention_type": str(item.get("intervention_type") or "mdt_action").strip() or "mdt_action",
                    "description": description,
                    "timestamp": str(item.get("timestamp") or self._utc_now_iso()),
                }
            )
        return self._merge_intervention_items(cleaned)

    def _infer_risk_severity(self, text: Any) -> str:
        normalized = str(text or "").strip().lower()
        if any(token in normalized for token in ("critical", "禁忌", "脑疝", "休克", "呼吸衰竭")):
            return "critical"
        if any(token in normalized for token in ("high", "增高", "风险", "不佳", "异常")):
            return "warning"
        return "low"

    def _risk_item_to_text(self, item: dict[str, Any]) -> str:
        risk_type = str(item.get("risk_type") or "").strip()
        severity = str(item.get("severity") or "").strip().lower()
        if risk_type and severity:
            return f"{risk_type} ({severity})"
        return risk_type

    def _filter_icu_problem_items(self, items: list[str]) -> list[str]:
        return [item for item in items if self._is_valid_icu_problem_item(item)]

    def _filter_icu_risk_items(self, items: list[str]) -> list[str]:
        cleaned: list[str] = []
        for item in items:
            normalized = self._normalize_icu_risk_item(item)
            if normalized and self._is_valid_icu_risk_item(normalized):
                cleaned.append(normalized)
        return self._merge_string_lists(cleaned)

    def _filter_icu_intervention_items(self, items: list[str]) -> list[str]:
        cleaned: list[str] = []
        for item in items:
            normalized = self._normalize_icu_intervention_item(item)
            if normalized and self._is_valid_icu_intervention_item(normalized):
                cleaned.append(normalized)
        return self._merge_string_lists(cleaned)

    def _is_valid_icu_problem_item(self, text: str) -> bool:
        normalized = str(text).strip()
        if not normalized:
            return False
        reject_terms = (
            "缺少",
            "未返回",
            "建议重试",
            "评估结果暂不可用",
            "超时或接口失败",
            "请进一步说明",
            "需要补充",
            "补充关键信息",
            "confidence",
            "error",
            "review_model",
            "trigger:",
            "owner:",
            "步骤",
        )
        lowered = normalized.lower()
        return not any(term in normalized or term in lowered for term in reject_terms)

    def _is_valid_icu_risk_item(self, text: str) -> bool:
        normalized = str(text).strip()
        if not normalized:
            return False
        reject_terms = (
            "缺少",
            "未返回",
            "建议重试",
            "评估结果暂不可用",
            "超时或接口失败",
            "请进一步说明",
            "补充关键信息",
            "error",
            "review_model",
            "trigger:",
            "owner:",
            "步骤",
        )
        lowered = normalized.lower()
        return not any(term in normalized or term in lowered for term in reject_terms)

    def _normalize_icu_risk_item(self, text: str) -> str:
        normalized = str(text).strip()
        replacements = (
            ("恶性高热触发药规避", "malignant_hyperthermia_risk"),
            ("残余肌松复核", "residual_neuromuscular_blockade_risk"),
            ("舒更葡糖适用范围复核", "neuromuscular_reversal_risk"),
            ("新斯的明需配伍抗胆碱药", "anticholinergic_cotherapy_risk"),
            ("高钾风险时规避琥珀胆碱", "hyperkalemia_succinylcholine_risk"),
            ("缺少完整气道评估", ""),
            ("缺少心电图摘要", ""),
        )
        for source, target in replacements:
            if source == normalized:
                return target
        return normalized

    def _is_valid_icu_intervention_item(self, text: str) -> bool:
        normalized = str(text).strip()
        if not normalized:
            return False
        reject_terms = (
            "缺少",
            "未返回",
            "建议重试",
            "评估结果暂不可用",
            "超时或接口失败",
            "请进一步说明",
            "confidence",
            "error",
            "review_model",
        )
        lowered = normalized.lower()
        return not any(term in normalized or term in lowered for term in reject_terms)

    def _normalize_icu_intervention_item(self, text: str) -> str:
        normalized = str(text).strip()
        normalized = re.sub(r"^步骤\d+[：:]\s*", "", normalized)
        replacements = (
            ("麻醉科建议优先按 全身麻醉-气管插管 路径处理", "general_anesthesia_intubation_preferred"),
            ("心内科建议优先按 ACS 初始评估路径 路径处理", "cardiovascular_risk_pathway_activated"),
        )
        for source, target in replacements:
            if source == normalized:
                return target
        return normalized.strip()

    def _map_current_status_to_care_phase(
        self,
        current_status_level: Any,
        *,
        active_risks: list[str],
    ) -> str:
        level = str(current_status_level or "").strip().upper()
        if level == "READY":
            return "stable"
        if level == "CONTRAINDICATED":
            return "critical"
        if any("禁忌" in item or "critical" in item.lower() for item in active_risks):
            return "critical"
        return "unstable"

    def _merge_care_phase(self, input_phase: str | None, derived_phase: str) -> str:
        phase_rank = {"stable": 0, "unstable": 1, "critical": 2}
        input_rank = phase_rank.get(str(input_phase or "").lower(), -1)
        derived_rank = phase_rank.get(derived_phase, 1)
        return input_phase if input_rank >= derived_rank and input_phase else derived_phase

    def _new_mdt_id(self, prefix: str) -> str:
        return f"mdt_{prefix}_{uuid.uuid4().hex[:12]}"

    def _utc_now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _normalize_timestamp(self, value: Any) -> str:
        if isinstance(value, str) and value.strip():
            return value.strip()
        return self._utc_now_iso()

    def _dispatch_initial_specialty_eval(
        self,
        specialty_id: str,
        patient_input: dict[str, Any] | str,
        use_api_for_structuring: bool,
        use_api_for_entity_linking: bool,
        use_api_for_specialists: bool,
    ) -> dict[str, Any]:
        adapter = self.specialists[specialty_id]
        return adapter.evaluate_case(
            patient_input=patient_input,
            use_api_for_structuring=use_api_for_structuring,
            use_api_for_entity_linking=use_api_for_entity_linking,
            use_api_for_retrieval=use_api_for_specialists,
            use_api_for_decision=use_api_for_specialists,
        )

    def _dispatch_clarification(
        self,
        specialty_id: str,
        question: str,
        use_api_for_clarification: bool,
        patient_input: dict[str, Any] | str,
        specialty_opinions: dict[str, Any],
    ) -> dict[str, Any]:
        adapter = self.specialists[specialty_id]
        return adapter.clarify_case(
            patient_input=patient_input,
            specialty_evaluation=specialty_opinions[specialty_id],
            question=question,
            use_api_for_clarification=use_api_for_clarification,
        )

    def _collect_uncertainties(self, specialty_opinions: dict[str, Any]) -> dict[str, Any]:
        unresolved_items: list[dict[str, Any]] = []
        primary_markers: dict[str, list[str]] = {}

        for specialty_id, payload in specialty_opinions.items():
            decision_result = payload.get("decision_result", {})
            retrieval_result = payload.get("retrieval_result", {})

            for item in self._string_list(decision_result.get("need_more_info")):
                unresolved_items.append(
                    {
                        "source": specialty_id,
                        "type": "missing_information",
                        "question": item,
                        "reason": f"{payload.get('specialty_label')}认为当前缺少该信息。",
                        "priority": "high",
                    }
                )

            for item in self._string_list(decision_result.get("risk_flags")):
                unresolved_items.append(
                    {
                        "source": specialty_id,
                        "type": "risk_flag",
                        "question": f"请进一步说明风险点：{item}",
                        "reason": f"{payload.get('specialty_label')}提示存在高风险事项。",
                        "priority": "medium",
                    }
                )

            for item in self._string_list(retrieval_result.get("missing_information")):
                unresolved_items.append(
                    {
                        "source": specialty_id,
                        "type": "retrieval_gap",
                        "question": item,
                        "reason": f"{payload.get('specialty_label')}知识检索阶段提示该字段缺失。",
                        "priority": "medium",
                    }
                )

            primary_marker = self._extract_primary_marker(decision_result)
            if primary_marker:
                primary_markers.setdefault(primary_marker, []).append(specialty_id)

        potential_conflicts = [
            marker for marker, owners in primary_markers.items() if len(set(owners)) == 1
        ]
        return {
            "unresolved_items": self._dedupe_unresolved_items(unresolved_items),
            "potential_conflicts": potential_conflicts,
            "summary": f"共发现 {len(self._dedupe_unresolved_items(unresolved_items))} 个待澄清点。",
        }

    def _integrate_with_api(
        self,
        patient_input: dict[str, Any] | str,
        specialty_opinions: dict[str, Any],
        uncertainty_review: dict[str, Any],
        mdt_follow_up: dict[str, Any] | None,
    ) -> dict[str, Any]:
        panel_opinions: list[dict[str, Any]] = []
        successful_reviews = 0
        max_workers = min(HEAD_DOCTOR_REVIEW_MAX_WORKERS, len(HEAD_DOCTOR_REVIEW_MODELS))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_by_model = {
                model_key: executor.submit(
                    self._review_with_model,
                    model_key=model_key,
                    patient_input=patient_input,
                    specialty_opinions=specialty_opinions,
                    uncertainty_review=uncertainty_review,
                    mdt_follow_up=mdt_follow_up,
                )
                for model_key in HEAD_DOCTOR_REVIEW_MODELS
            }
            for model_key in HEAD_DOCTOR_REVIEW_MODELS:
                future = future_by_model[model_key]
                try:
                    panel_opinions.append(future.result())
                    successful_reviews += 1
                except Exception as exc:
                    panel_opinions.append(
                        {
                            "review_model": MODEL_CONFIGS[model_key].name,
                            "review_model_key": model_key,
                            "case_summary": f"{model_key} 复核失败。",
                            "specialty_consensus": [],
                            "key_risks": [],
                            "final_plan": [],
                            "next_steps": [],
                            "unresolved_issues": [f"{model_key} 复核失败：{exc}"],
                            "specific_surgical_plan": [],
                            "medical_record_plan": "",
                            "safety_boundary": "",
                            "thinking_log": f"{model_key} 复核超时或失败，已降级继续。",
                            "error": str(exc),
                        }
                    )

        if successful_reviews == 0:
            local_review = self._integrate_locally(
                patient_input=patient_input,
                specialty_opinions=specialty_opinions,
                uncertainty_review=uncertainty_review,
                mdt_follow_up=mdt_follow_up,
            )
            local_review["case_summary"] = str(local_review.get("case_summary") or "") + " Qwen/DeepSeek 复核均失败，已回退本地整合。"
            local_review["model_panel_opinions"] = panel_opinions
            return self._format_final_recommendation(
                specialty_opinions=specialty_opinions,
                uncertainty_review=uncertainty_review,
                mdt_follow_up=mdt_follow_up,
                final_review=local_review,
            )

        try:
            merged_review = self._merge_panel_opinions(
                patient_input=patient_input,
                specialty_opinions=specialty_opinions,
                uncertainty_review=uncertainty_review,
                mdt_follow_up=mdt_follow_up,
                panel_opinions=panel_opinions,
            )
        except Exception:
            merged_review = self._synthesize_panel_opinions(
                specialty_opinions=specialty_opinions,
                uncertainty_review=uncertainty_review,
                mdt_follow_up=mdt_follow_up,
                panel_opinions=panel_opinions,
            )
        merged_review["model_panel_opinions"] = [dict(item) for item in panel_opinions]
        return self._format_final_recommendation(
            specialty_opinions=specialty_opinions,
            uncertainty_review=uncertainty_review,
            mdt_follow_up=mdt_follow_up,
            final_review=merged_review,
        )

    def _review_with_model(
        self,
        model_key: str,
        patient_input: dict[str, Any] | str,
        specialty_opinions: dict[str, Any],
        uncertainty_review: dict[str, Any],
        mdt_follow_up: dict[str, Any] | None,
    ) -> dict[str, Any]:
        prompt = (
            "你是医院的 head_doctor，需要整合多个专科 agent 的初始结论，以及 mdt_call 的二次分诊补充意见，"
            "形成一个统一的最终诊疗方案结论，并给出完整的思考日志。请只返回 JSON，字段固定为："
            "case_summary, specialty_consensus, key_risks, final_plan, next_steps, unresolved_issues, "
            "medical_record_plan, safety_boundary, thinking_log, "
            "current_status_level, surgery_channel_open, core_constraints, module_assessments, specific_surgical_plan。\n"
            "- specialty_consensus, key_risks, final_plan, next_steps, unresolved_issues, core_constraints 必须是字符串数组。\n"
            "- specific_surgical_plan 必须是字符串数组，按术前/术中/术后给出可执行步骤，每条一句话。\n"
            "- final_plan 中直接给出最终方案要点，语言简明扼要，避免冗长介绍。\n"
            "- medical_record_plan 必须是一段可直接写入病历/会诊记录的中文。\n"
            "- thinking_log 必须是一段中文，清晰说明你是如何权衡各专科结论、判断冲突、优先风险，并说明是否通过 mdt_call 追问补充不确定信息。\n"
            "- current_status_level 只能是 READY / OPTIMIZE / CONTRAINDICATED 之一，不允许模糊表述。\n"
            "- surgery_channel_open 只能是 yes 或 no。\n"
            "- core_constraints 最多 3 条，只保留真正影响当前手术决策的限制因素。\n"
            "- module_assessments 必须是数组，每项字段固定为 module, status, risk_level, evidence。"
            " risk_level 只能是 low / medium / high；evidence 必须是字符串数组并引用输入数据。\n"
            "- 若仍有信息不足，请明确写入 unresolved_issues，不要编造。但是，即使信息不足，请根据目前的状况先得出一个最合适的结论和决策。\n\n"
            f"患者输入：\n{json.dumps(patient_input, ensure_ascii=False, indent=2) if isinstance(patient_input, dict) else patient_input}\n\n"
            f"专科结论：\n{json.dumps(specialty_opinions, ensure_ascii=False, indent=2)}\n\n"
            f"存疑点汇总：\n{json.dumps(uncertainty_review, ensure_ascii=False, indent=2)}\n\n"
            f"mdt_call 二次分诊结果：\n{json.dumps(mdt_follow_up, ensure_ascii=False, indent=2)}"
        )
        response = self.api_client.chat_json(MODEL_CONFIGS[model_key], prompt)
        payload = self.api_client.extract_json(response)
        payload["review_model"] = MODEL_CONFIGS[model_key].name
        payload["review_model_key"] = model_key
        return payload

    def _format_final_recommendation(
        self,
        specialty_opinions: dict[str, Any],
        uncertainty_review: dict[str, Any],
        mdt_follow_up: dict[str, Any] | None,
        final_review: dict[str, Any],
    ) -> dict[str, Any]:
        if "thinking_log" not in final_review:
            final_review["thinking_log"] = (
                "已统一由 DeepSeek 生成最终结论，详见输出中的 case_summary 和最终方案。"
            )
        final_review["merged_by"] = final_review.get("merge_model") or final_review.get("review_model") or final_review.get("review_model_key") or "deepseek_v3_2"
        final_review["specialty_opinions"] = specialty_opinions
        final_review["uncertainty_review"] = uncertainty_review
        final_review["mdt_follow_up"] = mdt_follow_up
        final_review["uncertainty_flow"] = self._build_uncertainty_flow(
            uncertainty_review=uncertainty_review,
            mdt_follow_up=mdt_follow_up,
        )
        self._normalize_current_status_fields(final_review, specialty_opinions)
        self._normalize_specific_surgical_plan(final_review)
        return final_review

    def _merge_panel_opinions(
        self,
        patient_input: dict[str, Any] | str,
        specialty_opinions: dict[str, Any],
        uncertainty_review: dict[str, Any],
        mdt_follow_up: dict[str, Any] | None,
        panel_opinions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        prompt = (
            "你是医院的 head_doctor，现在手上有多个不同模型对专科结论的审阅结果，"
            "请将这些模型意见统一合成一个简洁的最终诊疗方案，并给出思考日志。请只返回 JSON，字段固定为："
            "case_summary, specialty_consensus, key_risks, final_plan, next_steps, unresolved_issues, "
            "medical_record_plan, safety_boundary, thinking_log, "
            "current_status_level, surgery_channel_open, core_constraints, module_assessments, specific_surgical_plan。\n"
            "- final_plan 必须直接给出方案要点，语言简明扼要。\n"
            "- specific_surgical_plan 必须按术前/术中/术后给出可执行步骤，避免泛化总结。\n"
            "- thinking_log 必须说明你如何比较各模型意见、识别冲突、优先风险，以及如何基于 mdt_call 追问结果形成最终结论。\n"
            "- current_status_level 只能是 READY / OPTIMIZE / CONTRAINDICATED 之一。\n"
            "- surgery_channel_open 只能是 yes 或 no。\n"
            "- core_constraints 最多 3 条，必须是真正影响当前手术通道判断的限制因素。\n"
            "- module_assessments 必须保留各模块分层结论，字段固定为 module, status, risk_level, evidence。\n"
            "- 若仍有信息不足，请明确写入 unresolved_issues，不要编造。但是，即使信息不足，请根据目前的状况先得出一个最合适的结论和决策，合成一个简洁的最终治疗方案。\n\n"
            f"患者输入：\n{json.dumps(patient_input, ensure_ascii=False, indent=2) if isinstance(patient_input, dict) else patient_input}\n\n"
            f"专科结论：\n{json.dumps(specialty_opinions, ensure_ascii=False, indent=2)}\n\n"
            f"存疑点汇总：\n{json.dumps(uncertainty_review, ensure_ascii=False, indent=2)}\n\n"
            f"mdt_call 二次分诊结果：\n{json.dumps(mdt_follow_up, ensure_ascii=False, indent=2)}\n\n"
            f"各模型审阅结果：\n{json.dumps(panel_opinions, ensure_ascii=False, indent=2)}"
        )
        response = self.api_client.chat_json(MODEL_CONFIGS["deepseek_v3_2"], prompt)
        payload = self.api_client.extract_json(response)
        payload["merge_model"] = MODEL_CONFIGS["deepseek_v3_2"].name
        return payload

    def _normalize_current_status_fields(
        self,
        final_review: dict[str, Any],
        specialty_opinions: dict[str, Any],
    ) -> None:
        allowed_levels = {"READY", "OPTIMIZE", "CONTRAINDICATED"}
        level = str(final_review.get("current_status_level") or "").strip().upper()
        unresolved = self._string_list(final_review.get("unresolved_issues"))
        key_risks = self._string_list(final_review.get("key_risks"))

        if level not in allowed_levels:
            level = "OPTIMIZE" if unresolved else "READY"
            if any("禁忌" in item or "contraind" in item.lower() for item in key_risks):
                level = "CONTRAINDICATED"
        final_review["current_status_level"] = level

        channel_raw = str(final_review.get("surgery_channel_open") or "").strip().lower()
        if channel_raw in ("yes", "true", "1"):
            channel = "yes"
        elif channel_raw in ("no", "false", "0"):
            channel = "no"
        else:
            channel = "yes" if level == "READY" else "no"
        final_review["surgery_channel_open"] = channel

        core_constraints = self._string_list(final_review.get("core_constraints"))
        if not core_constraints:
            core_constraints = unresolved[:3] or key_risks[:3]
        final_review["core_constraints"] = core_constraints[:3]

        module_assessments = self._normalize_module_assessments(
            final_review.get("module_assessments"),
        )
        if not module_assessments:
            module_assessments = self._build_default_module_assessments(specialty_opinions)
        final_review["module_assessments"] = module_assessments[:8]

    def _normalize_specific_surgical_plan(self, final_review: dict[str, Any]) -> None:
        plan_steps = self._string_list(final_review.get("specific_surgical_plan"))
        final_plan = self._string_list(final_review.get("final_plan"))
        next_steps = self._string_list(final_review.get("next_steps"))

        if not plan_steps:
            merged = []
            for item in final_plan + next_steps:
                if item and item not in merged:
                    merged.append(item)
            plan_steps = [f"步骤{i + 1}：{text}" for i, text in enumerate(merged[:8])]
        else:
            plan_steps = [str(item).strip() for item in plan_steps if str(item).strip()][:8]
        final_review["specific_surgical_plan"] = plan_steps

    def _normalize_module_assessments(self, raw: Any) -> list[dict[str, Any]]:
        if not isinstance(raw, list):
            return []
        result: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            module = str(item.get("module") or item.get("module_name") or "").strip()
            if not module:
                continue
            status = str(item.get("status") or "").strip() or "undetermined"
            risk_level = str(item.get("risk_level") or item.get("risk") or "").strip().lower()
            if risk_level not in ("low", "medium", "high"):
                risk_level = "medium"
            evidence = self._string_list(
                item.get("evidence")
                or item.get("basis")
                or item.get("reason")
            )[:3]
            result.append(
                {
                    "module": module,
                    "status": status,
                    "risk_level": risk_level,
                    "evidence": evidence,
                }
            )
        return result

    def _build_default_module_assessments(self, specialty_opinions: dict[str, Any]) -> list[dict[str, Any]]:
        module_rows: list[dict[str, Any]] = []
        for specialty_id, payload in specialty_opinions.items():
            decision_result = payload.get("decision_result", {})
            retrieval_result = payload.get("retrieval_result", {})
            label = str(payload.get("specialty_label") or specialty_id)

            risk_flags = self._string_list(decision_result.get("risk_flags"))
            missing_info = self._string_list(decision_result.get("need_more_info")) + self._string_list(
                retrieval_result.get("missing_information")
            )

            if missing_info:
                status = "needs_more_information"
                risk_level = "high"
            elif risk_flags:
                status = "risk_noted"
                risk_level = "medium"
            else:
                status = "acceptable_for_current_stage"
                risk_level = "low"

            evidence = self._string_list(
                decision_result.get("preliminary_impression")
                or decision_result.get("primary_condition_id")
                or decision_result.get("primary_plan_id")
            )
            evidence = (evidence + risk_flags + missing_info)[:3]
            module_rows.append(
                {
                    "module": label,
                    "status": status,
                    "risk_level": risk_level,
                    "evidence": evidence,
                }
            )
        return module_rows

    def _build_uncertainty_flow(
        self,
        uncertainty_review: dict[str, Any],
        mdt_follow_up: dict[str, Any] | None,
    ) -> str:
        unresolved_items = uncertainty_review.get("unresolved_items", [])
        if not unresolved_items:
            return "当前无未解决的关键不确定项，直接整合各专科结论形成最终方案。"

        questions = [str(item.get("question") or item.get("reason") or "") for item in unresolved_items]
        questions = [q for q in questions if q and self._is_clinically_meaningful_uncertainty(q)]
        questions = [q for q in questions if q]
        if mdt_follow_up and mdt_follow_up.get("clarifications"):
            clarifications = []
            for clarification in mdt_follow_up.get("clarifications", []):
                specialty = clarification.get("specialty") or "相关专科"
                answer = clarification.get("answer") or "无明确回答"
                if not self._is_clinically_meaningful_uncertainty(answer):
                    continue
                clarifications.append(f"{specialty}：{answer}")
            if not questions and not clarifications:
                return "当前无需要展示的关键不确定项，最终方案已基于可用专科结论完成整合。"
            if questions and clarifications:
                return (
                    "先识别以下待澄清项：" + "；".join(questions) +
                    "。随后通过 MDT 追问获取补充意见，主要结果包括：" + "；".join(clarifications) +
                    "。最终将这些补充结果纳入判断，形成统一方案。"
                )
            if questions:
                return (
                    "先识别以下待澄清项：" + "；".join(questions) +
                    "。随后通过 MDT 追问补充信息，并将结果纳入最终判断。"
                )
            return (
                "通过 MDT 追问获取了以下补充意见：" + "；".join(clarifications) +
                "。最终将这些补充结果纳入判断，形成统一方案。"
            )

        if not questions:
            return "当前无需要展示的关键不确定项，最终方案仍基于现有专科结论和已有风险提示。"
        return (
            "识别到待澄清项：" + "；".join(questions) +
            "。但未产生 MDT 追问结果，最终方案仍基于现有专科结论和已有风险提示。"
        )

    def _is_clinically_meaningful_uncertainty(self, text: Any) -> bool:
        normalized = str(text or "").strip()
        if not normalized:
            return False
        reject_terms = (
            "建议重试",
            "评估结果暂不可用",
            "超时或接口失败",
            "接口失败",
            "api request failed",
            "read operation timed out",
            "当前未返回可用结论",
            "追问暂未返回",
            "error",
            "timeout",
        )
        lowered = normalized.lower()
        return not any(term in normalized or term in lowered for term in reject_terms)

    def _synthesize_panel_opinions(
        self,
        specialty_opinions: dict[str, Any],
        uncertainty_review: dict[str, Any],
        mdt_follow_up: dict[str, Any] | None,
        panel_opinions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        specialty_consensus: list[str] = []
        key_risks: list[str] = []
        final_plan: list[str] = []
        next_steps: list[str] = []
        unresolved_issues: list[str] = []
        specific_surgical_plan: list[str] = []
        medical_record_fragments: list[str] = []
        reviewed_by: list[str] = []

        for opinion in panel_opinions:
            reviewed_by.append(str(opinion.get("review_model") or opinion.get("review_model_key") or "unknown"))
            specialty_consensus.extend(self._string_list(opinion.get("specialty_consensus")))
            key_risks.extend(self._string_list(opinion.get("key_risks")))
            final_plan.extend(self._string_list(opinion.get("final_plan")))
            next_steps.extend(self._string_list(opinion.get("next_steps")))
            unresolved_issues.extend(self._string_list(opinion.get("unresolved_issues")))
            specific_surgical_plan.extend(self._string_list(opinion.get("specific_surgical_plan")))
            fragment = str(opinion.get("medical_record_plan") or "").strip()
            if fragment:
                medical_record_fragments.append(f"[{opinion.get('review_model', 'unknown')}] {fragment}")

        specialty_consensus = list(dict.fromkeys(item for item in specialty_consensus if item))
        key_risks = list(dict.fromkeys(item for item in key_risks if item))
        next_steps = list(dict.fromkeys(item for item in next_steps if item))
        unresolved_issues = list(dict.fromkeys(item for item in unresolved_issues if item))
        final_plan = list(dict.fromkeys(item for item in (final_plan + next_steps) if item))
        specific_surgical_plan = list(dict.fromkeys(item for item in specific_surgical_plan if item))
        if not specific_surgical_plan:
            specific_surgical_plan = final_plan[:8]

        case_summary = (
            f"已完成 {len(panel_opinions)} 个不同模型的 head_doctor 会诊审阅，并汇总 {len(specialty_opinions)} 个专科 agent 结论。"
        )
        if mdt_follow_up:
            case_summary += " 本次汇总已纳入 mdt_call 二次追问结果。"

        if not medical_record_fragments:
            medical_record_fragments.append(
                "综合多模型与多专科智能体意见，当前建议先按共识方案完成风险分层、术前补充检查与围术期准备。"
            )

        return {
            "case_summary": case_summary,
            "reviewed_by_models": reviewed_by,
            "model_panel_opinions": panel_opinions,
            "specialty_consensus": specialty_consensus,
            "key_risks": key_risks,
            "final_plan": final_plan,
            "next_steps": next_steps,
            "unresolved_issues": unresolved_issues,
            "specific_surgical_plan": specific_surgical_plan,
            "medical_record_plan": "\n".join(medical_record_fragments),
            "safety_boundary": "本结果仅作为多模型、多专科会诊辅助建议，不能替代临床医生面诊、查体与正式医嘱。",
        }

    def _integrate_locally(
        self,
        patient_input: dict[str, Any] | str,
        specialty_opinions: dict[str, Any],
        uncertainty_review: dict[str, Any],
        mdt_follow_up: dict[str, Any] | None,
    ) -> dict[str, Any]:
        specialty_consensus: list[str] = []
        key_risks: list[str] = []
        next_steps: list[str] = []
        unresolved = [
            item["question"]
            for item in uncertainty_review.get("unresolved_items", [])
        ]

        for specialty_id, payload in specialty_opinions.items():
            label = payload.get("specialty_label", specialty_id)
            decision_result = payload.get("decision_result", {})
            primary_plan = decision_result.get("primary_plan")
            primary_condition = decision_result.get("primary_condition")

            if primary_condition:
                specialty_consensus.append(
                    f"{label}主要关注 {primary_condition.get('name_zh') or primary_condition.get('id')}"
                )
            elif decision_result.get("primary_condition_id"):
                specialty_consensus.append(
                    f"{label}主要关注 {decision_result.get('primary_condition_id')}"
                )

            if primary_plan:
                specialty_consensus.append(
                    f"{label}建议优先按 {primary_plan.get('name_zh') or primary_plan.get('id')} 路径处理"
                )
            elif decision_result.get("primary_plan_id"):
                specialty_consensus.append(
                    f"{label}建议优先按 {decision_result.get('primary_plan_id')} 路径处理"
                )

            key_risks.extend(self._string_list(decision_result.get("risk_flags")))
            next_steps.extend(self._string_list(decision_result.get("need_more_info")))

        if mdt_follow_up:
            for clarification in mdt_follow_up.get("clarifications", []):
                answer = str(clarification.get("answer") or "").strip()
                if answer:
                    specialty_consensus.append(f"{clarification.get('specialty')}: {answer}")
                next_steps.extend(self._string_list(clarification.get("action_items")))
                unresolved.extend(self._string_list(clarification.get("remaining_uncertainties")))

        specialty_consensus = list(dict.fromkeys(item for item in specialty_consensus if item))
        key_risks = list(dict.fromkeys(item for item in key_risks if item))
        next_steps = list(dict.fromkeys(item for item in next_steps if item))
        unresolved = list(dict.fromkeys(item for item in unresolved if item))
        final_plan = list(dict.fromkeys(specialty_consensus + next_steps))

        case_summary = (
            f"当前已汇总 {len(specialty_opinions)} 个专科 agent 结论，并针对存疑点进行了 mdt_call 回流。"
            if mdt_follow_up
            else f"当前已汇总 {len(specialty_opinions)} 个专科 agent 初始结论。"
        )
        medical_record_plan = (
            "综合多专科智能体意见，当前建议先按各专科首选路径进行风险分层和处置准备，"
            "对仍不确定的关键字段继续通过 MDT 二次分诊补充后，再由 head_doctor 形成最终执行方案。"
        )
        current_status_level = "OPTIMIZE" if unresolved else "READY"
        if any("禁忌" in item or "contraind" in item.lower() for item in key_risks):
            current_status_level = "CONTRAINDICATED"
        specific_surgical_plan = [f"步骤{i + 1}：{text}" for i, text in enumerate((final_plan + next_steps)[:8])]
        return {
            "case_summary": case_summary,
            "specialty_consensus": specialty_consensus,
            "key_risks": key_risks,
            "final_plan": final_plan,
            "next_steps": next_steps,
            "unresolved_issues": unresolved,
            "specific_surgical_plan": specific_surgical_plan,
            "medical_record_plan": medical_record_plan,
            "safety_boundary": "本结果仅作为多智能体会诊辅助建议，不能替代临床医生面诊、查体与正式医嘱。",
            "current_status_level": current_status_level,
            "surgery_channel_open": "yes" if current_status_level == "READY" else "no",
            "core_constraints": (unresolved[:3] or key_risks[:3]),
            "module_assessments": self._build_default_module_assessments(specialty_opinions),
        }

    def _extract_primary_marker(self, decision_result: dict[str, Any]) -> str | None:
        if isinstance(decision_result.get("primary_plan"), dict):
            marker = str(decision_result["primary_plan"].get("id") or "").strip()
            return marker or None
        if decision_result.get("primary_plan_id"):
            marker = str(decision_result.get("primary_plan_id") or "").strip()
            return marker or None
        return None

    def _dedupe_unresolved_items(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[tuple[str, str, str]] = set()
        result: list[dict[str, Any]] = []
        for item in items:
            key = (
                str(item.get("source") or ""),
                str(item.get("type") or ""),
                str(item.get("question") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result

    def _string_list(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []
