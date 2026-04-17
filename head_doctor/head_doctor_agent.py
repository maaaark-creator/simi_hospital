from __future__ import annotations

import importlib.util
import json
import re
import socket
import ssl
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
MDT_CALL_DIR = PROJECT_ROOT / "mdt_call"


GENAI_API_URL = "https://genaiapi.shanghaitech.edu.cn/api/v1/start"


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model: str
    api_key: str


MODEL_CONFIGS = {
    "gpt_5_2": ModelConfig(
        name="GPT-5.2",
        model="GPT-5.2",
        api_key="bb336cff66f54e7a9d6f48b3dba97657",
    ),
    "qwen3": ModelConfig(
        name="Qwen3",
        model="qwen-instruct",
        api_key="791e88f506f441ba8185adb3a8a9f98a",
    ),
    "deepseek_r1": ModelConfig(
        name="deepseek-r1",
        model="deepseek-r1:671b",
        api_key="e693397f5e1e41259f8e3bef4e502ca4",
    ),
}

HEAD_DOCTOR_REVIEW_MODELS = ("gpt_5_2", "qwen3", "deepseek_r1")


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
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with request.urlopen(req, timeout=90) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except (error.URLError, ssl.SSLError, TimeoutError, socket.timeout) as exc:
                last_error = exc
                if attempt == 2:
                    break
                time.sleep(1.5 * (attempt + 1))

        raise RuntimeError(f"Head doctor API request failed after retries: {last_error}") from last_error

    def extract_text(self, response: dict[str, Any]) -> str:
        choices = response.get("choices", [])
        if not choices:
            raise ValueError("API response does not contain choices.")
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
        structured_patient, normalized_patient = self._prepare_patient(
            patient_input,
            use_api_for_structuring=use_api_for_structuring,
        )
        entity_linking = (
            self.mapper.call_medical_entity_linking_api(normalized_patient)
            if use_api_for_entity_linking
            else {}
        )
        retrieval_result = self._retrieve_case(
            normalized_patient,
            entity_linking,
            use_api_for_retrieval=use_api_for_retrieval,
        )
        decision_result = (
            self.decision_agent.decide_with_api(normalized_patient, retrieval_result)
            if use_api_for_decision
            else self.decision_agent.decide(normalized_patient, retrieval_result)
        )
        return {
            "specialty": self.specialty_id,
            "specialty_label": self.label,
            "structured_patient": structured_patient,
            "normalized_patient": normalized_patient,
            "entity_linking": entity_linking,
            "retrieval_result": retrieval_result,
            "decision_result": decision_result,
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
                "- action_items 和 remaining_uncertainties 必须是字符串数组。\n"
                "- 若仍不能确认，请明确写入 remaining_uncertainties。\n\n"
                f"患者信息：\n{json.dumps(patient_input, ensure_ascii=False, indent=2) if isinstance(patient_input, dict) else patient_input}\n\n"
                f"既有专科结论：\n{json.dumps(specialty_evaluation, ensure_ascii=False, indent=2)}\n\n"
                f"追问问题：\n{question}"
            )
            response = self.api_client.chat_json(MODEL_CONFIGS["gpt_5_2"], prompt)
            payload = self.api_client.extract_json(response)
            payload.setdefault("specialty", self.specialty_id)
            payload.setdefault("question", question)
            payload.setdefault("action_items", [])
            payload.setdefault("remaining_uncertainties", [])
            return payload

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
    ) -> dict[str, Any]:
        if use_api_for_retrieval and hasattr(self.retriever, "retrieve_with_api"):
            return self.retriever.retrieve_with_api(normalized_patient, entity_linking)
        return self.retriever.retrieve(normalized_patient, entity_linking)

    def _prepare_patient(
        self,
        patient_input: dict[str, Any] | str,
        use_api_for_structuring: bool,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if isinstance(patient_input, dict):
            structured_patient = patient_input
        elif use_api_for_structuring:
            structured_patient = self.mapper.call_patient_structuring_api(patient_input)
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
        initial_triage = self.mdt_call_agent.coordinate_initial_triage(
            patient_input=patient_input,
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
                patient_input=patient_input,
                specialty_opinions=specialty_opinions,
                unresolved_items=uncertainty_review["unresolved_items"],
                specialist_callback=self._dispatch_clarification,
                use_api_for_planning=use_api_for_mdt,
                use_api_for_clarification=use_api_for_clarification,
            )

        final_recommendation = (
            self._integrate_with_api(
                patient_input=patient_input,
                specialty_opinions=specialty_opinions,
                uncertainty_review=uncertainty_review,
                mdt_follow_up=mdt_follow_up,
            )
            if use_api_for_final
            else self._integrate_locally(
                patient_input=patient_input,
                specialty_opinions=specialty_opinions,
                uncertainty_review=uncertainty_review,
                mdt_follow_up=mdt_follow_up,
            )
        )

        return {
            "patient_input": patient_input,
            "initial_triage": initial_triage,
            "specialty_opinions": specialty_opinions,
            "uncertainty_review": uncertainty_review,
            "mdt_follow_up": mdt_follow_up,
            "head_doctor_recommendation": final_recommendation,
        }

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
        panel_opinions = [
            self._review_with_model(
                model_key=model_key,
                patient_input=patient_input,
                specialty_opinions=specialty_opinions,
                uncertainty_review=uncertainty_review,
                mdt_follow_up=mdt_follow_up,
            )
            for model_key in HEAD_DOCTOR_REVIEW_MODELS
        ]
        merged_review = self._merge_panel_opinions(
            patient_input=patient_input,
            specialty_opinions=specialty_opinions,
            uncertainty_review=uncertainty_review,
            mdt_follow_up=mdt_follow_up,
            panel_opinions=panel_opinions,
        )
        merged_review["model_panel_opinions"] = panel_opinions
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
            "case_summary, specialty_consensus, key_risks, final_plan, next_steps, unresolved_issues, medical_record_plan, safety_boundary, thinking_log。\n"
            "- specialty_consensus, key_risks, final_plan, next_steps, unresolved_issues 必须是字符串数组。\n"
            "- final_plan 中直接给出最终方案要点，语言简明扼要，避免冗长介绍。\n"
            "- medical_record_plan 必须是一段可直接写入病历/会诊记录的中文。\n"
            "- thinking_log 必须是一段中文，清晰说明你是如何权衡各专科结论、判断冲突、优先风险，并说明是否通过 mdt_call 追问补充不确定信息。\n"
            "- 若仍有信息不足，请明确写入 unresolved_issues，不要编造。\n\n"
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
                "已统一由 GPT 生成最终结论，详见输出中的 case_summary 和最终方案。"
            )
        final_review["merged_by"] = final_review.get("merge_model") or final_review.get("review_model") or final_review.get("review_model_key") or "gpt"
        final_review["specialty_opinions"] = specialty_opinions
        final_review["uncertainty_review"] = uncertainty_review
        final_review["mdt_follow_up"] = mdt_follow_up
        final_review["uncertainty_flow"] = self._build_uncertainty_flow(
            uncertainty_review=uncertainty_review,
            mdt_follow_up=mdt_follow_up,
        )
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
            "case_summary, specialty_consensus, key_risks, final_plan, next_steps, unresolved_issues, medical_record_plan, safety_boundary, thinking_log。\n"
            "- final_plan 必须直接给出方案要点，语言简明扼要。\n"
            "- thinking_log 必须说明你如何比较各模型意见、识别冲突、优先风险，以及如何基于 mdt_call 追问结果形成最终结论。\n"
            "- 若仍有信息不足，请明确写入 unresolved_issues，不要编造。\n\n"
            f"患者输入：\n{json.dumps(patient_input, ensure_ascii=False, indent=2) if isinstance(patient_input, dict) else patient_input}\n\n"
            f"专科结论：\n{json.dumps(specialty_opinions, ensure_ascii=False, indent=2)}\n\n"
            f"存疑点汇总：\n{json.dumps(uncertainty_review, ensure_ascii=False, indent=2)}\n\n"
            f"mdt_call 二次分诊结果：\n{json.dumps(mdt_follow_up, ensure_ascii=False, indent=2)}\n\n"
            f"各模型审阅结果：\n{json.dumps(panel_opinions, ensure_ascii=False, indent=2)}"
        )
        response = self.api_client.chat_json(MODEL_CONFIGS["gpt_5_2"], prompt)
        payload = self.api_client.extract_json(response)
        payload["merge_model"] = MODEL_CONFIGS["gpt_5_2"].name
        return payload

    def _build_uncertainty_flow(
        self,
        uncertainty_review: dict[str, Any],
        mdt_follow_up: dict[str, Any] | None,
    ) -> str:
        unresolved_items = uncertainty_review.get("unresolved_items", [])
        if not unresolved_items:
            return "当前无未解决的关键不确定项，直接整合各专科结论形成最终方案。"

        questions = [str(item.get("question") or item.get("reason") or "") for item in unresolved_items]
        questions = [q for q in questions if q]
        if mdt_follow_up and mdt_follow_up.get("clarifications"):
            clarifications = []
            for clarification in mdt_follow_up.get("clarifications", []):
                specialty = clarification.get("specialty") or "相关专科"
                answer = clarification.get("answer") or "无明确回答"
                clarifications.append(f"{specialty}：{answer}")
            return (
                "先识别以下待澄清项：" + "；".join(questions) +
                "。随后通过 MDT 追问获取补充意见，主要结果包括：" + "；".join(clarifications) +
                "。最终将这些补充结果纳入判断，形成统一方案。"
            )

        return (
            "识别到待澄清项：" + "；".join(questions) +
            "。但未产生 MDT 追问结果，最终方案仍基于现有专科结论和已有风险提示。"
        )

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
        medical_record_fragments: list[str] = []
        reviewed_by: list[str] = []

        for opinion in panel_opinions:
            reviewed_by.append(str(opinion.get("review_model") or opinion.get("review_model_key") or "unknown"))
            specialty_consensus.extend(self._string_list(opinion.get("specialty_consensus")))
            key_risks.extend(self._string_list(opinion.get("key_risks")))
            final_plan.extend(self._string_list(opinion.get("final_plan")))
            next_steps.extend(self._string_list(opinion.get("next_steps")))
            unresolved_issues.extend(self._string_list(opinion.get("unresolved_issues")))
            fragment = str(opinion.get("medical_record_plan") or "").strip()
            if fragment:
                medical_record_fragments.append(f"[{opinion.get('review_model', 'unknown')}] {fragment}")

        specialty_consensus = list(dict.fromkeys(item for item in specialty_consensus if item))
        key_risks = list(dict.fromkeys(item for item in key_risks if item))
        next_steps = list(dict.fromkeys(item for item in next_steps if item))
        unresolved_issues = list(dict.fromkeys(item for item in unresolved_issues if item))
        final_plan = list(dict.fromkeys(item for item in (final_plan + next_steps) if item))

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
        return {
            "case_summary": case_summary,
            "specialty_consensus": specialty_consensus,
            "key_risks": key_risks,
            "final_plan": final_plan,
            "next_steps": next_steps,
            "unresolved_issues": unresolved,
            "medical_record_plan": medical_record_plan,
            "safety_boundary": "本结果仅作为多智能体会诊辅助建议，不能替代临床医生面诊、查体与正式医嘱。",
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
