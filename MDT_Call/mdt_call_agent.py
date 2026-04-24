from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import re
import socket
import ssl
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib import error, request

from mdt_kb_retriever import MDTKnowledgeRetriever
from mdt_patient_mapper_agent import MDTCallMapper


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
}

MAX_PARALLEL_WORKERS = 4


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

        raise RuntimeError(f"MDT call API request failed after retries: {last_error}") from last_error

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


class MDTCallAgent:
    def __init__(self, api_client: SharedGenAIChatClient | None = None) -> None:
        self.api_client = api_client or SharedGenAIChatClient()
        self.mapper = MDTCallMapper()
        self.retriever = MDTKnowledgeRetriever()

    def coordinate_follow_up(
        self,
        patient_input: dict[str, Any] | str,
        specialty_opinions: dict[str, Any],
        unresolved_items: list[dict[str, Any]],
        specialist_callback: Callable[[str, str, bool, dict[str, Any] | str, dict[str, Any]], dict[str, Any]],
        use_api_for_planning: bool = True,
        use_api_for_clarification: bool = True,
    ) -> dict[str, Any]:
        request_context = self.mapper.build_request_context(
            patient_input=patient_input,
            specialty_opinions=specialty_opinions,
            unresolved_items=unresolved_items,
        )
        dispatch_plan = (
            self._plan_with_api(request_context)
            if use_api_for_planning
            else self.retriever.build_dispatch_plan(request_context)
        )
        clarifications = []
        valid_tasks: list[tuple[str, str]] = []
        for task in dispatch_plan.get("tasks", []):
            specialty_id = str(task.get("specialty") or "").strip()
            question = str(task.get("question") or "").strip()
            if not specialty_id or not question:
                continue
            valid_tasks.append((specialty_id, question))

        if valid_tasks:
            max_workers = min(MAX_PARALLEL_WORKERS, len(valid_tasks))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_items = [
                    (
                        specialty_id,
                        question,
                        executor.submit(
                            specialist_callback,
                            specialty_id,
                            question,
                            use_api_for_clarification,
                            patient_input,
                            specialty_opinions,
                        ),
                    )
                    for specialty_id, question in valid_tasks
                ]
                for specialty_id, question, future in future_items:
                    try:
                        clarifications.append(future.result())
                    except Exception as exc:
                        clarifications.append(
                            {
                                "specialty": specialty_id,
                                "question": question,
                                "answer": f"{specialty_id} 追问暂未返回（超时或接口失败）。",
                                "action_items": [],
                                "remaining_uncertainties": [question],
                                "confidence": "low",
                                "error": str(exc),
                            }
                        )

        return {
            "request_context": request_context,
            "dispatch_plan": dispatch_plan,
            "clarifications": clarifications,
            "summary": f"mdt_call 共下派 {len(dispatch_plan.get('tasks', []))} 个追问任务，并回收 {len(clarifications)} 份补充意见。",
        }

    def coordinate_initial_triage(
        self,
        patient_input: dict[str, Any] | str,
        specialist_callback: Callable[[str, dict[str, Any] | str], dict[str, Any]],
        use_api_for_planning: bool = True,
    ) -> dict[str, Any]:
        request_context = self.mapper.build_initial_context(patient_input)
        dispatch_plan = (
            self._plan_initial_with_api(request_context)
            if use_api_for_planning
            else self.retriever.build_initial_triage_plan(request_context)
        )
        specialty_opinions: dict[str, Any] = {}
        specialty_ids: list[str] = []
        failed_specialties: list[str] = []
        for task in dispatch_plan.get("tasks", []):
            specialty_id = str(task.get("specialty") or "").strip()
            if not specialty_id or specialty_id in specialty_ids:
                continue
            specialty_ids.append(specialty_id)

        if specialty_ids:
            max_workers = min(MAX_PARALLEL_WORKERS, len(specialty_ids))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_by_specialty = {
                    specialty_id: executor.submit(specialist_callback, specialty_id, patient_input)
                    for specialty_id in specialty_ids
                }
                for specialty_id in specialty_ids:
                    try:
                        specialty_opinions[specialty_id] = future_by_specialty[specialty_id].result()
                    except Exception as exc:
                        failed_specialties.append(specialty_id)
                        specialty_opinions[specialty_id] = {
                            "specialty": specialty_id,
                            "specialty_label": specialty_id,
                            "structured_patient": {},
                            "normalized_patient": {},
                            "entity_linking": {},
                            "retrieval_result": {
                                "missing_information": [f"{specialty_id} 评估超时或接口失败：{exc}"],
                            },
                            "decision_result": {
                                "need_more_info": [f"{specialty_id} 当前未返回可用结论，建议重试。"],
                                "risk_flags": [f"{specialty_id} 评估结果暂不可用。"],
                            },
                            "error": str(exc),
                        }

        summary = f"mdt_call 首轮分诊触发了 {len(specialty_opinions)} 个专科 agent。"
        if failed_specialties:
            summary += f" 其中 {len(failed_specialties)} 个专科超时/失败，已保留占位结果：{', '.join(failed_specialties)}。"
        return {
            "request_context": request_context,
            "dispatch_plan": dispatch_plan,
            "specialty_opinions": specialty_opinions,
            "summary": summary,
        }

    def _plan_with_api(self, request_context: dict[str, Any]) -> dict[str, Any]:
        prompt = (
            "你是 MDT 分诊台 mdt_call，需要根据 head_doctor 汇总出的存疑点，把问题再次下派给最合适的专科 agent。\n"
            "请只返回 JSON，字段固定为：tasks, rationale, priority_summary。\n"
            "- tasks 必须是数组，每个元素包含 specialty, question, reason, priority。\n"
            "- specialty 只能从 anesthesia, cardiology, hepatobiliary, neurosurgery, orthopaedics 中选择。\n"
            "- 若多个问题应发给同一专科，可以拆成多个任务。\n"
            "- 不要把最终诊疗结论写在这里，只负责分诊。\n\n"
            f"输入上下文：\n{json.dumps(request_context, ensure_ascii=False, indent=2)}"
        )
        response = self.api_client.chat_json(MODEL_CONFIGS["gpt_5_2"], prompt)
        payload = self.api_client.extract_json(response)
        payload.setdefault("tasks", [])
        return payload

    def _plan_initial_with_api(self, request_context: dict[str, Any]) -> dict[str, Any]:
        prompt = (
            "你是 MDT 分诊台 mdt_call，需要根据患者首轮信息决定应先触发哪些专科 agent。\\n"
            "请只返回 JSON，字段固定为：tasks, rationale, priority_summary。\\n"
            "- tasks 必须是数组，每个元素包含 specialty, reason, priority。\\n"
            "- specialty 只能从 anesthesia, cardiology, hepatobiliary, neurosurgery, orthopaedics 中选择。\\n"
            "- 这是首轮分诊，不负责最终结论。\\n\\n"
            f"输入上下文：\\n{json.dumps(request_context, ensure_ascii=False, indent=2)}"
        )
        response = self.api_client.chat_json(MODEL_CONFIGS["gpt_5_2"], prompt)
        payload = self.api_client.extract_json(response)
        payload.setdefault("tasks", [])
        return payload
