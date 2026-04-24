from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from head_doctor.head_doctor_agent import HeadDoctorAgent
from final_surgical_plan_memory_agent import FinalSurgicalPlanMemoryAgent


HTML_PATH = PROJECT_ROOT / "mdt_chat_page_v2.html"
JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
SESSIONS: dict[str, dict[str, Any]] = {}
SESSIONS_LOCK = threading.Lock()
PLAN_AGENT = FinalSurgicalPlanMemoryAgent()
PLAN_AGENT_LOCK = threading.Lock()
SPECIALTY_LABELS = {
    "anesthesia": "麻醉科",
    "cardiology": "心内科",
    "hepatobiliary": "肝胆胰外科",
    "neurosurgery": "神经外科",
    "orthopaedics": "骨科",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve an interactive MDT chat page.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8787, help="Bind port.")
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not auto-open the chat page in a browser.",
    )
    return parser


def build_answer_text(workflow: dict[str, Any]) -> str:
    final = workflow.get("head_doctor_recommendation", {})
    current_assessment = str(final.get("current_assessment") or "").strip()
    readiness_status = str(final.get("readiness_status") or "").strip()
    priority_actions = _string_list(final.get("priority_actions"))
    critical_risks = _string_list(final.get("critical_risks") or final.get("key_risks"))
    specific_surgical_plan = _string_list(final.get("specific_surgical_plan") or final.get("final_plan"))

    lines: list[str] = ["MDT总体结果", ""]
    if current_assessment:
        lines.append("当前判断：")
        lines.append(current_assessment)
        lines.append("")
    if readiness_status:
        lines.append("当前状态：")
        lines.append(readiness_status)
        lines.append("")
    if specific_surgical_plan:
        lines.append("鍏蜂綋鎵嬫湳鏂规锛?")
        lines.extend([f"{idx + 1}. {item}" for idx, item in enumerate(specific_surgical_plan[:8])])
        lines.append("")
    if priority_actions:
        lines.append("当前要点：")
        lines.extend([f"- {item}" for item in priority_actions[:6]])
        lines.append("")
    if critical_risks:
        lines.append("最高风险：")
        lines.extend([f"- {item}" for item in critical_risks[:4]])
    normalized_lines = lines if any(line.strip() for line in lines[2:]) else []
    return "\n".join(normalized_lines) if normalized_lines else "已完成分析，但未返回可展示的结构化结论。"


def _extract_text_fragments(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, (list, tuple, set)):
        parts: list[str] = []
        for item in value:
            parts.extend(_extract_text_fragments(item))
        return parts
    if isinstance(value, dict):
        ordered_parts: list[str] = []
        preferred_keys = (
            "summary",
            "text",
            "body",
            "question",
            "reason",
            "description",
            "preliminary_impression",
            "patient_summary",
            "primary_condition_id",
            "primary_plan_id",
            "urgency_level",
            "surgical_readiness",
            "title",
            "label",
            "name_zh",
            "name",
            "term",
            "value",
            "source_text",
            "items",
        )
        for key in preferred_keys:
            if key in value:
                ordered_parts.extend(_extract_text_fragments(value.get(key)))
        return ordered_parts
    return []


def _string_list(value: Any) -> list[str]:
    return _dedupe_keep_order(_extract_text_fragments(value))


def _dedupe_keep_order(items: list[str], limit: int | None = None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
        if limit is not None and len(result) >= limit:
            break
    return result


def _pick_first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _looks_blocking_issue(text: str) -> bool:
    lowered = text.lower()
    blocking_terms = (
        "不明",
        "未明",
        "未完成",
        "未明确",
        "无法",
        "不能",
        "需补",
        "需明确",
        "需先",
        "待补",
        "缺乏",
        "缺少",
        "insufficient",
        "cannot",
        "unable",
        "need more",
        "not ready",
    )
    return any(term in lowered for term in blocking_terms)


def _watchpoint_priority_score(text: str) -> int:
    normalized = str(text or "").strip()
    if not normalized:
        return -1
    lowered = normalized.lower()
    score = 0

    phase_weights = (
        ("[\u672f\u540e]", 4),
        ("\u672f\u540e", 4),
        ("post-op", 4),
        ("postop", 4),
        ("[\u672f\u4e2d]", 3),
        ("\u672f\u4e2d", 3),
        ("intra-op", 3),
        ("[\u672f\u524d]", 2),
        ("\u672f\u524d", 2),
        ("pre-op", 2),
    )
    for marker, weight in phase_weights:
        if marker in lowered:
            score += weight
            break

    high_risk_terms = (
        "\u98ce\u9669",  # ??
        "\u51fa\u8840",  # ??
        "\u51dd\u8840",  # ??
        "\u4f4e\u8840\u538b",  # ???
        "\u9ad8\u8840\u538b",  # ???
        "\u5931\u4ee3\u507f",  # ???
        "\u5e76\u53d1\u75c7",  # ???
        "\u611f\u67d3",  # ??
        "\u75ab\u75eb",  # ??
        "\u6c14\u9053",  # ??
        "\u8f93\u8840",  # ??
        "\u7d27\u6025",  # ??
        "\u6025\u8bca",  # ??
        "icu",
        "pacu",
        "mep",
        "ssep",
        "cpp",
        "icp",
    )
    for term in high_risk_terms:
        if term in lowered:
            score += 2

    actionable_terms = (
        "\u7acb\u5373",  # ??
        "\u89e6\u53d1",  # ??
        "\u542f\u52a8",  # ??
        "\u9608\u503c",  # ??
        "\u9884\u6848",  # ??
        "\u76ee\u6807",  # ??
        "\u76d1\u6d4b",  # ??
        "\u590d\u67e5",  # ??
        "\u8bc4\u4f30",  # ??
        "\u7ea0\u6b63",  # ??
        "\u5347\u538b",  # ??
        "\u505c\u836f",  # ??
        "owner:",
        "trigger:",
    )
    for term in actionable_terms:
        if term in lowered:
            score += 1

    return score


def _select_top_watchpoints(items: list[str], limit: int = 6) -> list[str]:
    unique_items = _dedupe_keep_order(items)
    if not unique_items:
        return []

    scored: list[tuple[int, int, str]] = []
    for idx, item in enumerate(unique_items):
        score = _watchpoint_priority_score(item)
        scored.append((score, idx, item))

    scored.sort(key=lambda row: (-row[0], row[1]))
    picked = [item for _, _, item in scored[: max(1, limit)]]
    return _dedupe_keep_order(picked, limit=limit)


def _build_specialty_briefs(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    opinions = workflow.get("specialty_opinions", {})
    briefs: list[dict[str, Any]] = []
    for specialty_id, payload in opinions.items():
        decision = payload.get("decision_result", {})
        retrieval = payload.get("retrieval_result", {})
        need_more = _dedupe_keep_order(
            _string_list(decision.get("need_more_info")) + _string_list(retrieval.get("missing_information")),
            limit=3,
        )
        next_actions = _dedupe_keep_order(
            _string_list(decision.get("management_recommendations"))
            + _string_list(decision.get("recommended_workup"))
            + _string_list(decision.get("perioperative_considerations")),
            limit=3,
        )
        unique_points = _dedupe_keep_order(
            _string_list(decision.get("risk_flags"))
            + _string_list(decision.get("diagnostic_considerations"))
            + _string_list(decision.get("recommended_collaboration")),
            limit=3,
        )
        readiness = str(decision.get("surgical_readiness") or "").strip().lower()
        is_blocking = (
            bool(need_more)
            or readiness.startswith("not_")
            or "not_ready" in readiness
            or "needs_more_information" in str(decision.get("urgency_level") or "")
        )
        specialty_label = SPECIALTY_LABELS.get(str(specialty_id), str(payload.get("specialty_label") or specialty_id))
        position = _pick_first_text(
            decision.get("preliminary_impression"),
            decision.get("primary_condition_id"),
            decision.get("primary_plan_id"),
        )
        if not position:
            position = (
                "当前仍需先补齐关键信息后，才能形成本科明确判断。"
                if is_blocking
                else "当前已形成本科初步判断，可并行推进下一阶段处理。"
            )
        briefs.append(
            {
                "specialty": str(specialty_id),
                "specialty_label": specialty_label,
                "diagnosis_or_position": position,
                "missing_key_info": need_more,
                "next_actions": next_actions or ["按本科建议继续完成围术期评估与处置。"],
                "key_point": unique_points[0] if unique_points else "",
            }
        )
    return briefs


def _derive_current_assessment(final: dict[str, Any], has_blocking_issues: bool) -> str:
    status_level = str(final.get("current_status_level") or "").strip().upper()
    if status_level in {"READY", "OPTIMIZE", "CONTRAINDICATED"}:
        level_label = {
            "READY": "可直接手术",
            "OPTIMIZE": "需优化后手术",
            "CONTRAINDICATED": "当前禁忌手术",
        }.get(status_level, status_level)
        channel_raw = str(final.get("surgery_channel_open") or "").strip().lower()
        channel = "是" if channel_raw == "yes" else "否"
        constraints = _string_list(final.get("core_constraints"))
        summary = f"总体等级：{level_label}（{status_level}）；是否可进入手术通道：{channel}"
        if constraints:
            summary += "；核心限制：" + "；".join(constraints[:3])
        return summary

    current_assessment = str(final.get("current_assessment") or "").strip()
    if current_assessment:
        return current_assessment

    specialty_consensus = _string_list(final.get("specialty_consensus"))
    if specialty_consensus:
        return "；".join(specialty_consensus[:2])

    final_plan = _string_list(final.get("final_plan"))
    if final_plan:
        return final_plan[0]

    if has_blocking_issues:
        return "当前已有初步MDT方向，但仍存在关键阻塞点，建议先优化后再进入手术流程。"
    return "当前已形成初步MDT共识，可进入下一阶段处理。"


def _enrich_workflow_for_display(workflow: dict[str, Any]) -> dict[str, Any]:
    final = workflow.get("head_doctor_recommendation")
    if not isinstance(final, dict):
        return workflow

    key_risks = _string_list(final.get("key_risks"))
    next_steps = _string_list(final.get("next_steps"))
    final_plan = _string_list(final.get("final_plan"))
    unresolved = _string_list(final.get("unresolved_issues"))
    uncertainty_items = workflow.get("uncertainty_review", {}).get("unresolved_items", [])
    specialty_briefs = _build_specialty_briefs(workflow)

    blocking_candidates = list(unresolved)
    for item in uncertainty_items:
        question = str(item.get("question") or "").strip()
        if question and (_looks_blocking_issue(question) or str(item.get("priority") or "").lower() == "high"):
            blocking_candidates.append(question)
    for brief in specialty_briefs:
        for data_needed in brief.get("missing_key_info", []):
            if data_needed and _looks_blocking_issue(str(data_needed)):
                blocking_candidates.append(str(data_needed))

    deduped_blocking = _dedupe_keep_order(blocking_candidates, limit=5)
    status_level = str(final.get("current_status_level") or "").strip().upper()
    if status_level not in {"READY", "OPTIMIZE", "CONTRAINDICATED"}:
        status_level = "OPTIMIZE" if deduped_blocking else "READY"

    surgery_channel_open = str(final.get("surgery_channel_open") or "").strip().lower()
    if surgery_channel_open not in {"yes", "no"}:
        surgery_channel_open = "yes" if status_level == "READY" else "no"

    core_constraints = _dedupe_keep_order(
        _string_list(final.get("core_constraints")) + deduped_blocking + unresolved,
        limit=3,
    )

    module_lines: list[str] = []
    module_assessments = final.get("module_assessments")
    if isinstance(module_assessments, list):
        for row in module_assessments:
            if not isinstance(row, dict):
                continue
            module = str(row.get("module") or row.get("module_name") or "").strip()
            if not module:
                continue
            risk_level = str(row.get("risk_level") or "").strip().lower() or "medium"
            status = str(row.get("status") or "").strip() or "undetermined"
            risk_label = {"low": "低", "medium": "中", "high": "高"}.get(risk_level, risk_level)
            status_label = {
                "needs_more_information": "信息待补充",
                "risk_noted": "已识别风险",
                "acceptable_for_current_stage": "当前阶段可接受",
                "undetermined": "待明确",
            }.get(status, status)
            module_lines.append(f"{module}：风险{risk_label} / {status_label}")

    readiness_status = str(final.get("readiness_status") or "").strip()
    if not readiness_status:
        readiness_status = (
            "分层评估：" + " | ".join(module_lines[:5])
            if module_lines
            else (
                "分层评估：当前仍有关键限制，建议先优化后手术。"
                if status_level in {"OPTIMIZE", "CONTRAINDICATED"}
                else "分层评估：当前未见新的关键阻塞项，可继续推进。"
            )
        )

    priority_actions = _dedupe_keep_order(core_constraints + blocking_candidates + next_steps + final_plan, limit=6)
    existing_postop = _string_list(final.get("postop_watchpoints"))
    specific_plan = _string_list(final.get("specific_surgical_plan"))
    watchpoint_candidates = existing_postop + specific_plan + final_plan + next_steps + priority_actions
    postop_watchpoints = _select_top_watchpoints(watchpoint_candidates, limit=6)

    final["current_status_level"] = status_level
    final["surgery_channel_open"] = surgery_channel_open
    final["core_constraints"] = core_constraints
    final["current_assessment"] = _derive_current_assessment(final, has_blocking_issues=bool(deduped_blocking))
    final["readiness_status"] = readiness_status
    final["priority_actions"] = priority_actions
    final["critical_risks"] = _dedupe_keep_order(key_risks, limit=4)
    final["postop_watchpoints"] = postop_watchpoints
    final["specialty_briefs"] = specialty_briefs

    workflow["head_doctor_recommendation"] = final
    return workflow


def _to_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(item) for item in value]
    return str(value)


def _upsert_session(session_id: str | None, user_message: str) -> tuple[str, str]:
    now = time.time()
    sid = str(session_id or "").strip() or uuid.uuid4().hex
    with SESSIONS_LOCK:
        session = SESSIONS.get(sid)
        if session is None:
            session = {
                "session_id": sid,
                "created_at": now,
                "updated_at": now,
                "base_case": "",
                "supplements": [],
                "turns": [],
                "last_workflow": None,
                "last_reply": "",
            }
            SESSIONS[sid] = session

        if not session.get("base_case"):
            session["base_case"] = user_message
            mode = "initial_case"
        else:
            session["supplements"].append(user_message)
            mode = "supplement"

        session["turns"].append(
            {
                "role": "user",
                "content": user_message,
                "timestamp": now,
            }
        )
        session["updated_at"] = now
        context_text = _build_case_context_text(session)
    return sid, context_text


def _build_case_context_text(session: dict[str, Any]) -> str:
    base_case = str(session.get("base_case") or "").strip()
    supplements = [str(item).strip() for item in session.get("supplements", []) if str(item).strip()]
    last_workflow = session.get("last_workflow") if isinstance(session.get("last_workflow"), dict) else {}
    last_final = last_workflow.get("head_doctor_recommendation", {}) if isinstance(last_workflow, dict) else {}

    unresolved = _string_list(last_final.get("core_constraints")) + _string_list(last_final.get("unresolved_issues"))
    unresolved = _dedupe_keep_order(unresolved, limit=6)
    last_plan = _string_list(last_final.get("specific_surgical_plan") or last_final.get("final_plan"))

    parts: list[str] = []
    if base_case:
        parts.append(f"【原始病历】\n{base_case}")
    if supplements:
        supplement_lines = "\n".join([f"{idx + 1}. {text}" for idx, text in enumerate(supplements)])
        parts.append(f"【后续补充信息】\n{supplement_lines}")
    if unresolved:
        parts.append("【上一轮仍不确定或待补充要点】\n" + "\n".join([f"- {item}" for item in unresolved]))
    if last_plan:
        parts.append("【上一轮方案草案（供更新参考）】\n" + "\n".join([f"- {item}" for item in last_plan[:8]]))
    parts.append(
        "【任务】请基于上述完整信息重新评估，若补充信息已解决关键不确定项，请直接给出可执行的具体手术方案（术前/术中/术后步骤）。"
    )
    return "\n\n".join(parts)


def _update_session_after_workflow(session_id: str, user_message: str, workflow: dict[str, Any], reply: str) -> None:
    now = time.time()
    with SESSIONS_LOCK:
        session = SESSIONS.get(session_id)
        if not session:
            return
        session["last_workflow"] = _to_json_safe(workflow)
        session["last_reply"] = reply
        session["updated_at"] = now
        session["turns"].append(
            {
                "role": "assistant",
                "content": reply,
                "timestamp": now,
                "input_message": user_message,
            }
        )


def _build_plan_agent_workflow(
    plan_result: dict[str, Any],
    patient_input: str,
) -> dict[str, Any]:
    status_level = str(plan_result.get("status_level") or "OPTIMIZE").strip().upper()
    surgery_ready = str(plan_result.get("surgery_ready") or "no").strip().lower()
    plan_rows = plan_result.get("final_surgical_plan", [])
    plan_steps: list[str] = []
    if isinstance(plan_rows, list):
        for row in plan_rows:
            if not isinstance(row, dict):
                continue
            phase = str(row.get("phase") or "preop").strip()
            action = str(row.get("action") or "").strip()
            owner = str(row.get("owner") or "").strip()
            trigger = str(row.get("trigger") or "").strip()
            if not action:
                continue
            suffix_parts = []
            if owner:
                suffix_parts.append(f"owner:{owner}")
            if trigger:
                suffix_parts.append(f"trigger:{trigger}")
            suffix = f" ({'; '.join(suffix_parts)})" if suffix_parts else ""
            step_text = f"[{phase}] {action}{suffix}"
            plan_steps.append(step_text)

    pending_blockers = _string_list(plan_result.get("pending_blockers"))
    risk_points = _string_list(plan_result.get("risk_control_points"))
    summary = (
        f"Final surgical plan updated from supplements. "
        f"Status level: {status_level}. "
        f"Surgery channel: {'open' if surgery_ready == 'yes' else 'closed'}."
    )
    return {
        "workflow_mode": "plan_agent_followup",
        "patient_input": patient_input,
        "final_surgical_plan": plan_result,
        "head_doctor_recommendation": {
            "display_mode": "full_plan",
            "current_status_level": status_level,
            "surgery_channel_open": surgery_ready if surgery_ready in {"yes", "no"} else "no",
            "core_constraints": pending_blockers,
            "current_assessment": summary,
            "readiness_status": "Final surgical plan has been refreshed using supplemental info.",
            "specific_surgical_plan": plan_steps,
            "final_plan": plan_steps,
            "next_steps": pending_blockers,
            "priority_actions": pending_blockers,
            "critical_risks": risk_points,
            "key_risks": risk_points,
            "postop_watchpoints": _select_top_watchpoints(plan_steps, limit=6),
            "medical_record_plan": str(plan_result.get("notes") or "").strip(),
            "specialty_consensus": [],
            "module_assessments": [],
        },
    }


def _apply_summary_display_mode(workflow: dict[str, Any]) -> dict[str, Any]:
    final = workflow.get("head_doctor_recommendation")
    if not isinstance(final, dict):
        return workflow

    core_constraints = _string_list(final.get("core_constraints"))
    next_steps = _string_list(final.get("next_steps"))
    unresolved = _string_list(final.get("unresolved_issues"))
    if not next_steps and unresolved:
        next_steps = unresolved

    final["display_mode"] = "summary_only"
    final["specific_surgical_plan"] = []
    final["final_plan"] = []
    final["postop_watchpoints"] = []
    final["priority_actions"] = _dedupe_keep_order(core_constraints + next_steps, limit=6)
    workflow["head_doctor_recommendation"] = final
    workflow["workflow_mode"] = "workflow"
    return workflow


def _append_event(job_id: str, event: dict[str, Any]) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        event["index"] = len(job["events"])
        job["events"].append(event)
        job["updated_at"] = time.time()


def _attach_step_logging(obj: Any, method_name: str, label: str, job_id: str) -> None:
    method = getattr(obj, method_name, None)
    if not callable(method):
        return
    flag = f"_mdt_chat_logging_attached_{method_name}"
    if getattr(obj, flag, False):
        return

    def logged_method(*args: Any, **kwargs: Any) -> Any:
        dynamic_label = label
        if method_name == "_review_with_model":
            model_key = kwargs.get("model_key")
            if model_key is None and args:
                model_key = args[0]
            if model_key:
                dynamic_label = f"{label}/{model_key}"

        started = time.time()
        _append_event(
            job_id,
            {
                "type": "start",
                "label": dynamic_label,
                "method": method_name,
                "timestamp": started,
            },
        )
        result = method(*args, **kwargs)
        duration = time.time() - started
        _append_event(
            job_id,
            {
                "type": "done",
                "label": dynamic_label,
                "method": method_name,
                "timestamp": time.time(),
                "duration_seconds": round(duration, 3),
                "output": _to_json_safe(result),
            },
        )
        return result

    setattr(obj, method_name, logged_method)
    setattr(obj, flag, True)


def _attach_progress_logging(agent: HeadDoctorAgent, job_id: str) -> None:
    _attach_step_logging(agent.mdt_call_agent, "_plan_initial_with_api", "mdt_call/initial_triage_planning", job_id)
    _attach_step_logging(agent.mdt_call_agent, "_plan_with_api", "mdt_call/follow_up_planning", job_id)
    _attach_step_logging(agent, "_integrate_with_api", "head_doctor/model_panel_review", job_id)
    _attach_step_logging(agent, "_review_with_model", "head_doctor/model_review", job_id)
    for specialty_id, adapter in agent.specialists.items():
        _attach_step_logging(adapter.mapper, "call_patient_structuring_api", f"{specialty_id}/structuring", job_id)
        _attach_step_logging(adapter.mapper, "call_medical_entity_linking_api", f"{specialty_id}/entity_linking", job_id)
        _attach_step_logging(adapter.retriever, "retrieve_with_api", f"{specialty_id}/retrieval", job_id)
        _attach_step_logging(adapter.decision_agent, "decide_with_api", f"{specialty_id}/decision", job_id)
        _attach_step_logging(adapter, "clarify_case", f"{specialty_id}/clarification", job_id)


def _run_workflow_job(job_id: str, payload: dict[str, Any]) -> None:
    try:
        with JOBS_LOCK:
            if job_id not in JOBS:
                return
            JOBS[job_id]["status"] = "running"

        message = str(payload.get("message") or "").strip()
        session_id = str(payload.get("session_id") or "").strip()
        patient_context = str(payload.get("_patient_context") or message).strip()
        use_api_for_structuring = bool(payload.get("use_api_for_structuring", True))
        use_api_for_entity_linking = bool(payload.get("use_api_for_entity_linking", True))
        use_api_for_specialists = bool(payload.get("use_api_for_specialists", True))
        use_mdt_for_uncertainty = bool(payload.get("use_mdt_for_uncertainty", True))
        use_api_for_mdt = bool(payload.get("use_api_for_mdt", True))
        use_api_for_clarification = bool(payload.get("use_api_for_clarification", True))
        use_api_for_final = bool(payload.get("use_api_for_final", True))
        run_mode = str(payload.get("_run_mode") or "workflow").strip()
        use_api_for_plan_agent = bool(payload.get("use_api_for_plan_agent", True))

        _append_event(
            job_id,
            {
                "type": "system",
                "label": "workflow/start",
                "message": "MDT workflow started.",
                "timestamp": time.time(),
            },
        )

        if run_mode == "plan_agent_followup" and session_id:
            _append_event(
                job_id,
                {
                    "type": "system",
                    "label": "plan_agent/start",
                    "message": "Entering final surgical planning mode with new supplements.",
                    "timestamp": time.time(),
                },
            )
            with SESSIONS_LOCK:
                session = SESSIONS.get(session_id)
                base_workflow = dict(session.get("last_workflow") or {}) if session else {}
                plan_agent_session_id = str(session.get("plan_agent_session_id") or "").strip() if session else ""

            with PLAN_AGENT_LOCK:
                if not plan_agent_session_id:
                    plan_agent_session_id = PLAN_AGENT.create_session(base_workflow)
                    with SESSIONS_LOCK:
                        session = SESSIONS.get(session_id)
                        if session is not None:
                            session["plan_agent_session_id"] = plan_agent_session_id

                plan_result = PLAN_AGENT.update_and_plan(
                    session_id=plan_agent_session_id,
                    supplemental_info=message,
                    use_api=use_api_for_plan_agent,
                )

            workflow = _build_plan_agent_workflow(plan_result=plan_result, patient_input=message)
            _append_event(
                job_id,
                {
                    "type": "done",
                    "label": "plan_agent/done",
                    "message": "Final surgical plan updated.",
                    "timestamp": time.time(),
                    "output": _to_json_safe(plan_result),
                },
            )
        else:
            agent = HeadDoctorAgent()
            _attach_progress_logging(agent, job_id)
            workflow = agent.evaluate_case(
                patient_input=patient_context,
                use_api_for_structuring=use_api_for_structuring,
                use_api_for_entity_linking=use_api_for_entity_linking,
                use_api_for_specialists=use_api_for_specialists,
                use_mdt_for_uncertainty=use_mdt_for_uncertainty,
                use_api_for_mdt=use_api_for_mdt,
                use_api_for_clarification=use_api_for_clarification,
                use_api_for_final=use_api_for_final,
            )
            if session_id:
                with PLAN_AGENT_LOCK:
                    plan_agent_session_id = PLAN_AGENT.create_session(workflow)
                with SESSIONS_LOCK:
                    session = SESSIONS.get(session_id)
                    if session is not None:
                        session["plan_agent_session_id"] = plan_agent_session_id

        workflow = _enrich_workflow_for_display(workflow)
        if run_mode == "workflow":
            workflow = _apply_summary_display_mode(workflow)
        else:
            final = workflow.get("head_doctor_recommendation")
            if isinstance(final, dict) and not str(final.get("display_mode") or "").strip():
                final["display_mode"] = "full_plan"
                workflow["head_doctor_recommendation"] = final
        reply = build_answer_text(workflow)
        if session_id:
            _update_session_after_workflow(
                session_id=session_id,
                user_message=message,
                workflow=workflow,
                reply=reply,
            )

        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if not job:
                return
            job["status"] = "done"
            job["reply"] = reply
            job["workflow"] = _to_json_safe(workflow)
            job["run_mode"] = run_mode
            if session_id:
                job["session_id"] = session_id
            job["updated_at"] = time.time()

        _append_event(
            job_id,
            {
                "type": "system",
                "label": "workflow/done",
                "message": "MDT workflow finished.",
                "timestamp": time.time(),
            },
        )
    except Exception as exc:
        traceback.print_exc()
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if not job:
                return
            job["status"] = "error"
            job["error"] = f"workflow failed: {exc}"
            job["updated_at"] = time.time()
        _append_event(
            job_id,
            {
                "type": "error",
                "label": "workflow/error",
                "message": str(exc),
                "timestamp": time.time(),
            },
        )


class MDTChatHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str, status: int = HTTPStatus.OK) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/chat"):
            if not HTML_PATH.exists():
                self._send_html("mdt_chat_page.html not found", HTTPStatus.NOT_FOUND)
                return
            self._send_html(HTML_PATH.read_text(encoding="utf-8"))
            return
        if parsed.path == "/api/health":
            self._send_json({"status": "ok"})
            return
        if parsed.path == "/api/job":
            qs = parse_qs(parsed.query)
            job_id = (qs.get("job_id") or [""])[0].strip()
            if not job_id:
                self._send_json({"error": "job_id is required"}, HTTPStatus.BAD_REQUEST)
                return
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if not job:
                    self._send_json({"error": "job not found"}, HTTPStatus.NOT_FOUND)
                    return
                self._send_json(_to_json_safe(job))
            return
        self._send_json({"error": "Not Found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/chat":
            self._send_json({"error": "Not Found"}, HTTPStatus.NOT_FOUND)
            return

        try:
            content_len = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_len) if content_len > 0 else b"{}"
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            self._send_json({"error": "Invalid JSON body"}, HTTPStatus.BAD_REQUEST)
            return

        message = str(payload.get("message") or "").strip()
        if not message:
            self._send_json({"error": "message is required"}, HTTPStatus.BAD_REQUEST)
            return
        requested_session_id = str(payload.get("session_id") or "").strip()
        session_id, patient_context = _upsert_session(requested_session_id, message)
        use_plan_agent_followup = bool(payload.get("use_plan_agent_followup", True))
        force_full_workflow = bool(payload.get("force_full_workflow", False))
        use_api_for_plan_agent = bool(payload.get("use_api_for_plan_agent", True))
        with SESSIONS_LOCK:
            session = SESSIONS.get(session_id)
            has_last_workflow = bool(session and isinstance(session.get("last_workflow"), dict))
        run_mode = "plan_agent_followup" if (use_plan_agent_followup and has_last_workflow and not force_full_workflow) else "workflow"
        payload["session_id"] = session_id
        payload["_patient_context"] = patient_context
        payload["_run_mode"] = run_mode
        payload["use_api_for_plan_agent"] = use_api_for_plan_agent

        job_id = uuid.uuid4().hex
        now = time.time()
        with JOBS_LOCK:
            JOBS[job_id] = {
                "job_id": job_id,
                "session_id": session_id,
                "run_mode": run_mode,
                "status": "queued",
                "created_at": now,
                "updated_at": now,
                "events": [],
                "reply": None,
                "workflow": None,
                "error": None,
            }
        thread = threading.Thread(target=_run_workflow_job, args=(job_id, payload), daemon=True)
        thread.start()
        self._send_json({"job_id": job_id, "session_id": session_id, "run_mode": run_mode, "status": "queued"})


def main() -> None:
    args = build_parser().parse_args()
    server = ThreadingHTTPServer((args.host, args.port), MDTChatHandler)
    url = f"http://{args.host}:{args.port}/chat"
    print(f"[INFO] MDT chat server running at {url}", flush=True)
    if not args.no_open:
        try:
            webbrowser.open(url, new=2)
            print("[INFO] Browser opened.", flush=True)
        except Exception as exc:
            print(f"[WARN] Failed to open browser automatically: {exc}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
