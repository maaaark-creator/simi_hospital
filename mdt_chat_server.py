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


HTML_PATH = PROJECT_ROOT / "mdt_chat_page_v2.html"
JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
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

    lines: list[str] = ["MDT总体结果", ""]
    if current_assessment:
        lines.append("当前判断：")
        lines.append(current_assessment)
        lines.append("")
    if readiness_status:
        lines.append("当前状态：")
        lines.append(readiness_status)
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


def _looks_postop_watchpoint(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in ("术后", "复查", "监测", "观察", "icu", "dvt", "并发症", "随访"))


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
        return "当前已形成初步 MDT 方向，但仍存在关键阻塞点，未补齐前不宜直接放行最终路径。"
    return "当前已形成初步 MDT 共识，可进入下一阶段处理。"


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

    priority_actions = _dedupe_keep_order(blocking_candidates + next_steps + final_plan, limit=6)

    postop_watchpoints = [item for item in final_plan + next_steps if _looks_postop_watchpoint(item)]
    readiness_status = str(final.get("readiness_status") or "").strip()
    if not readiness_status:
        readiness_status = (
            "当前可进入快速术前优化流程，但存在明确阻塞点，未补齐前不建议直接放行最终路径。"
            if blocking_candidates
            else "当前未见新的主要阻塞点，可按 MDT 共识推进下一阶段处理。"
        )

    deduped_blocking = _dedupe_keep_order(blocking_candidates, limit=5)
    final["current_assessment"] = _derive_current_assessment(final, has_blocking_issues=bool(deduped_blocking))
    final["readiness_status"] = readiness_status
    final["priority_actions"] = priority_actions
    final["critical_risks"] = _dedupe_keep_order(key_risks, limit=4)
    final["postop_watchpoints"] = _dedupe_keep_order(postop_watchpoints, limit=5)
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
        use_api_for_structuring = bool(payload.get("use_api_for_structuring", True))
        use_api_for_entity_linking = bool(payload.get("use_api_for_entity_linking", True))
        use_api_for_specialists = bool(payload.get("use_api_for_specialists", True))
        use_mdt_for_uncertainty = bool(payload.get("use_mdt_for_uncertainty", True))
        use_api_for_mdt = bool(payload.get("use_api_for_mdt", True))
        use_api_for_clarification = bool(payload.get("use_api_for_clarification", True))
        use_api_for_final = bool(payload.get("use_api_for_final", True))

        agent = HeadDoctorAgent()
        _attach_progress_logging(agent, job_id)

        _append_event(job_id, {"type": "system", "label": "workflow/start", "message": "MDT工作流开始执行。", "timestamp": time.time()})
        workflow = agent.evaluate_case(
            patient_input=message,
            use_api_for_structuring=use_api_for_structuring,
            use_api_for_entity_linking=use_api_for_entity_linking,
            use_api_for_specialists=use_api_for_specialists,
            use_mdt_for_uncertainty=use_mdt_for_uncertainty,
            use_api_for_mdt=use_api_for_mdt,
            use_api_for_clarification=use_api_for_clarification,
            use_api_for_final=use_api_for_final,
        )
        workflow = _enrich_workflow_for_display(workflow)
        reply = build_answer_text(workflow)

        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if not job:
                return
            job["status"] = "done"
            job["reply"] = reply
            job["workflow"] = _to_json_safe(workflow)
            job["updated_at"] = time.time()
        _append_event(job_id, {"type": "system", "label": "workflow/done", "message": "MDT工作流执行完成。", "timestamp": time.time()})
    except Exception as exc:
        traceback.print_exc()
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if not job:
                return
            job["status"] = "error"
            job["error"] = f"workflow failed: {exc}"
            job["updated_at"] = time.time()
        _append_event(job_id, {"type": "error", "label": "workflow/error", "message": str(exc), "timestamp": time.time()})


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

        job_id = uuid.uuid4().hex
        now = time.time()
        with JOBS_LOCK:
            JOBS[job_id] = {
                "job_id": job_id,
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
        self._send_json({"job_id": job_id, "status": "queued"})


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
