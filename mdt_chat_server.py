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
    case_summary = str(final.get("case_summary") or "").strip()
    final_plan = final.get("final_plan") or []
    key_risks = final.get("key_risks") or []
    next_steps = final.get("next_steps") or []

    lines: list[str] = []
    if case_summary:
        lines.append(f"病例总结：{case_summary}")
    if final_plan:
        lines.append("建议方案：")
        lines.extend([f"- {str(item)}" for item in final_plan[:8]])
    if key_risks:
        lines.append("关键风险：")
        lines.extend([f"- {str(item)}" for item in key_risks[:5]])
    if next_steps:
        lines.append("下一步：")
        lines.extend([f"- {str(item)}" for item in next_steps[:6]])
    return "\n".join(lines) if lines else "已完成分析，但未返回可展示的结构化结论。"


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
