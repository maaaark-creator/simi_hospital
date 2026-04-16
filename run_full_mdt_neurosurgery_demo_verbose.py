from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from head_doctor.head_doctor_agent import HeadDoctorAgent

TRACE_EVENTS: list[dict[str, Any]] = []


def configure_console_encoding() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full MDT neurosurgery demo with progress logs."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "full_mdt_neurosurgery_demo_output.json",
        help="Path to save the final JSON output.",
    )
    return parser


def build_demo_patient_text() -> str:
    return (
        "患者男，58岁。"
        "近2周进行性头痛，伴恶心、左侧肢体无力，近3天加重并出现步态不稳。"
        "MRI提示右额顶叶约4.2cm肿瘤样占位，周围脑水肿明显，并有轻度中线左移；CT未见急性脑出血。"
        "目前考虑右额顶叶占位性病变，高级别胶质瘤可能，伴颅内压增高。"
        "拟行开颅肿瘤切除术。"
        "既往有高血压10年、2型糖尿病5年，长期服用缬沙坦和二甲双胍。"
        "生命体征：血压156/92 mmHg，心率84次/分，血氧98%，体温36.8摄氏度。"
        "化验提示 Hb 132 g/L，PLT 248e9/L，Cr 78 umol/L，Glu 9.8 mmol/L。"
        "神经系统查体：神志清，左上肢肌力4级，左下肢肌力4级。"
        "ASA III级，已禁食8小时，气道评估 Mallampati II级，张口度可。"
    )


def build_demo_patient_text_clean() -> str:
    return (
        "患者男，58岁。"
        "近2周进行性头痛，伴恶心、左侧肢体无力，近3天加重并出现步态不稳。"
        "MRI提示右额顶叶约4.2cm肿瘤样占位，周围脑水肿明显，并有轻度中线左移；CT未见急性脑出血。"
        "目前考虑右额顶叶占位性病变，高级别胶质瘤可能，伴颅内压增高。"
        "拟行开颅肿瘤切除术。"
        "既往有高血压10年、2型糖尿病5年，长期服用缬沙坦和二甲双胍。"
        "生命体征：血压156/92 mmHg，心率84次/分，血氧98%，体温36.8摄氏度。"
        "化验提示 Hb 132 g/L，PLT 248e9/L，Cr 78 umol/L，Glu 9.8 mmol/L。"
        "神经系统查体：神志清，左上肢肌力4级，左下肢肌力4级。"
        "ASA III级，已禁食8小时，气道评估 Mallampati II级，张口度可。"
    )


def _attach_step_logging(obj: Any, method_name: str, label: str) -> None:
    method = getattr(obj, method_name, None)
    if not callable(method):
        return
    flag = f"_demo_logging_attached_{method_name}"
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
        print(f"[START] {dynamic_label}", flush=True)
        TRACE_EVENTS.append(
            {
                "type": "start",
                "label": dynamic_label,
                "method": method_name,
                "timestamp": started,
            }
        )
        result = method(*args, **kwargs)
        duration = time.time() - started
        print(f"[DONE ] {dynamic_label} -> {duration:.1f}s", flush=True)
        TRACE_EVENTS.append(
            {
                "type": "done",
                "label": dynamic_label,
                "method": method_name,
                "timestamp": time.time(),
                "duration_seconds": round(duration, 3),
                "output": _to_json_safe(result),
            }
        )
        return result

    setattr(obj, method_name, logged_method)
    setattr(obj, flag, True)


def _attach_logging(client: Any, label: str) -> None:
    chat_json = getattr(client, "chat_json", None)
    if not callable(chat_json) or getattr(client, "_demo_logging_attached", False):
        return

    def logged_chat_json(
        config: Any,
        prompt: str,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        started = time.time()
        model_name = getattr(config, "name", getattr(config, "model", "unknown"))
        print(f"[START] {label} -> model={model_name}", flush=True)
        response = chat_json(config, prompt, temperature)
        print(f"[DONE ] {label} -> {time.time() - started:.1f}s", flush=True)
        return response

    client.chat_json = logged_chat_json
    client._demo_logging_attached = True


def _to_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(item) for item in value]
    return str(value)


def attach_progress_logging(agent: HeadDoctorAgent) -> None:
    _attach_step_logging(agent.mdt_call_agent, "_plan_initial_with_api", "mdt_call/initial_triage_planning")
    _attach_step_logging(agent.mdt_call_agent, "_plan_with_api", "mdt_call/follow_up_planning")
    _attach_step_logging(agent, "_integrate_with_api", "head_doctor/model_panel_review")
    _attach_step_logging(agent, "_synthesize_panel_opinions", "head_doctor/panel_consensus_merge")
    _attach_step_logging(agent, "_review_with_model", "head_doctor/model_review")

    for specialty_id, adapter in agent.specialists.items():
        _attach_step_logging(adapter.mapper, "call_patient_structuring_api", f"{specialty_id}/structuring")
        _attach_step_logging(adapter.mapper, "call_medical_entity_linking_api", f"{specialty_id}/entity_linking")
        _attach_step_logging(adapter.retriever, "retrieve_with_api", f"{specialty_id}/retrieval")
        _attach_step_logging(adapter.decision_agent, "decide_with_api", f"{specialty_id}/decision")
        _attach_step_logging(
            adapter,
            "clarify_case",
            f"{specialty_id}/clarification",
        )


def build_printable_summary(result: dict[str, Any]) -> dict[str, Any]:
    final_recommendation = result.get("head_doctor_recommendation", {})
    return {
        "case_summary": final_recommendation.get("case_summary"),
        "final_plan": final_recommendation.get("final_plan"),
        "next_steps": final_recommendation.get("next_steps"),
        "key_risks": final_recommendation.get("key_risks"),
        "unresolved_issues": final_recommendation.get("unresolved_issues"),
        "uncertainty_flow": final_recommendation.get("uncertainty_flow"),
        "thinking_log": final_recommendation.get("thinking_log"),
        "safety_boundary": final_recommendation.get("safety_boundary"),
    }


def build_workflow_sections(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "initial_triage": result.get("initial_triage"),
        "specialty_opinions": result.get("specialty_opinions"),
        "uncertainty_review": result.get("uncertainty_review"),
        "mdt_follow_up": result.get("mdt_follow_up"),
        "head_doctor_recommendation": result.get("head_doctor_recommendation"),
    }


def main() -> None:
    configure_console_encoding()
    args = build_parser().parse_args()

    patient = build_demo_patient_text_clean()
    print("[INFO ] Building HeadDoctorAgent...", flush=True)
    agent = HeadDoctorAgent()
    attach_progress_logging(agent)

    print("[INFO ] Starting full API-driven MDT workflow...", flush=True)
    try:
        started = time.time()
        result = agent.evaluate_case(
            patient_input=patient,
            use_api_for_structuring=True,
            use_api_for_entity_linking=True,
            use_api_for_specialists=True,
            use_mdt_for_uncertainty=True,
            use_api_for_mdt=True,
            use_api_for_clarification=True,
            use_api_for_final=True,
        )
        print(f"[INFO ] Workflow completed in {time.time() - started:.1f}s", flush=True)
    except Exception as exc:
        print(f"[ERROR] Workflow failed: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise

    output = {
        "demo_patient": patient,
        "workflow_result": build_printable_summary(result),
        "workflow_sections": build_workflow_sections(result),
        "workflow_raw": result,
        "trace_events": TRACE_EVENTS,
    }
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[INFO ] Saved output to {args.output}", flush=True)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
