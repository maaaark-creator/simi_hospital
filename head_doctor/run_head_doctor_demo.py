from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import sys

from head_doctor_agent import HeadDoctorAgent


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multi_agent_config import load_project_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the head doctor orchestration demo.")
    parser.add_argument("--mode", choices=["local", "api"], default="local")
    parser.add_argument("--specialist-mode", choices=["heuristic", "api"], default="heuristic")
    parser.add_argument("--mdt-mode", choices=["heuristic", "api"], default="heuristic")
    parser.add_argument("--final-mode", choices=["heuristic", "api"], default="heuristic")
    parser.add_argument("--patient-file", type=Path)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "multi_agent_config.json")
    return parser


def get_local_patient() -> dict[str, Any]:
    return {
        "年龄": 63,
        "性别": "男",
        "主诉": "发热伴黄疸2天，右上腹痛",
        "症状": ["发热", "黄疸", "右上腹痛", "寒战"],
        "诊断提示": "急性胆管炎，胆总管结石可能",
        "基础疾病": ["2型糖尿病", "高血压"],
        "生命体征": {"BP": "88/54 mmHg", "HR": 118, "SpO2": "96%", "T": "39.1C"},
        "化验": {"TBil": "96 umol/L", "WBC": "18.2e9/L", "乳酸": "3.5 mmol/L"},
        "影像摘要": "MRCP提示胆总管下段结石并胆道扩张",
        "感染情况": "高热、寒战，考虑胆源性感染",
        "急诊": "急诊",
        "拟行手术": "ERCP 或胆道减压",
        "手术部位": "胆道系统",
        "ASA分级": "III",
        "禁食情况": "未明确",
        "气道评估": "暂缺",
    }


def get_api_patient_text() -> str:
    return (
        "患者男，63岁。"
        "发热伴黄疸2天，右上腹痛并寒战。"
        "既往有2型糖尿病和高血压。"
        "当前血压88/54 mmHg，心率118次/分，体温39.1C。"
        "实验室提示TBil 96 umol/L，WBC 18.2e9/L，乳酸3.5 mmol/L。"
        "MRCP提示胆总管下段结石并胆道扩张，考虑急性胆管炎。"
        "拟急诊评估ERCP或其他胆道减压，禁食情况和气道评估暂不清楚。"
    )


def load_input_file(path: Path) -> dict[str, Any] | str:
    content = path.read_text(encoding="utf-8").strip()
    if path.suffix.lower() == ".json":
        return json.loads(content)
    return content


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_project_config(args.config)
    runtime_config = config.get("runtime", {})
    patient_case_config = config.get("patient_case", {})
    output_config = config.get("output", {})

    mode = runtime_config.get("mode", args.mode)
    specialist_mode = runtime_config.get("specialist_mode", args.specialist_mode)
    mdt_mode = runtime_config.get("mdt_mode", args.mdt_mode)
    final_mode = runtime_config.get("final_mode", args.final_mode)

    configured_patient_file = patient_case_config.get("patient_file")
    patient_file = args.patient_file or (Path(configured_patient_file) if configured_patient_file else None)

    if patient_file:
        patient_input = load_input_file(patient_file)
    elif mode == "api":
        patient_input = get_api_patient_text()
    else:
        patient_input = get_local_patient()

    agent = HeadDoctorAgent()
    result = agent.evaluate_case(
        patient_input=patient_input,
        use_api_for_structuring=mode == "api",
        use_api_for_entity_linking=mode == "api",
        use_api_for_specialists=specialist_mode == "api",
        use_mdt_for_uncertainty=bool(runtime_config.get("use_mdt_for_uncertainty", True)),
        use_api_for_mdt=mdt_mode == "api",
        use_api_for_clarification=bool(runtime_config.get("use_api_for_clarification", mdt_mode == "api")),
        use_api_for_final=final_mode == "api",
    )
    print(
        json.dumps(
            result,
            ensure_ascii=bool(output_config.get("ensure_ascii", False)),
            indent=int(output_config.get("indent", 2)),
        )
    )


if __name__ == "__main__":
    main()
