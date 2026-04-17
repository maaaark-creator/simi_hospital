from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from kb_retriever import KnowledgeRetriever
from orthopaedics_decision_agent import OrthopaedicsDecisionAgent
from patient_mapper_agent import (
    OrthopaedicsKnowledgeBase,
    PatientProfileMapperAgent,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the orthopedics mapper -> retriever -> decision pipeline demo."
    )
    parser.add_argument(
        "--mode",
        choices=["local", "api"],
        default="api",
        help="local: use local patient dict; api: call the external model API for text structuring.",
    )
    parser.add_argument(
        "--decision-mode",
        choices=["heuristic", "api"],
        default="api",
        help="heuristic: use local decision logic; api: call the model for final decision output.",
    )
    return parser


def get_local_patient() -> dict:
    return {
        "年龄": 79,
        "性别": "女",
        "体重": 56,
        "身高": 158,
        "主诉": "跌倒后右髋疼痛、不能负重 6 小时",
        "疼痛部位": "右髋部",
        "侧别": "右侧",
        "受伤史": "家中跌倒",
        "受伤机制": "低能量跌倒",
        "功能受限": "无法站立和行走",
        "伴随症状": ["轻度头晕"],
        "基础疾病": ["高血压", "2型糖尿病", "房颤"],
        "抗凝药": ["利伐沙班"],
        "骨质疏松史": "有",
        "家族史": ["母亲有髋部骨折史"],
        "拟行手术": "右侧股骨近端骨折手术评估",
        "ASA分级": "III",
        "检验": {"Hb": "103 g/L", "Cr": "112 umol/L", "CRP": "5 mg/L"},
        "生命体征": {"BP": "148/84 mmHg", "HR": 88, "Temp": "36.8 C"},
        "影像": {"xray": "疑似股骨颈骨折，建议进一步明确"},
        "急诊": "急诊",
    }


def get_api_patient_text() -> str:
    return (
        "患者女，79岁，身高158cm，体重56kg。"
        "家中跌倒后出现右髋部疼痛并完全不能负重 6 小时。"
        "既往有高血压、2型糖尿病和房颤，长期服用利伐沙班。"
        "有骨质疏松史，母亲有髋部骨折史。"
        "拟行右侧股骨近端骨折手术评估，ASA III。"
        "X线提示疑似股骨颈骨折，建议进一步明确。"
    )


def run_local_demo(
    mapper: PatientProfileMapperAgent,
    retriever: KnowledgeRetriever,
    decision_agent: OrthopaedicsDecisionAgent,
    decision_mode: str,
) -> None:
    normalized = mapper.normalize_patient_input(get_local_patient())
    retrieval_result = retriever.retrieve(normalized)
    decision_result = (
        decision_agent.decide_with_api(normalized, retrieval_result)
        if decision_mode == "api"
        else decision_agent.decide(normalized, retrieval_result)
    )
    print(
        json.dumps(
            {
                "normalized_profile": normalized,
                "retrieval_result": retrieval_result,
                "decision_result": decision_result,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def run_api_demo(
    mapper: PatientProfileMapperAgent,
    retriever: KnowledgeRetriever,
    decision_agent: OrthopaedicsDecisionAgent,
    decision_mode: str,
) -> None:
    raw_patient_text = get_api_patient_text()
    structured = mapper.call_patient_structuring_api(raw_patient_text)
    normalized = mapper.normalize_patient_input(structured)
    entity_linking = mapper.call_medical_entity_linking_api(normalized)
    retrieval_result = retriever.retrieve(normalized, entity_linking)
    decision_result = (
        decision_agent.decide_with_api(normalized, retrieval_result)
        if decision_mode == "api"
        else decision_agent.decide(normalized, retrieval_result)
    )
    print(
        json.dumps(
            {
                "structured_patient": structured,
                "normalized_profile": normalized,
                "entity_linking": entity_linking,
                "retrieval_result": retrieval_result,
                "decision_result": decision_result,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    kb = OrthopaedicsKnowledgeBase()
    mapper = PatientProfileMapperAgent(kb)
    retriever = KnowledgeRetriever(kb)
    decision_agent = OrthopaedicsDecisionAgent()

    if args.mode == "local":
        run_local_demo(mapper, retriever, decision_agent, args.decision_mode)
        return

    run_api_demo(mapper, retriever, decision_agent, args.decision_mode)


if __name__ == "__main__":
    main()
