from __future__ import annotations

import argparse
import json

from cardiology_decision_agent import CardiologyDecisionAgent
from kb_retriever import KnowledgeRetriever
from patient_mapper_agent import CardiologyKnowledgeBase, PatientProfileMapperAgent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the cardiology mapper -> retriever -> decision pipeline demo."
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
        "年龄": 64,
        "性别": "男",
        "主诉": "胸痛 2 小时，伴出汗",
        "症状": ["胸痛", "出汗"],
        "诊断提示": "急性冠脉综合征待排",
        "基础疾病": ["高血压", "糖尿病"],
        "用药史": ["阿司匹林"],
        "生命体征": {"BP": "92/58 mmHg", "HR": 104, "SpO2": "95%"},
        "检验": {"肌钙蛋白": "升高"},
        "心电图": "下壁导联 ST 段抬高",
        "急诊": "急诊",
    }


def get_api_patient_text() -> str:
    return (
        "患者男，64岁。"
        "胸痛2小时，伴大汗，急诊就诊。"
        "既往高血压、糖尿病。"
        "目前血压 92/58 mmHg，心率 104 次/分。"
        "心电图提示下壁导联 ST 段抬高，肌钙蛋白升高。"
        "考虑急性冠脉综合征，需尽快评估再灌注。"
    )


def run_local_demo(
    mapper: PatientProfileMapperAgent,
    retriever: KnowledgeRetriever,
    decision_agent: CardiologyDecisionAgent,
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
    decision_agent: CardiologyDecisionAgent,
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

    kb = CardiologyKnowledgeBase()
    mapper = PatientProfileMapperAgent(kb)
    retriever = KnowledgeRetriever(kb)
    decision_agent = CardiologyDecisionAgent()

    if args.mode == "local":
        run_local_demo(mapper, retriever, decision_agent, args.decision_mode)
        return

    run_api_demo(mapper, retriever, decision_agent, args.decision_mode)


if __name__ == "__main__":
    main()
