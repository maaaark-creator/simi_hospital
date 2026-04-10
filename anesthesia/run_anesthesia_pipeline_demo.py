from __future__ import annotations

import argparse
import json

from anesthesia_decision_agent import AnesthesiaDecisionAgent
from kb_retriever import KnowledgeRetriever
from patient_mapper_agent import AnesthesiaKnowledgeBase, PatientProfileMapperAgent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the mapper -> retriever -> decision pipeline demo."
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
        help="heuristic: local decision logic; api: call the model for final decision output.",
    )
    return parser


def get_local_patient() -> dict:
    return {
        "年龄": 67,
        "性别": "男",
        "体重": 72,
        "拟行手术": "下肢骨折内固定",
        "ASA分级": "III",
        "过敏史": "无",
        "基础疾病": ["高血压", "糖尿病"],
        "禁食情况": "已禁食 8 小时",
        "气道评估": "张口尚可，暂未提示困难气道",
        "急诊": "择期",
    }


def get_api_patient_text() -> str:
    return (
        "患者，男，67岁，72kg。"
        "拟行下肢骨折内固定术。"
        "既往有高血压、糖尿病。"
        "ASA III级。"
        "已禁食8小时。"
        "气道评估暂未提示困难气道。"
        "过敏史无特殊。"
    )


def run_local_demo(
    mapper: PatientProfileMapperAgent,
    retriever: KnowledgeRetriever,
    decision_agent: AnesthesiaDecisionAgent,
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
    decision_agent: AnesthesiaDecisionAgent,
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

    kb = AnesthesiaKnowledgeBase()
    mapper = PatientProfileMapperAgent(kb)
    retriever = KnowledgeRetriever(kb)
    decision_agent = AnesthesiaDecisionAgent()

    if args.mode == "local":
        run_local_demo(mapper, retriever, decision_agent, args.decision_mode)
        return

    run_api_demo(mapper, retriever, decision_agent, args.decision_mode)


if __name__ == "__main__":
    main()
