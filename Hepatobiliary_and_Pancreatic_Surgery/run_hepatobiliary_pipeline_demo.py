from __future__ import annotations

import argparse
import json

from hbp_decision_agent import HepatobiliaryDecisionAgent
from hbp_kb_retriever import KnowledgeRetriever
from patient_mapper_agent import HepatobiliaryKnowledgeBase, PatientProfileMapperAgent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the hepatobiliary mapper -> retriever -> decision pipeline demo."
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
        "年龄": 63,
        "性别": "男",
        "主诉": "发热伴黄疸 2 天，右上腹痛",
        "症状": ["发热", "黄疸", "右上腹痛", "寒战"],
        "诊断提示": "急性胆管炎，胆总管结石可能",
        "基础疾病": ["2 型糖尿病", "高血压"],
        "生命体征": {"BP": "88/54 mmHg", "HR": 118, "SpO2": "96%", "T": "39.1C"},
        "化验": {"TBil": "96 umol/L", "WBC": "18.2e9/L", "乳酸": "3.5 mmol/L"},
        "影像摘要": "MRCP 提示胆总管下段结石并胆道扩张",
        "感染情况": "高热、寒战，考虑胆源性感染",
        "急诊": "急诊",
    }


def get_api_patient_text() -> str:
    return (
        "患者男，63 岁。"
        "发热伴黄疸 2 天，右上腹痛并寒战。"
        "既往有 2 型糖尿病和高血压。"
        "入院时血压 88/54 mmHg，心率 118 次/分，体温 39.1C。"
        "实验室提示 TBil 96 umol/L，WBC 18.2e9/L，乳酸 3.5 mmol/L。"
        "MRCP 提示胆总管下段结石并胆道扩张，考虑急性胆管炎。"
        "拟急诊评估 ERCP 或其他胆道减压。"
    )


def run_local_demo(
    mapper: PatientProfileMapperAgent,
    retriever: KnowledgeRetriever,
    decision_agent: HepatobiliaryDecisionAgent,
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
    decision_agent: HepatobiliaryDecisionAgent,
    decision_mode: str,
) -> None:
    raw_patient_text = get_api_patient_text()
    structured = mapper.call_patient_structuring_api(raw_patient_text)
    normalized = mapper.normalize_patient_input(structured)
    entity_linking = mapper.call_medical_entity_linking_api(normalized)
    retrieval_result = retriever.retrieve_with_api(normalized, entity_linking)
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

    kb = HepatobiliaryKnowledgeBase()
    mapper = PatientProfileMapperAgent(kb)
    retriever = KnowledgeRetriever(kb)
    decision_agent = HepatobiliaryDecisionAgent()

    if args.mode == "local":
        run_local_demo(mapper, retriever, decision_agent, args.decision_mode)
        return

    run_api_demo(mapper, retriever, decision_agent, args.decision_mode)


if __name__ == "__main__":
    main()
