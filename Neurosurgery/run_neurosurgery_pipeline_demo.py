from __future__ import annotations

import argparse
import json
import sys

from neurosurgery_decision_agent import NeurosurgeryDecisionAgent
from neurosurgery_kb_retriever import NeurosurgeryKnowledgeRetriever
from neurosurgery_patient_mapper_agent import (
    NeurosurgeryKnowledgeBase,
    NeurosurgeryPatientProfileMapperAgent,
)


def configure_console_encoding() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the neurosurgery mapper -> retriever -> decision pipeline demo."
    )
    parser.add_argument(
        "--mode",
        choices=["local", "api"],
        default="api",
        help="local: start from the built-in patient example; api: start from the built-in free-text case.",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=["heuristic", "api"],
        default="api",
        help="heuristic: local KB matching; api: use the model for specialist retrieval and triage.",
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
        "年龄": 29,
        "性别": "女",
        "体重": 56,
        "身高": 165,
        "主诉": "反复发作性意识模糊伴右手自动症 5 年，近半年每月 3 到 4 次",
        "现病史": (
            "患者5年前出现发作性凝视、应答差，随后伴右手摸索动作，约1分钟后自行缓解，"
            "发作后疲乏。近半年发作频率增加，曾在规律服用左乙拉西坦和拉莫三嗪后仍反复发作。"
            "偶诉近记忆下降，无明显发热、外伤史。"
        ),
        "初步诊断": ["药物难治性局灶性癫痫待评估", "左侧颞叶致痫灶可能"],
        "拟行手术": "癫痫外科术前评估，必要时 SEEG/左侧颞叶手术讨论",
        "手术部位": "左侧颞叶",
        "ASA分级": "II",
        "症状": ["反复发作", "意识模糊", "自动症", "近记忆下降"],
        "神经系统查体": ["清醒", "语言流利", "四肢肌力5级", "未见明确病理征"],
        "基础疾病": ["无明确高血压糖尿病病史"],
        "既往病史": ["儿童期热性惊厥史"],
        "既往手术史": [],
        "家族史": ["母系表亲有癫痫病史"],
        "长期用药": ["左乙拉西坦", "拉莫三嗪"],
        "过敏史": [],
        "功能状态": "日常生活自理，可正常步行，但发作影响学习和工作",
        "癫痫史": "近半年每月3到4次，规律服药仍发作",
        "头痛特点": [],
        "视觉症状": [],
        "认知症状": ["近记忆下降"],
        "外伤史": "否认近期头部外伤",
        "脑血管病史": [],
        "器械史": [],
        "生命体征": {"BP": "118/72 mmHg", "HR": "78 bpm", "SpO2": "99%"},
        "实验室检查": {"CBC": "未见明显异常", "Na": "139 mmol/L", "Cr": "66 umol/L"},
        "影像学检查": {
            "MRI": "左侧海马体积减小并信号异常，考虑海马硬化",
            "VEEG": "发作间期左颞区棘波"
        },
        "病理": {}
    }


def get_api_patient_text() -> str:
    return (
        "29岁女性，反复发作性意识模糊伴右手自动症5年，近半年每月3到4次。"
        "规律服用左乙拉西坦和拉莫三嗪后仍有发作，发作后疲乏，近记忆下降。"
        "MRI提示左侧海马硬化，视频脑电提示左颞区棘波，考虑药物难治性局灶性癫痫。"
        "拟进行癫痫外科术前评估，必要时SEEG或左侧颞叶手术讨论。"
        "神经查体未见明确定位体征，母系表亲有癫痫病史，ASA II。"
    )


def run_local_demo(
    mapper: NeurosurgeryPatientProfileMapperAgent,
    retriever: NeurosurgeryKnowledgeRetriever,
    decision_agent: NeurosurgeryDecisionAgent,
    retrieval_mode: str,
    decision_mode: str,
) -> None:
    raw_patient = get_local_patient()
    raw_patient_text = json.dumps(raw_patient, ensure_ascii=False, indent=2)
    structured = mapper.call_patient_structuring_api(raw_patient_text)
    normalized = mapper.normalize_patient_input(structured)
    entity_linking = mapper.call_medical_entity_linking_api(normalized)
    retrieval_result = (
        retriever.retrieve_with_api(normalized, entity_linking)
        if retrieval_mode == "api"
        else retriever.retrieve(normalized, entity_linking)
    )
    decision_result = (
        decision_agent.decide_with_api(normalized, retrieval_result)
        if decision_mode == "api"
        else decision_agent.decide(normalized, retrieval_result)
    )
    print(
        json.dumps(
            {
                "source_patient": raw_patient,
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


def run_api_demo(
    mapper: NeurosurgeryPatientProfileMapperAgent,
    retriever: NeurosurgeryKnowledgeRetriever,
    decision_agent: NeurosurgeryDecisionAgent,
    retrieval_mode: str,
    decision_mode: str,
) -> None:
    raw_patient_text = get_api_patient_text()
    structured = mapper.call_patient_structuring_api(raw_patient_text)
    normalized = mapper.normalize_patient_input(structured)
    entity_linking = mapper.call_medical_entity_linking_api(normalized)
    retrieval_result = (
        retriever.retrieve_with_api(normalized, entity_linking)
        if retrieval_mode == "api"
        else retriever.retrieve(normalized, entity_linking)
    )
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
    configure_console_encoding()

    parser = build_parser()
    args = parser.parse_args()

    kb = NeurosurgeryKnowledgeBase()
    mapper = NeurosurgeryPatientProfileMapperAgent(kb)
    retriever = NeurosurgeryKnowledgeRetriever(kb)
    decision_agent = NeurosurgeryDecisionAgent()

    if args.mode == "local":
        run_local_demo(
            mapper,
            retriever,
            decision_agent,
            args.retrieval_mode,
            args.decision_mode,
        )
        return

    run_api_demo(
        mapper,
        retriever,
        decision_agent,
        args.retrieval_mode,
        args.decision_mode,
    )


if __name__ == "__main__":
    main()
