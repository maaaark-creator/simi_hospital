from __future__ import annotations

import argparse
import json

from mdt_call_agent import MDTCallAgent
from mdt_kb_retriever import MDTKnowledgeRetriever
from mdt_patient_mapper_agent import MDTKnowledgeBase, MDTPatientProfileMapperAgent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the MDT mapper -> retriever -> decision pipeline demo."
    )
    parser.add_argument(
        "--mode",
        choices=["local", "api"],
        default="local",
        help="local: use local patient dict; api: call the external model API for text structuring.",
    )
    parser.add_argument(
        "--decision-mode",
        choices=["heuristic", "api"],
        default="heuristic",
        help="heuristic: local decision logic; api: call the model for final decision output.",
    )
    return parser


def get_local_patient() -> dict:
    return {
        "年龄": 74,
        "性别": "女",
        "体重": 58,
        "身高": 160,
        "主诉": "跌倒后左髋部疼痛活动受限 6 小时，近 2 周反复胸闷心悸",
        "现病史": (
            "患者跌倒后左髋部疼痛明显，不能负重。近2周活动后胸闷、心悸，偶有头晕，"
            "1个月前曾出现短暂右上肢无力约10分钟，自行缓解。"
        ),
        "初步诊断": ["左股骨颈骨折", "冠心病待排", "短暂性脑缺血发作待排"],
        "拟行手术": "左髋部骨折内固定或关节置换",
        "手术部位": "左髋部",
        "ASA分级": "III",
        "症状": ["髋部疼痛", "胸闷", "心悸", "头晕", "短暂肢体无力"],
        "基础疾病": ["高血压", "2型糖尿病", "房颤"],
        "既往病史": ["5年前脑梗死", "冠脉粥样硬化病史"],
        "既往手术史": ["阑尾切除术"],
        "家族史": ["母亲脑卒中", "外甥疑似恶性高热病史"],
        "长期用药": ["阿司匹林", "美托洛尔", "二甲双胍"],
        "过敏史": ["青霉素皮试阳性"],
        "活动耐量": "平地步行约100米后胸闷",
        "吸烟史": "无",
        "饮酒史": "偶尔",
        "禁食情况": "已禁食 6 小时",
        "气道评估": "张口尚可，颈活动可",
        "急诊": "急诊",
        "生命体征": {"BP": "168/92 mmHg", "HR": "112 bpm 不齐", "SpO2": "96%"},
        "实验室检查": {"Hb": "101 g/L", "Cr": "118 umol/L", "Glu": "11.2 mmol/L"},
        "影像学检查": {
            "Xray": "左股骨颈骨折",
            "ECG": "房颤伴快速心室率",
            "CT_head": "陈旧性脑梗死灶",
        },
    }


def get_api_patient_text() -> str:
    return (
        "74岁女性，跌倒后左髋部疼痛活动受限6小时，拟行左髋部骨折内固定或关节置换。"
        "近2周反复胸闷心悸，活动100米后加重，偶有头晕。1个月前曾短暂右上肢无力10分钟后缓解。"
        "既往高血压、2型糖尿病、房颤、脑梗死、冠脉粥样硬化病史。"
        "长期服用阿司匹林、美托洛尔、二甲双胍。母亲脑卒中，外甥疑似恶性高热。"
        "生命体征BP 168/92 mmHg，HR 112次/分不齐。ECG示房颤伴快速心室率。ASA III。"
    )


def run_local_demo(
    mapper: MDTPatientProfileMapperAgent,
    retriever: MDTKnowledgeRetriever,
    decision_agent: MDTCallAgent,
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
    mapper: MDTPatientProfileMapperAgent,
    retriever: MDTKnowledgeRetriever,
    decision_agent: MDTCallAgent,
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

    kb = MDTKnowledgeBase()
    mapper = MDTPatientProfileMapperAgent(kb)
    retriever = MDTKnowledgeRetriever(kb)
    decision_agent = MDTCallAgent()

    if args.mode == "local":
        run_local_demo(mapper, retriever, decision_agent, args.decision_mode)
        return

    run_api_demo(mapper, retriever, decision_agent, args.decision_mode)


if __name__ == "__main__":
    main()