from __future__ import annotations

import json
from typing import Any

from patient_mapper_agent import (
    GenAIChatClient,
    MODEL_CONFIGS,
    TASK_MODEL_SELECTION,
)


class OrthopaedicsDecisionAgent:
    def __init__(self, api_client: GenAIChatClient | None = None) -> None:
        self.api_client = api_client or GenAIChatClient()

    def decide(
        self,
        normalized_patient: dict[str, Any],
        retrieval_result: dict[str, Any],
    ) -> dict[str, Any]:
        candidate_pathways = retrieval_result.get("candidate_pathways", [])
        suspected_conditions = retrieval_result.get("suspected_conditions", [])
        matched_rules = retrieval_result.get("matched_perioperative_rules", [])
        workups = retrieval_result.get("recommended_workups", [])
        complications = retrieval_result.get("complication_watchlist", [])
        missing_information = retrieval_result.get("missing_information", [])

        primary_condition = suspected_conditions[0] if suspected_conditions else None
        primary_plan = candidate_pathways[0] if candidate_pathways else None
        backup_plans = candidate_pathways[1:] if len(candidate_pathways) > 1 else []
        differential_diagnoses = suspected_conditions[1:4] if len(suspected_conditions) > 1 else []
        triage_level = self._infer_triage_level(primary_condition, matched_rules)

        return {
            "patient_summary": self._build_patient_summary(normalized_patient),
            "triage_level": triage_level,
            "primary_condition": primary_condition,
            "primary_plan": primary_plan,
            "backup_plans": backup_plans,
            "primary_impression": primary_condition,
            "differential_diagnoses": differential_diagnoses,
            "recommended_workup": self._flatten_workups(workups),
            "recommended_management": self._build_management(primary_condition, matched_rules),
            "perioperative_considerations": self._build_periop_considerations(
                normalized_patient,
                matched_rules,
            ),
            "potential_complications": [item["name_zh"] for item in complications],
            "risk_flags": [rule["name_zh"] for rule in matched_rules],
            "need_more_info": missing_information,
            "reasoning_trace": self._build_reasoning_trace(
                normalized_patient,
                primary_condition,
                primary_plan,
                matched_rules,
            ),
            "safety_disclaimer": "本结果仅供骨科评估与教学原型参考，不能替代专科医生面对面诊疗与影像复核。",
        }

    def decide_with_api(
        self,
        normalized_patient: dict[str, Any],
        retrieval_result: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = (
            "You are an orthopaedics decision-support agent.\n"
            "Think like a senior orthopedic surgeon with perioperative awareness.\n"
            "You must consider trauma mechanism, weight-bearing status, neurovascular risk, infection, "
            "tumor red flags, degenerative disease, fragility fracture risk, anticoagulants, family "
            "history, frailty, and likely postoperative complications.\n"
            "Do not invent unavailable facts. If information is insufficient, put it into need_more_info.\n"
            "Return JSON only with keys:\n"
            "patient_summary, triage_level, primary_condition_id, primary_plan_id, backup_plan_ids, "
            "differential_condition_ids, recommended_workup, recommended_management, perioperative_considerations, "
            "potential_complications, risk_flags, need_more_info, reasoning_trace, safety_disclaimer.\n\n"
            "Normalized patient:\n"
            f"{json.dumps(normalized_patient, ensure_ascii=False, indent=2)}\n\n"
            "Retrieval result:\n"
            f"{json.dumps(retrieval_result, ensure_ascii=False, indent=2)}"
        )
        config = MODEL_CONFIGS[TASK_MODEL_SELECTION["differential_ranking"]]
        response = self.api_client.chat_json(config, prompt)
        payload = self.api_client.extract_json(response)
        payload.setdefault("backup_plan_ids", [])
        payload.setdefault("differential_condition_ids", [])
        payload.setdefault("recommended_workup", [])
        payload.setdefault("recommended_management", [])
        payload.setdefault("perioperative_considerations", [])
        payload.setdefault("potential_complications", [])
        payload.setdefault("risk_flags", [])
        payload.setdefault("need_more_info", [])
        payload.setdefault("reasoning_trace", [])
        return payload

    def _infer_triage_level(
        self,
        primary_condition: dict[str, Any] | None,
        matched_rules: list[dict[str, Any]],
    ) -> str:
        condition_id = (primary_condition or {}).get("id")
        rule_ids = {rule["id"] for rule in matched_rules}

        if "urgent_neurovascular_review" in rule_ids or condition_id in {
            "cauda_equina_or_progressive_neurologic_compromise",
            "compartment_syndrome_or_acute_limb_ischaemia",
            "septic_arthritis_or_osteomyelitis",
        }:
            return "急诊立即升级评估"
        if condition_id in {
            "hip_fracture_or_occult_hip_fracture",
            "prosthetic_joint_or_implant_infection",
            "pathologic_fracture_or_bone_tumor",
        }:
            return "加速住院骨科评估"
        return "门诊或择期进一步评估"

    def _build_patient_summary(self, normalized_patient: dict[str, Any]) -> str:
        parts = [
            f"年龄 {normalized_patient.get('age')}" if normalized_patient.get("age") is not None else "年龄未提供",
            f"性别 {normalized_patient.get('sex')}" if normalized_patient.get("sex") else "性别未提供",
            f"主诉 {normalized_patient.get('chief_complaint')}" if normalized_patient.get("chief_complaint") else "主诉未提供",
            f"疼痛/病变部位 {normalized_patient.get('pain_site')}" if normalized_patient.get("pain_site") else "病变部位未提供",
        ]
        return "；".join(parts)

    def _flatten_workups(self, workups: list[dict[str, Any]]) -> list[str]:
        recommendations: list[str] = []
        for bundle in workups:
            for key in ("history_points", "exam_points", "labs", "imaging"):
                for item in bundle.get(key, []):
                    recommendations.append(f"{bundle['name_zh']}：{item}")
        return list(dict.fromkeys(recommendations))

    def _build_management(
        self,
        primary_condition: dict[str, Any] | None,
        matched_rules: list[dict[str, Any]],
    ) -> list[str]:
        suggestions: list[str] = []
        if primary_condition is not None:
            suggestions.extend(primary_condition.get("initial_management_principles", []))

        rule_ids = {rule["id"] for rule in matched_rules}
        if "urgent_neurovascular_review" in rule_ids:
            suggestions.append("优先完成重复神经血管评估，不要因非关键检查延误处置。")
        if "vte_risk_review" in rule_ids:
            suggestions.append("结合出血风险评估机械或药物性 VTE 预防。")
        if "fragility_fracture_secondary_prevention" in rule_ids:
            suggestions.append("纳入脆性骨折二级预防与跌倒风险干预。")
        if "preop_history_and_family_review" in rule_ids:
            suggestions.append("补齐过敏史、家族遗传病史、既往麻醉史与长期用药。")
        return list(dict.fromkeys(suggestions))

    def _build_periop_considerations(
        self,
        normalized_patient: dict[str, Any],
        matched_rules: list[dict[str, Any]],
    ) -> list[str]:
        items = [rule["name_zh"] for rule in matched_rules]

        comorbidity_text = " ".join(normalized_patient.get("comorbidities", []))
        if "糖尿病" in comorbidity_text:
            items.append("糖尿病提示伤口感染与围手术期血糖管理需求。")
        if normalized_patient.get("anticoagulants"):
            items.append("存在抗凝/抗血小板用药，需要围术期停药和出血风险复核。")
        if (normalized_patient.get("age") or 0) >= 70:
            items.append("高龄患者需关注衰弱、谵妄、肺部并发症和早期康复。")
        if normalized_patient.get("smoking_status"):
            items.append("吸烟会增加伤口并发症和骨愈合风险。")
        return list(dict.fromkeys(items))

    def _build_reasoning_trace(
        self,
        normalized_patient: dict[str, Any],
        primary_condition: dict[str, Any] | None,
        primary_plan: dict[str, Any] | None,
        matched_rules: list[dict[str, Any]],
    ) -> list[str]:
        trace: list[str] = []
        if primary_condition is not None:
            trace.append(f"首要骨科印象暂定为 {primary_condition['name_zh']}。")
        if primary_plan is not None:
            trace.append(f"首选处理路径暂定为 {primary_plan['name_zh']}。")
        if normalized_patient.get("trauma_history") or normalized_patient.get("onset_mechanism"):
            trace.append("已将受伤史/起病机制纳入判断。")
        if normalized_patient.get("family_history"):
            trace.append("已纳入家族史与潜在遗传风险考虑。")
        if normalized_patient.get("comorbidities"):
            trace.append("已纳入基础疾病和围手术期风险因素。")
        if matched_rules:
            trace.append("已结合围手术期安全规则与并发症风险进行修正。")
        return trace


OrthopedicsDecisionAgent = OrthopaedicsDecisionAgent
