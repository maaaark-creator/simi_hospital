from __future__ import annotations

import json
from typing import Any

from patient_mapper_agent import (
    GenAIChatClient,
    MODEL_CONFIGS,
    TASK_MODEL_SELECTION,
)


class CardiologyDecisionAgent:
    def __init__(
        self,
        api_client: GenAIChatClient | None = None,
    ) -> None:
        self.api_client = api_client or GenAIChatClient()

    def decide(
        self,
        normalized_patient: dict[str, Any],
        retrieval_result: dict[str, Any],
    ) -> dict[str, Any]:
        candidate_plans = retrieval_result.get("candidate_plans", [])
        matched_rules = retrieval_result.get("matched_safety_rules", [])
        matched_medical_conditions = retrieval_result.get("matched_medical_conditions", [])
        matched_surgical_conditions = retrieval_result.get("matched_surgical_conditions", [])
        missing_information = retrieval_result.get("missing_information", [])

        primary_condition = self._choose_primary_condition(
            normalized_patient,
            matched_medical_conditions,
            matched_surgical_conditions,
        )
        primary_plan = self._choose_primary_plan(
            normalized_patient,
            candidate_plans,
            primary_condition,
            matched_rules,
        )
        backup_plans = [
            plan for plan in candidate_plans
            if primary_plan is None or plan["id"] != primary_plan["id"]
        ]

        return {
            "patient_summary": self._build_patient_summary(normalized_patient),
            "primary_condition": primary_condition,
            "primary_plan": primary_plan,
            "backup_plans": backup_plans,
            "risk_flags": self._build_risk_flags(normalized_patient, matched_rules, primary_condition),
            "need_more_info": missing_information,
            "reasoning_trace": self._build_reasoning_trace(
                normalized_patient,
                primary_condition,
                primary_plan,
                matched_rules,
            ),
        }

    def decide_with_api(
        self,
        normalized_patient: dict[str, Any],
        retrieval_result: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = (
            "You are a cardiology decision-support agent.\n"
            "Given the normalized cardiovascular patient profile and retrieval results, "
            "return JSON only with keys:\n"
            "patient_summary, primary_condition_id, primary_plan_id, backup_plan_ids, risk_flags, need_more_info, reasoning_trace.\n\n"
            "Normalized patient:\n"
            f"{json.dumps(normalized_patient, ensure_ascii=False, indent=2)}\n\n"
            "Retrieval result:\n"
            f"{json.dumps(retrieval_result, ensure_ascii=False, indent=2)}"
        )
        config = MODEL_CONFIGS[TASK_MODEL_SELECTION["plan_ranking"]]
        response = self.api_client.chat_json(config, prompt)
        return self.api_client.extract_json(response)

    def _choose_primary_condition(
        self,
        normalized_patient: dict[str, Any],
        matched_medical_conditions: list[dict[str, Any]],
        matched_surgical_conditions: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        combined_text = " ".join(
            [
                str(normalized_patient.get("chief_complaint") or ""),
                " ".join(normalized_patient.get("symptoms", [])),
                str(normalized_patient.get("diagnosis_hint") or ""),
                str(normalized_patient.get("ecg_summary") or ""),
                str(normalized_patient.get("echo_summary") or ""),
                str(normalized_patient.get("imaging_summary") or ""),
                str(normalized_patient.get("postop_context") or ""),
            ]
        ).lower()

        medical_by_id = {item["id"]: item for item in matched_medical_conditions}
        surgical_by_id = {item["id"]: item for item in matched_surgical_conditions}

        priority_ids = [
            "stemi",
            "acute_aortic_dissection",
            "pulmonary_embolism",
            "acute_decompensated_heart_failure",
            "atrial_fibrillation",
            "supraventricular_tachycardia",
        ]
        for condition_id in priority_ids:
            if condition_id in medical_by_id:
                return medical_by_id[condition_id]
            if condition_id in surgical_by_id:
                return surgical_by_id[condition_id]

        if "st 段抬高" in combined_text or "stemi" in combined_text:
            return medical_by_id.get("stemi")
        if "主动脉夹层" in combined_text or "aortic dissection" in combined_text:
            return surgical_by_id.get("acute_aortic_dissection")
        if "房颤" in combined_text or "atrial fibrillation" in combined_text:
            return medical_by_id.get("atrial_fibrillation")
        if "心衰" in combined_text or "肺水肿" in combined_text:
            return medical_by_id.get("acute_decompensated_heart_failure")

        if matched_medical_conditions:
            return matched_medical_conditions[0]
        if matched_surgical_conditions:
            return matched_surgical_conditions[0]
        return None

    def _choose_primary_plan(
        self,
        normalized_patient: dict[str, Any],
        candidate_plans: list[dict[str, Any]],
        primary_condition: dict[str, Any] | None,
        matched_rules: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not candidate_plans:
            return None

        plan_by_id = {plan["id"]: plan for plan in candidate_plans}
        condition_id = (primary_condition or {}).get("id")
        rule_ids = {rule["id"] for rule in matched_rules}
        ecg_text = str(normalized_patient.get("ecg_summary") or "").lower()
        chief_text = str(normalized_patient.get("chief_complaint") or "").lower()
        postop_text = str(normalized_patient.get("postop_context") or "").lower()

        if condition_id == "stemi" or "stemi_reperfusion_priority" in rule_ids:
            return plan_by_id.get("stemi_reperfusion_pathway") or plan_by_id.get("acs_initial_evaluation") or candidate_plans[0]
        if condition_id in {"nste_acs", "chronic_coronary_syndrome"}:
            return plan_by_id.get("acs_initial_evaluation") or candidate_plans[0]
        if condition_id in {"acute_decompensated_heart_failure", "hfrEF", "hfpEF"}:
            return plan_by_id.get("acute_hf_decongestion") or plan_by_id.get("hfrEF_gdmt_buildout") or candidate_plans[0]
        if condition_id == "atrial_fibrillation":
            return plan_by_id.get("af_rate_rhythm_anticoag") or candidate_plans[0]
        if condition_id == "supraventricular_tachycardia":
            return plan_by_id.get("svt_termination_pathway") or candidate_plans[0]
        if condition_id in {"acute_aortic_dissection", "ascending_aortic_aneurysm"}:
            return plan_by_id.get("aortic_emergency_pathway") or candidate_plans[0]
        if condition_id in {"severe_aortic_stenosis", "severe_mitral_regurgitation"}:
            return plan_by_id.get("valve_intervention_pathway") or candidate_plans[0]
        if condition_id in {"multivessel_cad_for_revascularization", "post_cardiotomy_low_output_syndrome"}:
            return plan_by_id.get("cabg_perioperative_pathway") or candidate_plans[0]

        if "st 段抬高" in ecg_text or "胸痛" in chief_text:
            return plan_by_id.get("acs_initial_evaluation") or candidate_plans[0]
        if "心外术后" in postop_text:
            return plan_by_id.get("cabg_perioperative_pathway") or candidate_plans[0]

        return candidate_plans[0]

    def _build_patient_summary(self, normalized_patient: dict[str, Any]) -> str:
        summary_parts = [
            f"年龄 {normalized_patient.get('age')}" if normalized_patient.get("age") is not None else "年龄未提供",
            f"性别 {normalized_patient.get('sex')}" if normalized_patient.get("sex") else "性别未提供",
            f"主诉 {normalized_patient.get('chief_complaint')}" if normalized_patient.get("chief_complaint") else "主诉未明确",
            f"诊断提示 {normalized_patient.get('diagnosis_hint')}" if normalized_patient.get("diagnosis_hint") else "诊断提示未明确",
        ]
        return "；".join(summary_parts)

    def _build_risk_flags(
        self,
        normalized_patient: dict[str, Any],
        matched_rules: list[dict[str, Any]],
        primary_condition: dict[str, Any] | None,
    ) -> list[str]:
        flags = [rule["name_zh"] for rule in matched_rules]
        comorbidity_text = " ".join(normalized_patient.get("comorbidities", []))

        if any(term in comorbidity_text for term in ["糖尿病", "慢性肾病", "CKD", "高龄"]):
            flags.append("存在重要基础疾病，需纳入再血管化/抗栓/容量管理决策")

        if normalized_patient.get("ecg_summary") is None:
            flags.append("缺少心电图摘要")

        if primary_condition and primary_condition["id"] in {"stemi", "acute_aortic_dissection", "cardiogenic_shock"}:
            flags.append("当前疑似高危急症，应优先处理时间敏感通道")

        return list(dict.fromkeys(flags))

    def _build_reasoning_trace(
        self,
        normalized_patient: dict[str, Any],
        primary_condition: dict[str, Any] | None,
        primary_plan: dict[str, Any] | None,
        matched_rules: list[dict[str, Any]],
    ) -> list[str]:
        trace: list[str] = []
        if primary_condition is not None:
            trace.append(f"优先考虑的核心问题为 {primary_condition['name_zh']}。")
        if primary_plan is not None:
            trace.append(f"首选路径暂定为 {primary_plan['name_zh']}。")
        if normalized_patient.get("chief_complaint"):
            trace.append(f"方案判断参考了主诉信息：{normalized_patient['chief_complaint']}。")
        if normalized_patient.get("ecg_summary"):
            trace.append(f"已纳入心电图摘要：{normalized_patient['ecg_summary']}。")
        if matched_rules:
            trace.append("已同步参考安全规则命中结果。")
        return trace
