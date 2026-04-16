from __future__ import annotations

import json
from typing import Any

from patient_mapper_agent import GenAIChatClient, MODEL_CONFIGS, TASK_MODEL_SELECTION


class HepatobiliaryDecisionAgent:
    def __init__(self, api_client: GenAIChatClient | None = None) -> None:
        self.api_client = api_client or GenAIChatClient()

    def decide(
        self,
        normalized_patient: dict[str, Any],
        retrieval_result: dict[str, Any],
    ) -> dict[str, Any]:
        candidate_plans = retrieval_result.get("candidate_plans", [])
        candidate_plan_scores = retrieval_result.get("candidate_plan_scores", {})
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
            candidate_plan_scores,
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
            "You are a hepatobiliary-pancreatic surgery MDT decision-support agent.\n"
            "Given the normalized patient profile and retrieval results, "
            "return JSON only with keys:\n"
            "patient_summary, primary_condition_id, primary_plan_id, backup_plan_ids, risk_flags, need_more_info, reasoning_trace.\n\n"
            "Normalized patient:\n"
            f"{json.dumps(normalized_patient, ensure_ascii=False, indent=2)}\n\n"
            "Retrieval result:\n"
            f"{json.dumps(retrieval_result, ensure_ascii=False, indent=2)}"
        )
        config = MODEL_CONFIGS[TASK_MODEL_SELECTION["plan_ranking"]]
        return self.api_client.extract_json(self.api_client.chat_json(config, prompt))

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
                str(normalized_patient.get("imaging_summary") or ""),
                str(normalized_patient.get("infection_status") or ""),
                str(normalized_patient.get("bleeding_status") or ""),
                str(normalized_patient.get("postop_context") or ""),
            ]
        ).lower()
        medical_by_id = {item["id"]: item for item in matched_medical_conditions}
        surgical_by_id = {item["id"]: item for item in matched_surgical_conditions}

        priority_ids = [
            "acute_liver_failure_or_severe_hepatocellular_injury",
            "acute_cholangitis",
            "pyogenic_liver_abscess",
            "upper_gi_or_hepatobiliary_bleeding",
            "postoperative_pancreatic_fistula",
            "postoperative_bile_leak",
            "malignant_biliary_obstruction",
            "colorectal_or_other_liver_metastases",
            "pancreatic_cystic_lesion_for_mdt",
            "hepatocellular_jaundice_pattern",
            "acute_pancreatitis",
            "obstructive_jaundice_needing_biliary_drainage",
        ]
        for condition_id in priority_ids:
            if condition_id in medical_by_id:
                return medical_by_id[condition_id]
            if condition_id in surgical_by_id:
                return surgical_by_id[condition_id]

        if "急性肝衰竭" in combined_text or "acute liver failure" in combined_text:
            return medical_by_id.get("acute_liver_failure_or_severe_hepatocellular_injury")
        if "胆管炎" in combined_text or "cholangitis" in combined_text:
            return medical_by_id.get("acute_cholangitis")
        if "无痛性黄疸" in combined_text or "胆管癌" in combined_text or "胰头癌" in combined_text:
            return medical_by_id.get("malignant_biliary_obstruction") or surgical_by_id.get("hilar_or_distal_malignant_obstruction_requiring_mdt")
        if "药物性肝损伤" in combined_text or "dili" in combined_text:
            return medical_by_id.get("hepatocellular_jaundice_pattern")

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
        candidate_plan_scores: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not candidate_plans:
            return None

        scored_candidates = sorted(
            candidate_plans,
            key=lambda plan: float((candidate_plan_scores or {}).get(plan.get("id", ""), 0.0)),
            reverse=True,
        )
        plan_by_id = {plan["id"]: plan for plan in scored_candidates}
        condition_id = (primary_condition or {}).get("id")
        rule_ids = {rule["id"] for rule in matched_rules}
        chief_text = str(normalized_patient.get("chief_complaint") or "").lower()
        diagnosis_text = str(normalized_patient.get("diagnosis_hint") or "").lower()
        procedure_text = str(normalized_patient.get("procedure_name") or "").lower()
        postop_text = str(normalized_patient.get("postop_context") or "").lower()

        if any(token in chief_text + diagnosis_text for token in ["占位待定", "待排", "排除", "体检发现", "偶发"]) and "incidental_liver_mass_mdt" in plan_by_id:
            return plan_by_id["incidental_liver_mass_mdt"]
        if condition_id == "acute_liver_failure_or_severe_hepatocellular_injury" or "acute_liver_failure_escalation_rule" in rule_ids:
            return plan_by_id.get("acute_liver_failure_escalation") or plan_by_id.get("hepatocellular_injury_workup") or scored_candidates[0]
        if condition_id == "acute_cholangitis" or "source_control_needed_for_severe_cholangitis" in rule_ids:
            return plan_by_id.get("acute_cholangitis_resuscitation") or scored_candidates[0]
        if condition_id == "pyogenic_liver_abscess" or "liver_abscess_needs_drainage_window_review" in rule_ids:
            return plan_by_id.get("liver_abscess_source_control") or scored_candidates[0]
        if condition_id == "acute_pancreatitis":
            return plan_by_id.get("acute_pancreatitis_supportive_care") or scored_candidates[0]
        if condition_id == "obstructive_jaundice_needing_biliary_drainage":
            return plan_by_id.get("obstructive_jaundice_workup") or plan_by_id.get("jaundice_syndromic_evaluation") or scored_candidates[0]
        if condition_id == "benign_biliary_stricture":
            return plan_by_id.get("biliary_stricture_characterization") or scored_candidates[0]
        if condition_id == "hepatocellular_jaundice_pattern":
            return plan_by_id.get("hepatocellular_injury_workup") or scored_candidates[0]
        if condition_id in {"malignant_biliary_obstruction", "hilar_or_distal_malignant_obstruction_requiring_mdt"}:
            return plan_by_id.get("malignant_biliary_obstruction_mdt") or scored_candidates[0]
        if condition_id == "colorectal_or_other_liver_metastases":
            return plan_by_id.get("liver_metastases_mdt_pathway") or scored_candidates[0]
        if condition_id == "pancreatic_cystic_lesion_for_mdt":
            return plan_by_id.get("distal_pancreatectomy_pathway") or scored_candidates[0]
        if condition_id == "hcc_resection_candidate":
            return plan_by_id.get("hepatectomy_perioperative_pathway") or scored_candidates[0]
        if condition_id == "pancreatic_head_mass_resection_candidate":
            return plan_by_id.get("pancreaticoduodenectomy_pathway") or scored_candidates[0]
        if condition_id == "pancreatic_body_tail_mass_candidate":
            return plan_by_id.get("distal_pancreatectomy_pathway") or scored_candidates[0]
        if condition_id == "choledocholithiasis_for_ercp_or_surgery":
            return plan_by_id.get("choledocholithiasis_source_control") or scored_candidates[0]
        if condition_id == "postoperative_bile_leak":
            return plan_by_id.get("postoperative_bile_leak_management") or scored_candidates[0]
        if condition_id == "postoperative_pancreatic_fistula":
            return plan_by_id.get("postoperative_pancreatic_fistula_management") or scored_candidates[0]
        if condition_id == "upper_gi_or_hepatobiliary_bleeding":
            return plan_by_id.get("hepatobiliary_bleeding_stabilization") or scored_candidates[0]
        if condition_id in {"gallbladder_cancer_resection_candidate", "liver_transplant_or_bridge_therapy_candidate"}:
            return plan_by_id.get("incidental_liver_mass_mdt") or scored_candidates[0]
        if condition_id == "colorectal_liver_metastases_resection_candidate":
            return plan_by_id.get("liver_metastases_mdt_pathway") or scored_candidates[0]

        if "黄疸" in chief_text and "发热" not in chief_text and "jaundice_requires_pattern_classification_before_definitive_pathway" in rule_ids:
            return plan_by_id.get("jaundice_syndromic_evaluation") or scored_candidates[0]
        if "ercp" in procedure_text:
            return plan_by_id.get("choledocholithiasis_source_control") or scored_candidates[0]
        if "whipple" in procedure_text or "胰十二指肠切除" in procedure_text:
            return plan_by_id.get("pancreaticoduodenectomy_pathway") or scored_candidates[0]
        if "胆漏" in postop_text or "胰瘘" in postop_text:
            return plan_by_id.get("postoperative_bile_leak_management") or scored_candidates[0]

        return scored_candidates[0]

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

        if any(term in comorbidity_text for term in ["肝硬化", "糖尿病", "慢性肾病", "高龄"]):
            flags.append("存在重要基础疾病，需结合肝储备、感染和围术期风险综合评估")
        if normalized_patient.get("imaging_summary") is None:
            flags.append("缺少影像学摘要")
        if primary_condition and primary_condition["id"] in {
            "acute_liver_failure_or_severe_hepatocellular_injury",
            "acute_cholangitis",
            "upper_gi_or_hepatobiliary_bleeding",
            "postoperative_pancreatic_fistula",
        }:
            flags.append("当前疑似高危肝胆胰急症，应优先处理复苏、器官支持或源控制")

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
        if normalized_patient.get("imaging_summary"):
            trace.append(f"已纳入影像学摘要：{normalized_patient['imaging_summary']}。")
        if matched_rules:
            trace.append("已同步参考安全规则命中结果。")
        return trace
