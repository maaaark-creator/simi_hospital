from __future__ import annotations

import json
from typing import Any

from patient_mapper_agent import (
    GenAIChatClient,
    MODEL_CONFIGS,
    TASK_MODEL_SELECTION,
)


class AnesthesiaDecisionAgent:
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
        candidate_plan_scores = retrieval_result.get("candidate_plan_scores", {})
        matched_rules = retrieval_result.get("matched_safety_rules", [])
        missing_information = retrieval_result.get("missing_information", [])

        primary_plan = self._choose_primary_plan(
            normalized_patient,
            candidate_plans,
            matched_rules,
            candidate_plan_scores,
        )
        backup_plans = [
            plan for plan in candidate_plans
            if primary_plan is None or plan["id"] != primary_plan["id"]
        ]

        return {
            "patient_summary": self._build_patient_summary(normalized_patient),
            "primary_plan": primary_plan,
            "backup_plans": backup_plans,
            "risk_flags": self._build_risk_flags(normalized_patient, matched_rules),
            "need_more_info": missing_information,
            "reasoning_trace": self._build_reasoning_trace(
                normalized_patient,
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
            "You are an anesthesia decision-support agent.\n"
            "Given the normalized patient profile and retrieval results, "
            "return JSON only with keys:\n"
            "patient_summary, primary_plan_id, backup_plan_ids, risk_flags, need_more_info, reasoning_trace.\n\n"
            "Normalized patient:\n"
            f"{json.dumps(normalized_patient, ensure_ascii=False, indent=2)}\n\n"
            "Retrieval result:\n"
            f"{json.dumps(retrieval_result, ensure_ascii=False, indent=2)}"
        )
        config = MODEL_CONFIGS[TASK_MODEL_SELECTION["plan_ranking"]]
        response = self.api_client.chat_json(config, prompt)
        return self.api_client.extract_json(response)

    def _choose_primary_plan(
        self,
        normalized_patient: dict[str, Any],
        candidate_plans: list[dict[str, Any]],
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
        fasting_text = str(normalized_patient.get("fasting_status") or "")
        airway_text = str(normalized_patient.get("airway_notes") or "")
        procedure_text = " ".join(
            filter(
                None,
                [
                    str(normalized_patient.get("procedure_name") or ""),
                    str(normalized_patient.get("procedure_site") or ""),
                ],
            )
        )

        if any(token in fasting_text + airway_text for token in ["未禁食", "饱胃", "误吸", "困难气道"]):
            return plan_by_id.get("general_anesthesia_ett") or scored_candidates[0]

        if any(rule["id"] == "aspiration_risk_review" for rule in matched_rules):
            return plan_by_id.get("general_anesthesia_ett") or scored_candidates[0]

        if any(token in procedure_text for token in ["下肢", "下腹", "会阴"]):
            return plan_by_id.get("spinal_anesthesia") or scored_candidates[0]

        return scored_candidates[0]

    def _build_patient_summary(self, normalized_patient: dict[str, Any]) -> str:
        summary_parts = [
            f"年龄 {normalized_patient.get('age')}" if normalized_patient.get("age") is not None else "年龄未提供",
            f"性别 {normalized_patient.get('sex')}" if normalized_patient.get("sex") else "性别未提供",
            f"拟行 {normalized_patient.get('procedure_name')}" if normalized_patient.get("procedure_name") else "术式未明确",
            f"ASA 提示 {normalized_patient.get('asa_hint')}" if normalized_patient.get("asa_hint") else "ASA 未明确",
        ]
        return "；".join(summary_parts)

    def _build_risk_flags(
        self,
        normalized_patient: dict[str, Any],
        matched_rules: list[dict[str, Any]],
    ) -> list[str]:
        flags = [rule["name_zh"] for rule in matched_rules]

        comorbidities = normalized_patient.get("comorbidities", [])
        if any(item in " ".join(comorbidities) for item in ["高血压", "糖尿病"]):
            flags.append("存在基础疾病，需要围术期进一步评估")

        if normalized_patient.get("airway_notes") is None:
            flags.append("缺少完整气道评估")

        return list(dict.fromkeys(flags))

    def _build_reasoning_trace(
        self,
        normalized_patient: dict[str, Any],
        primary_plan: dict[str, Any] | None,
        matched_rules: list[dict[str, Any]],
    ) -> list[str]:
        trace: list[str] = []
        if primary_plan is not None:
            trace.append(f"首选方案暂定为 {primary_plan['name_zh']}。")
        if normalized_patient.get("procedure_name"):
            trace.append(f"方案判断参考了术式信息：{normalized_patient['procedure_name']}。")
        if matched_rules:
            trace.append("已同时参考安全规则命中结果。")
        if normalized_patient.get("asa_hint") is not None:
            trace.append(f"已纳入 ASA 提示：{normalized_patient['asa_hint']}。")
        return trace
