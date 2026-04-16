from __future__ import annotations

from typing import Any


class MDTCallMapper:
    def build_initial_context(
        self,
        patient_input: dict[str, Any] | str,
    ) -> dict[str, Any]:
        return {
            "patient_input": patient_input,
            "patient_summary": self._build_patient_summary(patient_input),
        }

    def build_request_context(
        self,
        patient_input: dict[str, Any] | str,
        specialty_opinions: dict[str, Any],
        unresolved_items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "patient_input": patient_input,
            "specialty_summary": self._build_specialty_summary(specialty_opinions),
            "unresolved_items": unresolved_items,
        }

    def _build_patient_summary(self, patient_input: dict[str, Any] | str) -> dict[str, Any]:
        if isinstance(patient_input, str):
            return {"text_preview": patient_input[:300]}

        summary_keys = [
            "年龄",
            "性别",
            "主诉",
            "症状",
            "诊断提示",
            "拟行手术",
            "手术部位",
            "急诊",
            "基础疾病",
            "生命体征",
            "化验",
            "影像摘要",
            "感染情况",
            "ASA分级",
        ]
        return {key: patient_input.get(key) for key in summary_keys if key in patient_input}

    def _build_specialty_summary(self, specialty_opinions: dict[str, Any]) -> list[dict[str, Any]]:
        summary = []
        for specialty_id, payload in specialty_opinions.items():
            decision_result = payload.get("decision_result", {})
            summary.append(
                {
                    "specialty": specialty_id,
                    "specialty_label": payload.get("specialty_label"),
                    "primary_condition": self._extract_primary_condition(decision_result),
                    "primary_plan": self._extract_primary_plan(decision_result),
                    "risk_flags": self._string_list(decision_result.get("risk_flags")),
                    "need_more_info": self._string_list(decision_result.get("need_more_info")),
                }
            )
        return summary

    def _extract_primary_condition(self, decision_result: dict[str, Any]) -> str | None:
        if isinstance(decision_result.get("primary_condition"), dict):
            condition = decision_result["primary_condition"]
            return str(condition.get("name_zh") or condition.get("id") or "").strip() or None
        if decision_result.get("primary_condition_id"):
            return str(decision_result.get("primary_condition_id") or "").strip() or None
        return None

    def _extract_primary_plan(self, decision_result: dict[str, Any]) -> str | None:
        if isinstance(decision_result.get("primary_plan"), dict):
            plan = decision_result["primary_plan"]
            return str(plan.get("name_zh") or plan.get("id") or "").strip() or None
        if decision_result.get("primary_plan_id"):
            return str(decision_result.get("primary_plan_id") or "").strip() or None
        return None

    def _string_list(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []
