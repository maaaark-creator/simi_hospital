from __future__ import annotations

from typing import Any

from patient_mapper_agent import AnesthesiaKnowledgeBase


class KnowledgeRetriever:
    def __init__(self, kb: AnesthesiaKnowledgeBase) -> None:
        self.kb = kb

    def retrieve(
        self,
        normalized_patient: dict[str, Any],
        entity_linking: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        asa_match_id = self._map_asa_hint(normalized_patient.get("asa_hint"))
        candidate_plan_ids = self._infer_candidate_plan_ids(normalized_patient, entity_linking)
        safety_rule_ids = self._infer_safety_rule_ids(normalized_patient, entity_linking)

        candidate_plans = [
            self.kb.get_plan(plan_id)
            for plan_id in candidate_plan_ids
            if self.kb.get_plan(plan_id) is not None
        ]
        matched_rules = [
            rule
            for rule in self.kb.list_safety_rules()
            if rule["id"] in safety_rule_ids
        ]

        return {
            "asa_match": self.kb.get_asa(asa_match_id) if asa_match_id else None,
            "candidate_plans": candidate_plans,
            "matched_safety_rules": matched_rules,
            "related_drugs": self._collect_related_drugs(candidate_plans),
            "retrieval_notes": self._build_retrieval_notes(candidate_plans, matched_rules),
            "missing_information": self._identify_missing_information(normalized_patient),
        }

    def _map_asa_hint(self, asa_hint: Any) -> str | None:
        if asa_hint is None:
            return None
        text = str(asa_hint).upper().replace(" ", "")
        if "ASA" not in text:
            text = f"ASA{text}"
        if "VI" in text:
            return "ASA_VI"
        if "IV" in text:
            return "ASA_IV"
        if "III" in text:
            return "ASA_III"
        if "II" in text:
            return "ASA_II"
        if "I" in text:
            return "ASA_I"
        return None

    def _infer_candidate_plan_ids(
        self,
        normalized_patient: dict[str, Any],
        entity_linking: dict[str, Any] | None,
    ) -> list[str]:
        procedure_text = " ".join(
            filter(
                None,
                [
                    str(normalized_patient.get("procedure_name") or ""),
                    str(normalized_patient.get("procedure_site") or ""),
                ],
            )
        )
        fasting_text = str(normalized_patient.get("fasting_status") or "")
        airway_text = str(normalized_patient.get("airway_notes") or "")
        linked_keywords = " ".join((entity_linking or {}).get("procedure_keywords", []))
        risk_keywords = " ".join((entity_linking or {}).get("risk_keywords", []))

        plan_ids: list[str] = []

        if any(token in procedure_text + linked_keywords for token in ["下肢", "下腹", "会阴"]):
            plan_ids.append("spinal_anesthesia")

        if any(token in procedure_text + linked_keywords for token in ["短小", "表浅", "内镜", "局麻"]):
            plan_ids.append("monitored_anesthesia_care")

        if any(
            token in fasting_text + airway_text + risk_keywords
            for token in ["未禁食", "饱胃", "误吸", "困难气道"]
        ):
            plan_ids.append("general_anesthesia_ett")

        if not plan_ids:
            plan_ids.append("general_anesthesia_ett")

        return list(dict.fromkeys(plan_ids))

    def _infer_safety_rule_ids(
        self,
        normalized_patient: dict[str, Any],
        entity_linking: dict[str, Any] | None,
    ) -> list[str]:
        comorbidity_text = " ".join(normalized_patient.get("comorbidities", []))
        fasting_text = str(normalized_patient.get("fasting_status") or "")
        linked_risks = " ".join((entity_linking or {}).get("risk_keywords", []))
        rule_ids: list[str] = []

        if any(
            token in comorbidity_text + linked_risks
            for token in ["恶性高热", "malignant hyperthermia", "MH"]
        ):
            rule_ids.append("mh_trigger_avoidance")

        if any(
            token in fasting_text + linked_risks
            for token in ["未禁食", "饱胃", "胃排空延迟", "肠梗阻"]
        ):
            rule_ids.append("aspiration_risk_review")

        return list(dict.fromkeys(rule_ids))

    def _collect_related_drugs(self, candidate_plans: list[dict[str, Any]]) -> list[dict[str, Any]]:
        drug_ids: list[str] = []
        for plan in candidate_plans:
            for ids in plan.get("candidate_drugs", {}).values():
                drug_ids.extend(ids)

        related_drugs: list[dict[str, Any]] = []
        for drug_id in dict.fromkeys(drug_ids):
            drug = self.kb.get_drug(drug_id)
            if drug is not None:
                related_drugs.append(drug)
        return related_drugs

    def _build_retrieval_notes(
        self,
        candidate_plans: list[dict[str, Any]],
        matched_rules: list[dict[str, Any]],
    ) -> list[str]:
        notes: list[str] = []
        if candidate_plans:
            notes.append(f"已召回 {len(candidate_plans)} 个候选方案。")
        if matched_rules:
            notes.append(f"已命中 {len(matched_rules)} 条安全规则。")
        return notes

    def _identify_missing_information(self, normalized_patient: dict[str, Any]) -> list[str]:
        missing: list[str] = []
        if normalized_patient.get("height_cm") is None:
            missing.append("缺少身高信息")
        if not normalized_patient.get("labs"):
            missing.append("缺少实验室检查摘要")
        if normalized_patient.get("procedure_site") is None:
            missing.append("缺少手术部位结构化字段")
        if normalized_patient.get("airway_notes") is None:
            missing.append("缺少气道评估")
        return missing
