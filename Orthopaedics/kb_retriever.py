from __future__ import annotations

from typing import Any

from patient_mapper_agent import (
    GenAIChatClient,
    OrthopaedicsKnowledgeBase,
    PatientProfileMapperAgent,
)


class KnowledgeRetriever:
    def __init__(
        self,
        kb: OrthopaedicsKnowledgeBase,
        api_client: GenAIChatClient | None = None,
    ) -> None:
        self.kb = kb
        self.api_client = api_client or GenAIChatClient()
        self.mapper = PatientProfileMapperAgent(kb)

    def retrieve(
        self,
        normalized_patient: dict[str, Any],
        entity_linking: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        asa_match_id = self.mapper._map_asa_hint(normalized_patient.get("asa_hint"))
        pathway_ids = self.mapper._infer_candidate_pathway_ids(normalized_patient)
        condition_ids = self.mapper._infer_candidate_condition_ids(normalized_patient)
        rule_ids = self.mapper._infer_risk_rule_ids(normalized_patient)

        linked_text = " ".join((entity_linking or {}).get("risk_keywords", []))
        if "感染" in linked_text and "infection_inflammatory_pathway" not in pathway_ids:
            pathway_ids.append("infection_inflammatory_pathway")
        if "肿瘤" in linked_text and "pathologic_fracture_or_bone_tumor" not in condition_ids:
            condition_ids.append("pathologic_fracture_or_bone_tumor")

        candidate_pathways = [
            self.kb.get_pathway(pathway_id)
            for pathway_id in dict.fromkeys(pathway_ids)
            if self.kb.get_pathway(pathway_id) is not None
        ]
        suspected_conditions = [
            self.kb.get_condition(condition_id)
            for condition_id in dict.fromkeys(condition_ids)
            if self.kb.get_condition(condition_id) is not None
        ]
        matched_rules = [
            rule
            for rule in self.kb.list_perioperative_rules()
            if rule["id"] in dict.fromkeys(rule_ids)
        ]

        return {
            "asa_match": self.kb.get_asa(asa_match_id) if asa_match_id else None,
            "candidate_pathways": candidate_pathways,
            "candidate_plans": candidate_pathways,
            "suspected_conditions": suspected_conditions,
            "matched_perioperative_rules": matched_rules,
            "matched_safety_rules": matched_rules,
            "recommended_workups": self._collect_workup_bundles(pathway_ids, condition_ids),
            "complication_watchlist": self._collect_complications(pathway_ids, condition_ids, rule_ids),
            "retrieval_notes": self._build_retrieval_notes(candidate_pathways, suspected_conditions, matched_rules),
            "missing_information": self._identify_missing_information(normalized_patient),
        }

    def retrieve_with_api(
        self,
        normalized_patient: dict[str, Any],
        entity_linking: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.retrieve(normalized_patient, entity_linking)

    def _collect_workup_bundles(
        self,
        pathway_ids: list[str],
        condition_ids: list[str],
    ) -> list[dict[str, Any]]:
        bundle_ids: list[str] = []
        if "acute_trauma_fracture_dislocation" in pathway_ids:
            bundle_ids.append("acute_trauma_bundle")
        if "degenerative_joint_disease" in pathway_ids:
            bundle_ids.append("degenerative_joint_bundle")
        if "spine_neurologic_pathway" in pathway_ids:
            bundle_ids.append("spine_neurologic_bundle")
        if "infection_inflammatory_pathway" in pathway_ids:
            bundle_ids.append("infection_bundle")
        if any("osteoarthritis" in condition_id or "fracture" in condition_id for condition_id in condition_ids):
            bundle_ids.append("elective_major_ortho_preop_bundle")

        bundles: list[dict[str, Any]] = []
        for bundle in self.kb.data.get("workup_bundles", []):
            if bundle["id"] in dict.fromkeys(bundle_ids):
                bundles.append(bundle)
        return bundles

    def _collect_complications(
        self,
        pathway_ids: list[str],
        condition_ids: list[str],
        rule_ids: list[str],
    ) -> list[dict[str, Any]]:
        complication_ids: list[str] = []
        if "acute_trauma_fracture_dislocation" in pathway_ids or "vte_risk_review" in rule_ids:
            complication_ids.extend(["dvt_pe", "nonunion_or_implant_failure"])
        if "infection_inflammatory_pathway" in pathway_ids:
            complication_ids.append("surgical_site_or_implant_infection")
        if "urgent_neurovascular_review" in rule_ids or "compartment_syndrome_or_acute_limb_ischaemia" in condition_ids:
            complication_ids.append("compartment_or_neurovascular_failure")
        if "geriatric_frailty_review" in rule_ids:
            complication_ids.append("delirium_and_functional_decline")
        if "joint_replacement_blood_loss_plan" in rule_ids or "hip_fracture_or_occult_hip_fracture" in condition_ids:
            complication_ids.append("major_bleeding_or_transfusion")

        complications: list[dict[str, Any]] = []
        for item in self.kb.data.get("complication_catalog", []):
            if item["id"] in dict.fromkeys(complication_ids):
                complications.append(item)
        return complications

    def _build_retrieval_notes(
        self,
        pathways: list[dict[str, Any]],
        conditions: list[dict[str, Any]],
        rules: list[dict[str, Any]],
    ) -> list[str]:
        notes: list[str] = []
        if pathways:
            notes.append(f"已召回 {len(pathways)} 条骨科评估路径。")
        if conditions:
            notes.append(f"已召回 {len(conditions)} 个重点疑似诊断。")
        if rules:
            notes.append(f"已命中 {len(rules)} 条围手术期/安全规则。")
        return notes

    def _identify_missing_information(self, normalized_patient: dict[str, Any]) -> list[str]:
        missing: list[str] = []
        if not normalized_patient.get("chief_complaint"):
            missing.append("缺少主诉")
        if not normalized_patient.get("pain_site"):
            missing.append("缺少疼痛/病变部位")
        if not normalized_patient.get("laterality"):
            missing.append("缺少侧别")
        if not normalized_patient.get("trauma_history") and not normalized_patient.get("onset_mechanism"):
            missing.append("缺少受伤史或起病机制")
        if not normalized_patient.get("imaging"):
            missing.append("缺少影像学摘要")
        if not normalized_patient.get("vitals"):
            missing.append("缺少生命体征")
        if not normalized_patient.get("medications") and not normalized_patient.get("anticoagulants"):
            missing.append("缺少长期用药/抗凝抗血小板信息")
        if not normalized_patient.get("family_history"):
            missing.append("缺少家族史")
        if (normalized_patient.get("age") or 0) >= 70 and not normalized_patient.get("assistive_device"):
            missing.append("高龄患者缺少助行能力/功能状态描述")
        return missing


OrthopaedicsKnowledgeRetriever = KnowledgeRetriever
