from __future__ import annotations

from typing import Any
from hbp_kb_retriever import KnowledgeRetriever as _GraphKnowledgeRetriever

from patient_mapper_agent import HepatobiliaryKnowledgeBase


class KnowledgeRetriever(_GraphKnowledgeRetriever):
    """Backward-compatible alias that uses the graph-enabled HBP retriever."""


class _LegacyKnowledgeRetriever:
    def __init__(self, kb: HepatobiliaryKnowledgeBase) -> None:
        self.kb = kb

    def retrieve(
        self,
        normalized_patient: dict[str, Any],
        entity_linking: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        terminology_hits = self._collect_terminology_hits(normalized_patient, entity_linking)
        matched_clusters = self._match_chief_complaint_clusters(terminology_hits)
        matched_medical_conditions = self._match_medical_conditions(normalized_patient, entity_linking)
        matched_surgical_conditions = self._match_surgical_conditions(normalized_patient, entity_linking)
        matched_templates = self._match_procedure_templates(normalized_patient, entity_linking)
        matched_comorbidity_rules = self._match_comorbidity_rules(normalized_patient)
        matched_contraindications = self._match_contraindications(normalized_patient, entity_linking)
        matched_emergency_algorithms = self._match_emergency_algorithms(normalized_patient, entity_linking)
        candidate_plan_ids = self._infer_candidate_plan_ids(
            normalized_patient,
            matched_clusters,
            matched_medical_conditions,
            matched_surgical_conditions,
            matched_templates,
            matched_emergency_algorithms,
        )
        safety_rule_ids = self._infer_safety_rule_ids(
            normalized_patient,
            matched_medical_conditions,
            matched_surgical_conditions,
            matched_emergency_algorithms,
        )

        candidate_plans = [
            self.kb.get_care_plan(plan_id)
            for plan_id in candidate_plan_ids
            if self.kb.get_care_plan(plan_id) is not None
        ]
        matched_rules = [
            rule for rule in self.kb.list_safety_rules() if rule["id"] in safety_rule_ids
        ]

        return {
            "matched_chief_complaint_clusters": matched_clusters,
            "matched_medical_conditions": matched_medical_conditions,
            "matched_surgical_conditions": matched_surgical_conditions,
            "candidate_plans": candidate_plans,
            "matched_safety_rules": matched_rules,
            "matched_emergency_algorithms": matched_emergency_algorithms,
            "matched_procedure_templates": matched_templates,
            "matched_comorbidity_rules": matched_comorbidity_rules,
            "matched_contraindications": matched_contraindications,
            "related_drugs": self._collect_related_drugs(candidate_plans, matched_medical_conditions),
            "terminology_hits": terminology_hits,
            "retrieval_notes": self._build_retrieval_notes(
                matched_clusters,
                matched_medical_conditions,
                matched_surgical_conditions,
                candidate_plans,
                matched_rules,
                matched_emergency_algorithms,
            ),
            "missing_information": self._identify_missing_information(normalized_patient),
        }

    def _infer_candidate_plan_ids(
        self,
        normalized_patient: dict[str, Any],
        matched_clusters: list[dict[str, Any]],
        matched_medical_conditions: list[dict[str, Any]],
        matched_surgical_conditions: list[dict[str, Any]],
        matched_templates: list[dict[str, Any]],
        matched_emergency_algorithms: list[dict[str, Any]],
    ) -> list[str]:
        plan_ids: list[str] = []
        condition_ids = {item["id"] for item in matched_medical_conditions}
        surgical_ids = {item["id"] for item in matched_surgical_conditions}
        cluster_ids = {item["id"] for item in matched_clusters}
        procedure_text = str(normalized_patient.get("procedure_name") or "").lower()
        diagnosis_text = str(normalized_patient.get("diagnosis_hint") or "").lower()
        postop_text = str(normalized_patient.get("postop_context") or "").lower()

        if "acute_cholangitis" in condition_ids or "cluster_jaundice_fever" in cluster_ids:
            plan_ids.append("acute_cholangitis_resuscitation")
        if "acute_pancreatitis" in condition_ids:
            plan_ids.append("acute_pancreatitis_supportive_care")
        if "obstructive_jaundice_needing_biliary_drainage" in condition_ids:
            plan_ids.append("obstructive_jaundice_workup")
        if "hcc_resection_candidate" in surgical_ids:
            plan_ids.append("hepatectomy_perioperative_pathway")
        if "pancreatic_head_mass_resection_candidate" in surgical_ids:
            plan_ids.append("pancreaticoduodenectomy_pathway")
        if "choledocholithiasis_for_ercp_or_surgery" in surgical_ids:
            plan_ids.append("choledocholithiasis_source_control")
        if "postoperative_bile_leak" in surgical_ids:
            plan_ids.append("postoperative_bile_leak_management")
        if "postoperative_pancreatic_fistula" in surgical_ids:
            plan_ids.append("postoperative_pancreatic_fistula_management")
        if "upper_gi_or_hepatobiliary_bleeding" in surgical_ids:
            plan_ids.append("hepatobiliary_bleeding_stabilization")

        for algorithm in matched_emergency_algorithms:
            plan_ids.extend(algorithm.get("common_candidate_plans", []))
        for template in matched_templates:
            plan_ids.extend(template.get("common_candidate_plans", []))

        if "ercp" in procedure_text:
            plan_ids.append("choledocholithiasis_source_control")
        if "whipple" in procedure_text or "胰十二指肠切除" in procedure_text:
            plan_ids.append("pancreaticoduodenectomy_pathway")
        if "肝切除" in procedure_text or "hepatectomy" in procedure_text:
            plan_ids.append("hepatectomy_perioperative_pathway")

        if not plan_ids and any(token in diagnosis_text for token in ["胆管炎", "cholangitis"]):
            plan_ids.append("acute_cholangitis_resuscitation")
        if not plan_ids and any(token in diagnosis_text for token in ["胰腺炎", "pancreatitis"]):
            plan_ids.append("acute_pancreatitis_supportive_care")
        if not plan_ids and any(token in postop_text for token in ["胆漏", "胰瘘", "术后感染"]):
            plan_ids.append("postoperative_bile_leak_management")
        if not plan_ids:
            plan_ids.append("obstructive_jaundice_workup")

        return list(dict.fromkeys(plan_ids))

    def _infer_safety_rule_ids(
        self,
        normalized_patient: dict[str, Any],
        matched_medical_conditions: list[dict[str, Any]],
        matched_surgical_conditions: list[dict[str, Any]],
        matched_emergency_algorithms: list[dict[str, Any]],
    ) -> list[str]:
        condition_ids = {item["id"] for item in matched_medical_conditions}
        surgical_ids = {item["id"] for item in matched_surgical_conditions}
        medication_text = " ".join(normalized_patient.get("medications", [])).lower()
        comorbidity_text = " ".join(normalized_patient.get("comorbidities", [])).lower()
        labs_text = " ".join(f"{k}:{v}" for k, v in normalized_patient.get("labs", {}).items()).lower()
        infection_text = str(normalized_patient.get("infection_status") or "").lower()
        bleeding_text = str(normalized_patient.get("bleeding_status") or "").lower()
        liver_function_text = str(normalized_patient.get("liver_function") or "").lower()
        urgency_text = str(normalized_patient.get("urgency") or "").lower()
        postop_text = str(normalized_patient.get("postop_context") or "").lower()

        rule_ids: list[str] = []
        if "acute_cholangitis" in condition_ids:
            rule_ids.append("source_control_needed_for_severe_cholangitis")
        if matched_emergency_algorithms and any(a["id"] == "biliary_sepsis_algorithm" for a in matched_emergency_algorithms):
            rule_ids.append("sepsis_bundle_hemodynamic_review")
        if "acute_pancreatitis" in condition_ids:
            rule_ids.append("avoid_unnecessary_early_antibiotics_in_pancreatitis")
        if any(token in bleeding_text + postop_text for token in ["活动性出血", "hematemesis", "黑便", "出血性休克"]):
            rule_ids.append("active_bleeding_requires_resuscitation_and_hemostasis")
        if any(token in comorbidity_text + liver_function_text + labs_text for token in ["肝硬化", "cirrhosis", "inr", "血小板", "platelet"]):
            rule_ids.append("coagulopathy_and_liver_reserve_review_before_invasive_procedure")
        if any(token in comorbidity_text + labs_text for token in ["ckd", "肾功能不全", "肌酐", "egfr"]):
            rule_ids.append("renal_function_review_for_contrast_and_antibiotics")
        if any(token in labs_text + medication_text for token in ["他克莫司", "tacrolimus", "化疗", "免疫抑制"]):
            rule_ids.append("immunosuppression_infection_escalation_review")
        if "postoperative_bile_leak" in surgical_ids or "postoperative_pancreatic_fistula" in surgical_ids:
            rule_ids.append("postoperative_drain_output_and_source_control_review")
        if "obstructive_jaundice_needing_biliary_drainage" in condition_ids:
            rule_ids.append("bilirubin_and_cholangitis_risk_review_before_major_surgery")
        if "急诊" in urgency_text and ("发热" in infection_text or "黄疸" in str(normalized_patient.get("chief_complaint") or "")):
            rule_ids.append("source_control_needed_for_severe_cholangitis")

        return list(dict.fromkeys(rule_ids))

    def _collect_related_drugs(
        self,
        candidate_plans: list[dict[str, Any]],
        matched_medical_conditions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        drug_ids: list[str] = []
        for plan in candidate_plans:
            for ids in plan.get("candidate_drugs", {}).values():
                drug_ids.extend(ids)
        for condition in matched_medical_conditions:
            drug_ids.extend(condition.get("common_related_drugs", []))
        related_drugs: list[dict[str, Any]] = []
        for drug_id in dict.fromkeys(drug_ids):
            drug = self.kb.get_drug(drug_id)
            if drug is not None:
                related_drugs.append(drug)
        return related_drugs

    def _build_retrieval_notes(
        self,
        matched_clusters: list[dict[str, Any]],
        matched_medical_conditions: list[dict[str, Any]],
        matched_surgical_conditions: list[dict[str, Any]],
        candidate_plans: list[dict[str, Any]],
        matched_rules: list[dict[str, Any]],
        matched_emergency_algorithms: list[dict[str, Any]],
    ) -> list[str]:
        notes: list[str] = []
        if matched_clusters:
            notes.append(f"已命中 {len(matched_clusters)} 个主诉簇。")
        if matched_medical_conditions:
            notes.append(f"已命中 {len(matched_medical_conditions)} 个肝胆胰内科/急症条目。")
        if matched_surgical_conditions:
            notes.append(f"已命中 {len(matched_surgical_conditions)} 个肝胆胰外科/围术期条目。")
        if candidate_plans:
            notes.append(f"已召回 {len(candidate_plans)} 条候选管理路径。")
        if matched_rules:
            notes.append(f"已命中 {len(matched_rules)} 条安全规则。")
        if matched_emergency_algorithms:
            notes.append(f"已命中 {len(matched_emergency_algorithms)} 条急症算法。")
        return notes

    def _identify_missing_information(self, normalized_patient: dict[str, Any]) -> list[str]:
        missing: list[str] = []
        if normalized_patient.get("chief_complaint") is None:
            missing.append("缺少主诉")
        if not normalized_patient.get("vitals"):
            missing.append("缺少生命体征摘要")
        if not normalized_patient.get("labs"):
            missing.append("缺少检验摘要")
        if normalized_patient.get("imaging_summary") is None:
            missing.append("缺少影像学摘要")
        return missing

    def _collect_terminology_hits(
        self,
        normalized_patient: dict[str, Any],
        entity_linking: dict[str, Any] | None,
    ) -> dict[str, list[str]]:
        aliases = self.kb.data.get("terminology_aliases", {})
        entity_linking = self._normalize_entity_linking(entity_linking)
        combined_text = " ".join(
            [
                str(normalized_patient.get("chief_complaint") or ""),
                " ".join(normalized_patient.get("symptoms", [])),
                str(normalized_patient.get("diagnosis_hint") or ""),
                str(normalized_patient.get("procedure_name") or ""),
                str(normalized_patient.get("imaging_summary") or ""),
                str(normalized_patient.get("pathology_summary") or ""),
                str(normalized_patient.get("drainage_status") or ""),
                str(normalized_patient.get("infection_status") or ""),
                str(normalized_patient.get("bleeding_status") or ""),
                str(normalized_patient.get("liver_function") or ""),
                str(normalized_patient.get("postop_context") or ""),
                " ".join(normalized_patient.get("comorbidities", [])),
                " ".join(normalized_patient.get("medications", [])),
                " ".join((entity_linking or {}).get("normalized_terms", [])),
                " ".join((entity_linking or {}).get("syndrome_keywords", [])),
                " ".join((entity_linking or {}).get("symptom_keywords", [])),
                " ".join((entity_linking or {}).get("imaging_keywords", [])),
                " ".join((entity_linking or {}).get("procedure_keywords", [])),
                " ".join((entity_linking or {}).get("risk_keywords", [])),
                " ".join((entity_linking or {}).get("medication_keywords", [])),
            ]
        ).lower()
        hits: dict[str, list[str]] = {}
        for group_name, group_values in aliases.items():
            group_hits: list[str] = []
            for canonical_name, term_aliases in group_values.items():
                if any(str(alias).lower() in combined_text for alias in term_aliases):
                    group_hits.append(canonical_name)
            hits[group_name] = group_hits
        return hits

    def _match_chief_complaint_clusters(self, terminology_hits: dict[str, list[str]]) -> list[dict[str, Any]]:
        matched: list[dict[str, Any]] = []
        symptom_hits = set(terminology_hits.get("symptom_terms", []))
        syndrome_hits = set(terminology_hits.get("syndrome_terms", []))
        for cluster in self.kb.data.get("chief_complaint_clusters", []):
            cluster_id = cluster["id"]
            if cluster_id == "cluster_jaundice_fever" and {"jaundice", "fever", "right_upper_quadrant_pain"} & symptom_hits:
                matched.append(cluster)
            if cluster_id == "cluster_pancreatitis_pain" and (
                "acute_pancreatitis" in syndrome_hits
                or {"epigastric_pain", "vomiting"} <= symptom_hits
            ):
                matched.append(cluster)
            if cluster_id == "cluster_postop_leak_or_infection" and {"drain_increase", "fever"} <= symptom_hits:
                matched.append(cluster)
            if cluster_id == "cluster_bleeding_or_shock" and {"melena", "hematemesis", "shock"} & symptom_hits:
                matched.append(cluster)
        return matched

    def _match_medical_conditions(
        self,
        normalized_patient: dict[str, Any],
        entity_linking: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        combined_text = self._combined_text(normalized_patient, entity_linking)
        matched: list[dict[str, Any]] = []
        matched_ids: set[str] = set()
        for condition in self.kb.data.get("medical_conditions", []):
            terms = [condition["id"], condition.get("name_zh", ""), condition.get("name_en", "")]
            terms.extend(condition.get("aliases", []))
            if any(str(term).lower() in combined_text for term in terms if term):
                matched.append(condition)
                matched_ids.add(condition["id"])
        explicit_condition_rules = {
            "acute_cholangitis": ["急性胆管炎", "cholangitis", "charcot", "寒战", "黄疸"],
            "obstructive_jaundice_needing_biliary_drainage": ["梗阻性黄疸", "胆道梗阻", "bilirubin", "胆道扩张"],
            "acute_pancreatitis": ["急性胰腺炎", "pancreatitis", "淀粉酶", "脂肪酶"],
            "acute_cholecystitis": ["急性胆囊炎", "cholecystitis"],
            "cirrhosis_with_portal_hypertension": ["肝硬化", "门静脉高压", "腹水", "食管胃底静脉曲张"],
        }
        for condition_id, terms in explicit_condition_rules.items():
            if condition_id in matched_ids:
                continue
            if any(term.lower() in combined_text for term in terms):
                condition = self.kb.get_medical_condition(condition_id)
                if condition is not None:
                    matched.append(condition)
                    matched_ids.add(condition_id)
        return matched

    def _match_surgical_conditions(
        self,
        normalized_patient: dict[str, Any],
        entity_linking: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        combined_text = self._combined_text(normalized_patient, entity_linking)
        matched: list[dict[str, Any]] = []
        matched_ids: set[str] = set()
        for condition in self.kb.data.get("surgical_conditions", []):
            terms = [condition["id"], condition.get("name_zh", ""), condition.get("name_en", "")]
            terms.extend(condition.get("typical_scenarios", []))
            if any(str(term).lower() in combined_text for term in terms if term):
                matched.append(condition)
                matched_ids.add(condition["id"])
        explicit_condition_rules = {
            "hcc_resection_candidate": ["肝癌", "肝细胞癌", "hcc", "肝切除"],
            "pancreatic_head_mass_resection_candidate": ["胰头肿物", "壶腹周围肿瘤", "whipple", "胰十二指肠切除"],
            "choledocholithiasis_for_ercp_or_surgery": ["胆总管结石", "choledocholithiasis", "ercp"],
            "postoperative_bile_leak": ["胆漏", "bile leak", "胆汁性引流"],
            "postoperative_pancreatic_fistula": ["胰瘘", "pancreatic fistula", "高淀粉酶引流"],
            "upper_gi_or_hepatobiliary_bleeding": ["呕血", "黑便", "消化道出血", "hemobilia"],
        }
        for condition_id, terms in explicit_condition_rules.items():
            if condition_id in matched_ids:
                continue
            if any(term.lower() in combined_text for term in terms):
                condition = self.kb.get_surgical_condition(condition_id)
                if condition is not None:
                    matched.append(condition)
                    matched_ids.add(condition_id)
        return matched

    def _match_procedure_templates(
        self,
        normalized_patient: dict[str, Any],
        entity_linking: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        combined_text = self._combined_text(normalized_patient, entity_linking)
        matched: list[dict[str, Any]] = []
        for template in self.kb.data.get("procedure_templates", []):
            template_id = template["id"]
            if template_id == "ercp_template" and any(token in combined_text for token in ["ercp", "胆道支架", "鼻胆管"]):
                matched.append(template)
            if template_id == "ptcd_template" and any(token in combined_text for token in ["ptcd", "经皮经肝胆道引流"]):
                matched.append(template)
            if template_id == "hepatectomy_template" and any(token in combined_text for token in ["hepatectomy", "肝切除"]):
                matched.append(template)
            if template_id == "whipple_template" and any(token in combined_text for token in ["whipple", "胰十二指肠切除"]):
                matched.append(template)
            if template_id == "lap_chole_template" and any(token in combined_text for token in ["胆囊切除", "lap chole", "腹腔镜胆囊"]):
                matched.append(template)
        return matched

    def _match_comorbidity_rules(self, normalized_patient: dict[str, Any]) -> list[dict[str, Any]]:
        combined_text = " ".join(normalized_patient.get("comorbidities", []) + normalized_patient.get("medications", [])).lower()
        return [
            rule
            for rule in self.kb.data.get("comorbidity_rules", [])
            if any(term.lower() in combined_text for term in rule.get("trigger_terms", []))
        ]

    def _match_contraindications(
        self,
        normalized_patient: dict[str, Any],
        entity_linking: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        combined_text = self._combined_text(normalized_patient, entity_linking)
        matched: list[dict[str, Any]] = []
        for item in self.kb.data.get("contraindication_catalog", []):
            item_id = item["id"]
            if item_id == "contra_active_uncontrolled_bleeding" and any(token in combined_text for token in ["活动性出血", "出血性休克", "呕血", "黑便"]):
                matched.append(item)
            if item_id == "contra_unresolved_septic_shock_for_definitive_major_resection" and any(token in combined_text for token in ["脓毒性休克", "乳酸升高", "去甲肾上腺素", "休克"]):
                matched.append(item)
            if item_id == "contra_poor_liver_reserve_major_hepatectomy" and any(token in combined_text for token in ["child c", "明显腹水", "严重黄疸", "门静脉高压"]):
                matched.append(item)
            if item_id == "contra_unreversed_anticoag_for_high_risk_procedure" and any(token in combined_text for token in ["warfarin", "apixaban", "rivaroxaban", "抗凝"]):
                matched.append(item)
        return matched

    def _match_emergency_algorithms(
        self,
        normalized_patient: dict[str, Any],
        entity_linking: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        combined_text = self._combined_text(normalized_patient, entity_linking)
        matched: list[dict[str, Any]] = []
        for algorithm in self.kb.data.get("emergency_algorithms", []):
            algorithm_id = algorithm["id"]
            if algorithm_id == "biliary_sepsis_algorithm" and any(token in combined_text for token in ["胆管炎", "寒战", "黄疸", "低血压", "感染性休克"]):
                matched.append(algorithm)
            if algorithm_id == "post_hepatectomy_liver_failure_watch" and any(token in combined_text for token in ["肝切除术后", "胆红素升高", "inr升高", "少尿"]):
                matched.append(algorithm)
            if algorithm_id == "post_pancreatectomy_complication_algorithm" and any(token in combined_text for token in ["胰瘘", "引流液淀粉酶高", "胰十二指肠切除术后", "胰切除术后"]):
                matched.append(algorithm)
            if algorithm_id == "hepatobiliary_bleeding_algorithm" and any(token in combined_text for token in ["呕血", "黑便", "血压低", "引流血性"]):
                matched.append(algorithm)
        return matched

    def _combined_text(self, normalized_patient: dict[str, Any], entity_linking: dict[str, Any] | None) -> str:
        entity_linking = self._normalize_entity_linking(entity_linking)
        return " ".join(
            [
                str(normalized_patient.get("chief_complaint") or ""),
                " ".join(normalized_patient.get("symptoms", [])),
                str(normalized_patient.get("diagnosis_hint") or ""),
                str(normalized_patient.get("procedure_name") or ""),
                str(normalized_patient.get("procedure_site") or ""),
                str(normalized_patient.get("urgency") or ""),
                str(normalized_patient.get("imaging_summary") or ""),
                str(normalized_patient.get("pathology_summary") or ""),
                str(normalized_patient.get("drainage_status") or ""),
                str(normalized_patient.get("infection_status") or ""),
                str(normalized_patient.get("bleeding_status") or ""),
                str(normalized_patient.get("liver_function") or ""),
                str(normalized_patient.get("postop_context") or ""),
                " ".join(normalized_patient.get("comorbidities", [])),
                " ".join(normalized_patient.get("medications", [])),
                " ".join(f"{k}:{v}" for k, v in normalized_patient.get("labs", {}).items()),
                " ".join(f"{k}:{v}" for k, v in normalized_patient.get("vitals", {}).items()),
                " ".join((entity_linking or {}).get("normalized_terms", [])),
                " ".join((entity_linking or {}).get("syndrome_keywords", [])),
                " ".join((entity_linking or {}).get("symptom_keywords", [])),
                " ".join((entity_linking or {}).get("imaging_keywords", [])),
                " ".join((entity_linking or {}).get("procedure_keywords", [])),
                " ".join((entity_linking or {}).get("risk_keywords", [])),
                " ".join((entity_linking or {}).get("medication_keywords", [])),
            ]
        ).lower()

    def _normalize_entity_linking(self, entity_linking: dict[str, Any] | None) -> dict[str, list[str]]:
        payload = entity_linking or {}
        keys = [
            "normalized_terms",
            "syndrome_keywords",
            "symptom_keywords",
            "imaging_keywords",
            "procedure_keywords",
            "risk_keywords",
            "medication_keywords",
        ]
        return {key: self._flatten_text_list(payload.get(key)) for key in keys}

    def _flatten_text_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, list):
            result: list[str] = []
            for item in value:
                result.extend(self._flatten_text_list(item))
            return result
        if isinstance(value, dict):
            result: list[str] = []
            for item in value.values():
                result.extend(self._flatten_text_list(item))
            return result
        return [str(value).strip()]
