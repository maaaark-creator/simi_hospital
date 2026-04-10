from __future__ import annotations

import json
from typing import Any

from patient_mapper_agent import (
    CardiologyKnowledgeBase,
    GenAIChatClient,
    MODEL_CONFIGS,
    TASK_MODEL_SELECTION,
)


class KnowledgeRetriever:
    def __init__(
        self,
        kb: CardiologyKnowledgeBase,
        api_client: GenAIChatClient | None = None,
    ) -> None:
        self.kb = kb
        self.api_client = api_client or GenAIChatClient()

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
            rule
            for rule in self.kb.list_safety_rules()
            if rule["id"] in safety_rule_ids
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

    def retrieve_with_api(
        self,
        normalized_patient: dict[str, Any],
        entity_linking: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        prompt = (
            "You are a cardiology knowledge-retrieval agent.\n"
            "Review the normalized cardiovascular patient profile, optional entity-linking output, and the cardiology KB.\n"
            "Return JSON only with keys:\n"
            "matched_chief_complaint_clusters, matched_medical_conditions, matched_surgical_conditions, "
            "candidate_plans, matched_safety_rules, matched_emergency_algorithms, matched_procedure_templates, "
            "matched_comorbidity_rules, matched_contraindications, related_drugs, terminology_hits, retrieval_notes, missing_information.\n"
            "Schema rules:\n"
            "- All matched_* arrays and candidate_plans/related_drugs are arrays of KB objects.\n"
            "- terminology_hits is an object whose values are arrays of strings.\n"
            "- retrieval_notes and missing_information are arrays of strings.\n"
            "- Prefer exact KB ids and wording when available.\n\n"
            "Normalized patient:\n"
            f"{json.dumps(normalized_patient, ensure_ascii=False, indent=2)}\n\n"
            "Entity linking:\n"
            f"{json.dumps(entity_linking or {}, ensure_ascii=False, indent=2)}\n\n"
            "Cardiology KB:\n"
            f"{json.dumps(self.kb.data, ensure_ascii=False, indent=2)}"
        )
        config = MODEL_CONFIGS[TASK_MODEL_SELECTION["plan_ranking"]]
        response = self.api_client.chat_json(config, prompt)
        payload = self.api_client.extract_json(response)
        payload.setdefault("matched_chief_complaint_clusters", [])
        payload.setdefault("matched_medical_conditions", [])
        payload.setdefault("matched_surgical_conditions", [])
        payload.setdefault("candidate_plans", [])
        payload.setdefault("matched_safety_rules", [])
        payload.setdefault("matched_emergency_algorithms", [])
        payload.setdefault("matched_procedure_templates", [])
        payload.setdefault("matched_comorbidity_rules", [])
        payload.setdefault("matched_contraindications", [])
        payload.setdefault("related_drugs", [])
        payload.setdefault("terminology_hits", {})
        payload.setdefault("retrieval_notes", [])
        payload.setdefault("missing_information", [])
        return payload

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
        emergency_ids = {item["id"] for item in matched_emergency_algorithms}
        procedure_text = str(normalized_patient.get("procedure_name") or "").lower()
        diagnosis_text = str(normalized_patient.get("diagnosis_hint") or "").lower()
        postop_text = str(normalized_patient.get("postop_context") or "").lower()

        if condition_ids & {"stemi", "nste_acs"} or cluster_ids == {"cluster_chest_pain"}:
            plan_ids.append("acs_initial_evaluation")
        if "stemi" in condition_ids:
            plan_ids.append("stemi_reperfusion_pathway")
        if condition_ids & {"acute_decompensated_heart_failure", "hfrEF", "hfpEF"}:
            plan_ids.append("acute_hf_decongestion")
        if "hfrEF" in condition_ids:
            plan_ids.append("hfrEF_gdmt_buildout")
        if "atrial_fibrillation" in condition_ids:
            plan_ids.append("af_rate_rhythm_anticoag")
        if "supraventricular_tachycardia" in condition_ids:
            plan_ids.append("svt_termination_pathway")
        if surgical_ids & {"multivessel_cad_for_revascularization"} or "cabg" in procedure_text:
            plan_ids.append("cabg_perioperative_pathway")
        if surgical_ids & {"severe_aortic_stenosis", "severe_mitral_regurgitation"}:
            plan_ids.append("valve_intervention_pathway")
        if surgical_ids & {"acute_aortic_dissection", "ascending_aortic_aneurysm"} or "tevar" in procedure_text:
            plan_ids.append("aortic_emergency_pathway")

        for algorithm in matched_emergency_algorithms:
            plan_ids.extend(algorithm.get("common_candidate_plans", []))
        for template in matched_templates:
            plan_ids.extend(template.get("common_candidate_plans", []))

        if "post_cardiotomy_low_output_syndrome" in surgical_ids or "心外术后" in postop_text:
            plan_ids.append("cabg_perioperative_pathway")

        if not plan_ids and any(token in diagnosis_text for token in ["房颤", "af", "atrial fibrillation"]):
            plan_ids.append("af_rate_rhythm_anticoag")

        if not plan_ids and any(token in diagnosis_text for token in ["心衰", "heart failure"]):
            plan_ids.append("acute_hf_decongestion")

        if not plan_ids:
            plan_ids.append("acs_initial_evaluation")

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
        bleeding_text = str(normalized_patient.get("bleeding_status") or "").lower()
        ecg_text = str(normalized_patient.get("ecg_summary") or "").lower()
        labs_text = " ".join(
            f"{k}:{v}" for k, v in normalized_patient.get("labs", {}).items()
        ).lower()
        urgency_text = str(normalized_patient.get("urgency") or "").lower()
        postop_text = str(normalized_patient.get("postop_context") or "").lower()

        rule_ids: list[str] = []

        if "stemi" in condition_ids:
            rule_ids.append("stemi_reperfusion_priority")

        if "troponin" in labs_text or "肌钙蛋白" in labs_text:
            rule_ids.append("troponin_requires_context")

        if matched_emergency_algorithms and any(
            algorithm["id"] == "unstable_tachyarrhythmia_algorithm"
            for algorithm in matched_emergency_algorithms
        ):
            rule_ids.append("hemodynamic_instability_requires_urgent_rhythm_action")

        if "atrial_fibrillation" in condition_ids:
            rule_ids.append("af_anticoag_should_use_stroke_risk_assessment")

        if any(token in bleeding_text for token in ["活动性出血", "active bleeding", "major bleeding"]) or (
            "出血" in postop_text and "术后" in postop_text
        ):
            rule_ids.append("active_bleeding_reassess_antithrombotics")

        if any(token in comorbidity_text + labs_text for token in ["ckd", "肾功能不全", "acute kidney injury", "肌酐", "egfr"]):
            rule_ids.append("renal_function_review_for_anticoag_and_hf_drugs")

        if any(token in labs_text for token in ["高钾", "k:", "potassium", "血钾"]):
            rule_ids.append("hyperkalemia_review_for_raas_and_mra")

        if condition_ids & {"acute_decompensated_heart_failure"}:
            rule_ids.append("acute_hf_requires_wet_warm_profile_assessment")

        if any(token in ecg_text + comorbidity_text for token in ["右室梗死", "right ventricular infarction", "hypotension", "低血压"]):
            rule_ids.append("nitrate_hypotension_right_ventricle_caution")

        if any(token in comorbidity_text + medication_text for token in ["机械瓣", "mechanical valve"]):
            rule_ids.append("mechanical_valve_doac_avoidance")

        if "post_cardiotomy_low_output_syndrome" in surgical_ids or (
            "术后" in postop_text and any(token in postop_text for token in ["低血压", "引流", "低心排"])
        ):
            rule_ids.append("post_cardiac_surgery_tamponade_watch")

        if "acute_aortic_dissection" in surgical_ids:
            rule_ids.append("aortic_dissection_bp_hr_control")

        if "infective_endocarditis" in condition_ids:
            rule_ids.append("infective_endocarditis_team_based_management")

        if "急诊" in urgency_text and "胸痛" in str(normalized_patient.get("chief_complaint") or ""):
            rule_ids.append("stemi_reperfusion_priority")

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
            notes.append(f"已命中 {len(matched_medical_conditions)} 个心内科疾病条目。")
        if matched_surgical_conditions:
            notes.append(f"已命中 {len(matched_surgical_conditions)} 个心外科/结构性疾病条目。")
        if candidate_plans:
            notes.append(f"已召回 {len(candidate_plans)} 条候选管理路径。")
        if matched_rules:
            notes.append(f"已命中 {len(matched_rules)} 条安全规则。")
        if matched_emergency_algorithms:
            notes.append(f"已命中 {len(matched_emergency_algorithms)} 条急诊算法。")
        return notes

    def _identify_missing_information(self, normalized_patient: dict[str, Any]) -> list[str]:
        missing: list[str] = []
        if normalized_patient.get("chief_complaint") is None:
            missing.append("缺少主诉")
        if not normalized_patient.get("vitals"):
            missing.append("缺少生命体征摘要")
        if normalized_patient.get("ecg_summary") is None:
            missing.append("缺少心电图摘要")
        if not normalized_patient.get("labs"):
            missing.append("缺少检验摘要")
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
                str(normalized_patient.get("ecg_summary") or ""),
                str(normalized_patient.get("echo_summary") or ""),
                str(normalized_patient.get("imaging_summary") or ""),
                str(normalized_patient.get("bleeding_status") or ""),
                str(normalized_patient.get("postop_context") or ""),
                " ".join(normalized_patient.get("comorbidities", [])),
                " ".join(normalized_patient.get("medications", [])),
                " ".join((entity_linking or {}).get("normalized_terms", [])),
                " ".join((entity_linking or {}).get("syndrome_keywords", [])),
                " ".join((entity_linking or {}).get("symptom_keywords", [])),
                " ".join((entity_linking or {}).get("ecg_keywords", [])),
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

    def _match_chief_complaint_clusters(
        self,
        terminology_hits: dict[str, list[str]],
    ) -> list[dict[str, Any]]:
        matched: list[dict[str, Any]] = []
        symptom_hits = set(terminology_hits.get("symptom_terms", []))

        for cluster in self.kb.data.get("chief_complaint_clusters", []):
            cluster_id = cluster["id"]
            if cluster_id == "cluster_chest_pain" and {"chest_pain", "back_tearing_pain"} & symptom_hits:
                matched.append(cluster)
            if cluster_id == "cluster_dyspnea" and {"dyspnea", "edema"} & symptom_hits:
                matched.append(cluster)
            if cluster_id == "cluster_palpitation_syncope" and {"palpitation", "syncope"} & symptom_hits:
                matched.append(cluster)
            if cluster_id == "cluster_postop_low_output" and any(
                token in symptom_hits for token in {"dyspnea", "syncope"}
            ):
                continue
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
            "stemi": ["stemi", "st 段抬高", "st段抬高", "st elevation"],
            "nste_acs": ["nstemi", "不稳定型心绞痛", "nste-acs", "acute coronary syndrome"],
            "acute_decompensated_heart_failure": ["急性失代偿心衰", "肺水肿", "acute decompensated heart failure"],
            "atrial_fibrillation": ["房颤", "atrial fibrillation", "afib", "af "],
            "supraventricular_tachycardia": ["室上速", "svt", "psvt", "规则窄 qrs"],
            "pulmonary_embolism": ["肺栓塞", "pe", "pulmonary embolism"],
            "infective_endocarditis": ["感染性心内膜炎", "endocarditis", "ie"],
        }

        for condition_id, terms in explicit_condition_rules.items():
            if condition_id in matched_ids:
                continue
            if any(term in combined_text for term in terms):
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
            "acute_aortic_dissection": ["主动脉夹层", "aortic dissection", "撕裂样"],
            "ascending_aortic_aneurysm": ["主动脉瘤", "aortic aneurysm"],
            "severe_aortic_stenosis": ["重度主动脉瓣狭窄", "severe as", "主动脉瓣狭窄"],
            "severe_mitral_regurgitation": ["重度二尖瓣反流", "severe mr", "mitral regurgitation"],
            "multivessel_cad_for_revascularization": ["左主干", "多支病变", "cabg 评估", "revascularization"],
            "post_cardiotomy_low_output_syndrome": ["术后低心排", "post cardiotomy low output", "心外术后低心排"],
        }

        for condition_id, terms in explicit_condition_rules.items():
            if condition_id in matched_ids:
                continue
            if any(term in combined_text for term in terms):
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
            if template_id == "pci_template" and any(token in combined_text for token in ["pci", "支架", "冠脉介入"]):
                matched.append(template)
            if template_id == "cabg_template" and any(token in combined_text for token in ["cabg", "搭桥", "旁路移植"]):
                matched.append(template)
            if template_id == "valve_surgery_template" and any(token in combined_text for token in ["瓣膜", "avr", "mvr", "valve surgery"]):
                matched.append(template)
            if template_id == "tavr_template" and any(token in combined_text for token in ["tavr", "tavi"]):
                matched.append(template)
            if template_id == "aortic_surgery_template" and any(token in combined_text for token in ["tevar", "主动脉", "aortic repair", "bentall", "sun手术"]):
                matched.append(template)
            if template_id == "electrical_cardioversion_template" and any(
                token in combined_text for token in ["电复律", "cardioversion", "房颤", "svt"]
            ):
                matched.append(template)

        return matched

    def _match_comorbidity_rules(self, normalized_patient: dict[str, Any]) -> list[dict[str, Any]]:
        combined_text = " ".join(
            normalized_patient.get("comorbidities", []) + normalized_patient.get("medications", [])
        ).lower()
        matched: list[dict[str, Any]] = []

        for rule in self.kb.data.get("comorbidity_rules", []):
            if any(term.lower() in combined_text for term in rule.get("trigger_terms", [])):
                matched.append(rule)

        return matched

    def _match_contraindications(
        self,
        normalized_patient: dict[str, Any],
        entity_linking: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        combined_text = self._combined_text(normalized_patient, entity_linking)
        matched: list[dict[str, Any]] = []

        for item in self.kb.data.get("contraindication_catalog", []):
            item_id = item["id"]
            if item_id == "contra_active_major_bleeding" and any(
                token in combined_text for token in ["活动性出血", "major bleeding", "消化道出血", "术后出血"]
            ):
                matched.append(item)
            if item_id == "contra_mechanical_valve_doac" and any(
                token in combined_text for token in ["机械瓣", "mechanical valve", "人工机械瓣"]
            ):
                matched.append(item)
            if item_id == "contra_severe_bradycardia_beta_blocker" and any(
                token in combined_text for token in ["严重心动过缓", "高度房室传导阻滞", "complete heart block"]
            ):
                matched.append(item)
            if item_id == "contra_hyperkalemia_mra_raas" and any(
                token in combined_text for token in ["高钾", "hyperkalemia", "血钾升高"]
            ):
                matched.append(item)
            if item_id == "contra_hypotension_nitrates" and any(
                token in combined_text for token in ["低血压", "右室梗死", "right ventricular infarction"]
            ):
                matched.append(item)
            if item_id == "contra_renal_impairment_digoxin_accumulation" and any(
                token in combined_text for token in ["肾功能不全", "ckd", "egfr", "肌酐升高"]
            ):
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
            if algorithm_id == "cardiogenic_shock_algorithm" and any(
                token in combined_text for token in ["心源性休克", "低灌注", "乳酸升高", "冷湿", "休克"]
            ):
                matched.append(algorithm)
            if algorithm_id == "unstable_tachyarrhythmia_algorithm" and any(
                token in combined_text for token in ["室上速", "房颤", "wide qrs", "低血压", "意识障碍", "快速心律失常"]
            ):
                matched.append(algorithm)
            if algorithm_id == "unstable_bradyarrhythmia_algorithm" and any(
                token in combined_text for token in ["心动过缓", "房室传导阻滞", "晕厥", "临时起搏"]
            ):
                matched.append(algorithm)
            if algorithm_id == "acute_aortic_syndrome_algorithm" and any(
                token in combined_text for token in ["主动脉夹层", "撕裂样", "背痛", "血压差"]
            ):
                matched.append(algorithm)
            if algorithm_id == "post_cardiac_surgery_crash_algorithm" and any(
                token in combined_text for token in ["心外术后", "低心排", "引流异常", "填塞"]
            ):
                matched.append(algorithm)

        return matched

    def _combined_text(
        self,
        normalized_patient: dict[str, Any],
        entity_linking: dict[str, Any] | None,
    ) -> str:
        entity_linking = self._normalize_entity_linking(entity_linking)
        return " ".join(
            [
                str(normalized_patient.get("chief_complaint") or ""),
                " ".join(normalized_patient.get("symptoms", [])),
                str(normalized_patient.get("diagnosis_hint") or ""),
                str(normalized_patient.get("procedure_name") or ""),
                str(normalized_patient.get("procedure_site") or ""),
                str(normalized_patient.get("urgency") or ""),
                str(normalized_patient.get("ecg_summary") or ""),
                str(normalized_patient.get("echo_summary") or ""),
                str(normalized_patient.get("imaging_summary") or ""),
                str(normalized_patient.get("bleeding_status") or ""),
                str(normalized_patient.get("postop_context") or ""),
                " ".join(normalized_patient.get("comorbidities", [])),
                " ".join(normalized_patient.get("medications", [])),
                " ".join(f"{k}:{v}" for k, v in normalized_patient.get("labs", {}).items()),
                " ".join(f"{k}:{v}" for k, v in normalized_patient.get("vitals", {}).items()),
                " ".join((entity_linking or {}).get("normalized_terms", [])),
                " ".join((entity_linking or {}).get("syndrome_keywords", [])),
                " ".join((entity_linking or {}).get("symptom_keywords", [])),
                " ".join((entity_linking or {}).get("ecg_keywords", [])),
                " ".join((entity_linking or {}).get("imaging_keywords", [])),
                " ".join((entity_linking or {}).get("procedure_keywords", [])),
                " ".join((entity_linking or {}).get("risk_keywords", [])),
                " ".join((entity_linking or {}).get("medication_keywords", [])),
            ]
        ).lower()

    def _normalize_entity_linking(
        self,
        entity_linking: dict[str, Any] | None,
    ) -> dict[str, list[str]]:
        payload = entity_linking or {}
        keys = [
            "normalized_terms",
            "syndrome_keywords",
            "symptom_keywords",
            "ecg_keywords",
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
