from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from patient_mapper_agent import (
    AnesthesiaKnowledgeBase,
    GenAIChatClient,
    MODEL_CONFIGS,
    TASK_MODEL_SELECTION,
)


class KnowledgeRetriever:
    def __init__(
        self,
        kb: AnesthesiaKnowledgeBase,
        api_client: GenAIChatClient | None = None,
    ) -> None:
        self.kb = kb
        self.api_client = api_client or GenAIChatClient()
        self._graph_settings = self._load_graph_settings()
        self._graph, self._node_types = self._build_knowledge_graph()

    def retrieve(
        self,
        normalized_patient: dict[str, Any],
        entity_linking: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        entity_linking = self._normalize_entity_linking(entity_linking)
        terminology_hits = self._collect_terminology_hits(normalized_patient, entity_linking)
        matched_templates = self._match_procedure_templates(terminology_hits)
        matched_comorbidity_rules = self._match_comorbidity_rules(normalized_patient)
        matched_contraindications = self._match_contraindications(normalized_patient, entity_linking)
        asa_match_id = self._map_asa_hint(normalized_patient.get("asa_hint"))
        candidate_plan_ids = self._infer_candidate_plan_ids(
            normalized_patient,
            entity_linking,
            terminology_hits,
            matched_templates,
        )
        safety_rule_ids = self._infer_safety_rule_ids(
            normalized_patient,
            entity_linking,
            terminology_hits,
            candidate_plan_ids,
        )
        graph_ranking = self._graph_rank_entities(
            candidate_plan_ids=candidate_plan_ids,
            safety_rule_ids=safety_rule_ids,
            matched_templates=matched_templates,
            matched_comorbidity_rules=matched_comorbidity_rules,
            matched_contraindications=matched_contraindications,
        )

        plan_scores = graph_ranking.get("plan_scores", {})
        ranked_plan_ids = graph_ranking.get("ranked_plan_ids", candidate_plan_ids)
        candidate_plans = [
            self.kb.get_plan(plan_id)
            for plan_id in ranked_plan_ids
            if self.kb.get_plan(plan_id) is not None
        ]

        ranked_rule_ids = graph_ranking.get("ranked_rule_ids", safety_rule_ids)
        matched_rules = [
            rule
            for rule in self.kb.list_safety_rules()
            if rule["id"] in ranked_rule_ids
        ]
        matched_rules.sort(
            key=lambda item: graph_ranking.get("rule_scores", {}).get(item.get("id", ""), 0.0),
            reverse=True,
        )

        ranked_drug_ids = graph_ranking.get("ranked_drug_ids", [])
        related_drugs = self._collect_related_drugs_from_ranked_ids(ranked_drug_ids)
        if not related_drugs:
            related_drugs = self._collect_related_drugs(candidate_plans)

        return {
            "asa_match": self.kb.get_asa(asa_match_id) if asa_match_id else None,
            "candidate_plans": candidate_plans,
            "candidate_plan_scores": plan_scores,
            "matched_safety_rules": matched_rules,
            "related_drugs": related_drugs,
            "matched_procedure_templates": matched_templates,
            "matched_comorbidity_rules": matched_comorbidity_rules,
            "matched_contraindications": matched_contraindications,
            "terminology_hits": terminology_hits,
            "retrieval_notes": self._build_retrieval_notes(
                candidate_plans,
                matched_rules,
                matched_templates,
                matched_comorbidity_rules,
                matched_contraindications,
            ),
            "missing_information": self._identify_missing_information(normalized_patient),
            "graph_context": graph_ranking.get("graph_context", {}),
        }

    def retrieve_with_api(
        self,
        normalized_patient: dict[str, Any],
        entity_linking: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        prompt = (
            "You are an anesthesia knowledge-retrieval agent.\n"
            "Review the normalized anesthesia patient profile, optional entity-linking output, and the anesthesia KB.\n"
            "Return JSON only with keys:\n"
            "asa_match, candidate_plans, matched_safety_rules, related_drugs, matched_procedure_templates, "
            "matched_comorbidity_rules, matched_contraindications, terminology_hits, retrieval_notes, missing_information, "
            "candidate_plan_scores, graph_context.\n"
            "Schema rules:\n"
            "- asa_match: null or one ASA object from the KB.\n"
            "- candidate_plans, matched_safety_rules, related_drugs, matched_procedure_templates, matched_comorbidity_rules, matched_contraindications: arrays of KB objects.\n"
            "- terminology_hits: object whose values are arrays of strings.\n"
            "- candidate_plan_scores: object mapping plan id to numeric score.\n"
            "- graph_context: object summary of seed nodes and ranked ids.\n"
            "- retrieval_notes and missing_information: arrays of strings.\n"
            "- Prefer exact KB ids and wording when available.\n\n"
            "Normalized patient:\n"
            f"{json.dumps(normalized_patient, ensure_ascii=False, indent=2)}\n\n"
            "Entity linking:\n"
            f"{json.dumps(entity_linking or {}, ensure_ascii=False, indent=2)}\n\n"
            "Anesthesia KB:\n"
            f"{json.dumps(self.kb.data, ensure_ascii=False, indent=2)}"
        )
        config = MODEL_CONFIGS[TASK_MODEL_SELECTION["plan_ranking"]]
        response = self.api_client.chat_json(config, prompt)
        payload = self.api_client.extract_json(response)
        payload.setdefault("candidate_plans", [])
        payload.setdefault("matched_safety_rules", [])
        payload.setdefault("related_drugs", [])
        payload.setdefault("matched_procedure_templates", [])
        payload.setdefault("matched_comorbidity_rules", [])
        payload.setdefault("matched_contraindications", [])
        payload.setdefault("terminology_hits", {})
        payload.setdefault("retrieval_notes", [])
        payload.setdefault("missing_information", [])
        payload.setdefault("candidate_plan_scores", {})
        payload.setdefault("graph_context", {})
        return payload

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
        terminology_hits: dict[str, list[str]],
        matched_templates: list[dict[str, Any]],
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
        site_hits = " ".join(terminology_hits.get("procedure_sites", []))
        procedure_hits = " ".join(terminology_hits.get("procedure_terms", []))
        airway_hits = " ".join(terminology_hits.get("airway_risk_terms", []))

        plan_ids: list[str] = []

        for template in matched_templates:
            plan_ids.extend(template.get("common_candidate_plans", []))

        if any(token in procedure_text + linked_keywords + site_hits for token in ["下肢", "下腹", "会阴", "lower_limb", "lower_abdomen", "perineum"]):
            plan_ids.append("spinal_anesthesia")

        if any(token in procedure_text + linked_keywords + procedure_hits for token in ["短小", "表浅", "内镜", "局麻", "endoscopy", "superficial_minor"]):
            plan_ids.append("monitored_anesthesia_care")

        if any(
            token in fasting_text + airway_text + risk_keywords + airway_hits
            for token in ["未禁食", "饱胃", "误吸", "困难气道", "full_stomach", "aspiration", "difficult_airway"]
        ):
            plan_ids.append("general_anesthesia_ett")
            plan_ids.append("rapid_sequence_induction")

        if any(token in linked_keywords for token in ["orthopedic trauma", "fracture fixation", "open reduction internal fixation", "ORIF"]):
            plan_ids.append("balanced_general_anesthesia")

        if any(
            token in procedure_text + linked_keywords + site_hits + procedure_hits
            for token in ["腹腔镜", "laparoscopy", "upper_abdomen", "thorax", "laparoscopic"]
        ):
            plan_ids.append("general_anesthesia_ett")
            plan_ids.append("balanced_general_anesthesia")

        if any(token in procedure_text + linked_keywords + procedure_hits for token in ["剖宫产", "cesarean", "cesarean_delivery"]):
            plan_ids.append("spinal_anesthesia")
            plan_ids.append("general_anesthesia_ett")

        if not plan_ids:
            plan_ids.append("general_anesthesia_ett")

        return list(dict.fromkeys(plan_ids))

    def _infer_safety_rule_ids(
        self,
        normalized_patient: dict[str, Any],
        entity_linking: dict[str, Any] | None,
        terminology_hits: dict[str, list[str]],
        candidate_plan_ids: list[str],
    ) -> list[str]:
        comorbidity_text = " ".join(normalized_patient.get("comorbidities", []))
        medication_text = " ".join(normalized_patient.get("medications", []))
        fasting_text = str(normalized_patient.get("fasting_status") or "")
        airway_text = str(normalized_patient.get("airway_notes") or "")
        age = normalized_patient.get("age")
        linked_risks = " ".join((entity_linking or {}).get("risk_keywords", []))
        airway_hits = " ".join(terminology_hits.get("airway_risk_terms", []))
        rule_ids: list[str] = []

        if any(
            token in comorbidity_text + linked_risks
            for token in ["恶性高热", "malignant hyperthermia", "MH"]
        ):
            rule_ids.append("mh_trigger_avoidance")

        if any(
            token in fasting_text + linked_risks + airway_hits
            for token in ["未禁食", "饱胃", "胃排空延迟", "肠梗阻", "full_stomach", "aspiration"]
        ):
            rule_ids.append("aspiration_risk_review")

        if any(
            token in comorbidity_text + linked_risks + airway_text
            for token in ["OSA", "sleep apnea", "obstructive sleep apnea", "阻塞性睡眠呼吸暂停"]
        ):
            rule_ids.append("osa_monitoring_review")

        if any(
            token in medication_text.lower() + fasting_text.lower() + linked_risks.lower()
            for token in ["semaglutide", "tirzepatide", "liraglutide", "dulaglutide", "glp-1", "glp1"]
        ):
            rule_ids.append("glp1_delayed_gastric_emptying_review")

        if "spinal_anesthesia" in candidate_plan_ids and any(
            token in medication_text.lower() + comorbidity_text.lower() + linked_risks.lower()
            for token in [
                "warfarin",
                "heparin",
                "enoxaparin",
                "apixaban",
                "rivaroxaban",
                "dabigatran",
                "clopidogrel",
                "抗凝",
                "抗血小板",
            ]
        ):
            rule_ids.append("neuraxial_antithrombotic_review")

        if any(
            token in airway_text + linked_risks
            for token in ["困难气道", "limited mouth opening", "neck immobility", "小下颌", "张口受限", "颈项活动受限"]
        ):
            rule_ids.append("difficult_airway_strategy_review")

        if "spinal_anesthesia" in candidate_plan_ids:
            rule_ids.append("spinal_hypotension_prepare")

        if "monitored_anesthesia_care" in candidate_plan_ids:
            rule_ids.append("mac_requires_rescue_capability")

        if any(token in linked_risks for token in ["diabetes mellitus", "perioperative glycemic risk", "ponv", "motion sickness"]):
            rule_ids.append("ponv_prophylaxis_consideration")

        if isinstance(age, (int, float)) and age >= 65:
            rule_ids.append("older_adult_perioperative_review")

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

    def _collect_related_drugs_from_ranked_ids(self, ranked_drug_ids: list[str]) -> list[dict[str, Any]]:
        related_drugs: list[dict[str, Any]] = []
        for drug_id in ranked_drug_ids:
            drug = self.kb.get_drug(drug_id)
            if drug is not None:
                related_drugs.append(drug)
        return related_drugs

    def _load_graph_settings(self) -> dict[str, Any]:
        graph_cfg = self.kb.data.get("knowledge_graph", {})
        retrieval_cfg = graph_cfg.get("retrieval", {})
        edge_weights = graph_cfg.get("edge_weights", {})
        return {
            "enabled": bool(graph_cfg.get("enabled", True)),
            "max_hops": int(retrieval_cfg.get("max_hops", 2)),
            "decay": float(retrieval_cfg.get("decay", 0.72)),
            "min_edge_weight": float(retrieval_cfg.get("min_edge_weight", 0.2)),
            "max_candidate_plans": int(retrieval_cfg.get("max_candidate_plans", 5)),
            "max_related_drugs": int(retrieval_cfg.get("max_related_drugs", 8)),
            "plan_expand_threshold": float(retrieval_cfg.get("plan_expand_threshold", 0.45)),
            "rule_expand_threshold": float(retrieval_cfg.get("rule_expand_threshold", 0.35)),
            "drug_expand_threshold": float(retrieval_cfg.get("drug_expand_threshold", 0.35)),
            "seed_weights": {
                "template": float(retrieval_cfg.get("template_seed_weight", 1.0)),
                "plan": float(retrieval_cfg.get("plan_seed_weight", 1.0)),
                "rule": float(retrieval_cfg.get("rule_seed_weight", 0.95)),
                "contra": float(retrieval_cfg.get("contra_seed_weight", 0.85)),
                "comorbidity": float(retrieval_cfg.get("comorbidity_seed_weight", 0.8)),
            },
            "edge_weights": {
                "template_to_plan": float(edge_weights.get("template_to_plan", 0.95)),
                "plan_to_drug": float(edge_weights.get("plan_to_drug", 0.9)),
                "rule_to_plan": float(edge_weights.get("rule_to_plan", 0.75)),
                "rule_to_drug": float(edge_weights.get("rule_to_drug", 0.8)),
                "contra_to_drug": float(edge_weights.get("contra_to_drug", 0.9)),
                "comorbidity_to_rule": float(edge_weights.get("comorbidity_to_rule", 0.75)),
            },
        }

    def _build_knowledge_graph(self) -> tuple[dict[str, list[tuple[str, float, str]]], dict[str, str]]:
        graph: dict[str, list[tuple[str, float, str]]] = defaultdict(list)
        node_types: dict[str, str] = {}
        edge_weights = self._graph_settings["edge_weights"]

        def node_key(node_type: str, node_id: str) -> str:
            return f"{node_type}:{node_id}"

        def add_edge(
            left_type: str,
            left_id: str,
            right_type: str,
            right_id: str,
            weight: float,
            relation: str,
        ) -> None:
            if not left_id or not right_id:
                return
            left = node_key(left_type, left_id)
            right = node_key(right_type, right_id)
            node_types[left] = left_type
            node_types[right] = right_type
            graph[left].append((right, weight, relation))
            graph[right].append((left, weight, relation))

        for template in self.kb.data.get("procedure_templates", []):
            template_id = str(template.get("id") or "")
            for plan_id in template.get("common_candidate_plans", []):
                add_edge(
                    "template",
                    template_id,
                    "plan",
                    str(plan_id),
                    edge_weights["template_to_plan"],
                    "template_recommends_plan",
                )

        for plan in self.kb.data.get("anesthesia_plans", []):
            plan_id = str(plan.get("id") or "")
            candidate_drugs = plan.get("candidate_drugs", {})
            for drug_ids in candidate_drugs.values():
                for drug_id in drug_ids:
                    add_edge(
                        "plan",
                        plan_id,
                        "drug",
                        str(drug_id),
                        edge_weights["plan_to_drug"],
                        "plan_uses_drug",
                    )

        for rule in self.kb.list_safety_rules():
            rule_id = str(rule.get("id") or "")
            for plan_id in rule.get("if_plan_selected", []):
                add_edge(
                    "rule",
                    rule_id,
                    "plan",
                    str(plan_id),
                    edge_weights["rule_to_plan"],
                    "rule_targets_plan",
                )
            for drug_id in rule.get("if_drugs_used", []):
                add_edge(
                    "rule",
                    rule_id,
                    "drug",
                    str(drug_id),
                    edge_weights["rule_to_drug"],
                    "rule_targets_drug",
                )
            for drug_id in rule.get("avoid_drugs", []):
                add_edge(
                    "rule",
                    rule_id,
                    "drug",
                    str(drug_id),
                    edge_weights["rule_to_drug"],
                    "rule_avoids_drug",
                )

        for contra in self.kb.data.get("contraindication_catalog", []):
            contra_id = str(contra.get("id") or "")
            for drug_id in contra.get("related_drugs", []):
                add_edge(
                    "contra",
                    contra_id,
                    "drug",
                    str(drug_id),
                    edge_weights["contra_to_drug"],
                    "contraindication_related_drug",
                )

        safety_rules = self.kb.list_safety_rules()
        for comorbidity_rule in self.kb.data.get("comorbidity_rules", []):
            comorbidity_id = str(comorbidity_rule.get("id") or "")
            trigger_terms = {str(item).lower() for item in comorbidity_rule.get("trigger_terms", [])}
            for safety_rule in safety_rules:
                rule_id = str(safety_rule.get("id") or "")
                safety_text = " ".join(
                    [
                        " ".join(safety_rule.get("if_patient_has", [])),
                        str(safety_rule.get("name_zh") or ""),
                        str(safety_rule.get("action") or ""),
                    ]
                ).lower()
                if any(term and term in safety_text for term in trigger_terms):
                    add_edge(
                        "comorbidity",
                        comorbidity_id,
                        "rule",
                        rule_id,
                        edge_weights["comorbidity_to_rule"],
                        "comorbidity_related_rule",
                    )

        return dict(graph), node_types

    def _graph_rank_entities(
        self,
        candidate_plan_ids: list[str],
        safety_rule_ids: list[str],
        matched_templates: list[dict[str, Any]],
        matched_comorbidity_rules: list[dict[str, Any]],
        matched_contraindications: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not self._graph_settings["enabled"]:
            return {
                "ranked_plan_ids": candidate_plan_ids,
                "ranked_rule_ids": safety_rule_ids,
                "ranked_drug_ids": [],
                "plan_scores": {},
                "rule_scores": {},
                "graph_context": {"enabled": False},
            }

        seed_nodes: dict[str, float] = {}
        seed_weights = self._graph_settings["seed_weights"]

        for template in matched_templates:
            template_id = str(template.get("id") or "")
            if template_id:
                seed_nodes[f"template:{template_id}"] = max(
                    seed_nodes.get(f"template:{template_id}", 0.0),
                    seed_weights["template"],
                )
        for plan_id in candidate_plan_ids:
            if plan_id:
                seed_nodes[f"plan:{plan_id}"] = max(
                    seed_nodes.get(f"plan:{plan_id}", 0.0),
                    seed_weights["plan"],
                )
        for rule_id in safety_rule_ids:
            if rule_id:
                seed_nodes[f"rule:{rule_id}"] = max(
                    seed_nodes.get(f"rule:{rule_id}", 0.0),
                    seed_weights["rule"],
                )
        for contra in matched_contraindications:
            contra_id = str(contra.get("id") or "")
            if contra_id:
                seed_nodes[f"contra:{contra_id}"] = max(
                    seed_nodes.get(f"contra:{contra_id}", 0.0),
                    seed_weights["contra"],
                )
        for comorbidity_rule in matched_comorbidity_rules:
            comorbidity_id = str(comorbidity_rule.get("id") or "")
            if comorbidity_id:
                seed_nodes[f"comorbidity:{comorbidity_id}"] = max(
                    seed_nodes.get(f"comorbidity:{comorbidity_id}", 0.0),
                    seed_weights["comorbidity"],
                )

        node_scores: dict[str, float] = defaultdict(float)
        frontier: dict[str, float] = {}
        for node, weight in seed_nodes.items():
            if node in self._node_types:
                node_scores[node] += weight
                frontier[node] = max(frontier.get(node, 0.0), weight)

        decay = self._graph_settings["decay"]
        min_edge_weight = self._graph_settings["min_edge_weight"]
        max_hops = self._graph_settings["max_hops"]

        for _ in range(max_hops):
            next_frontier: dict[str, float] = {}
            for source, source_weight in frontier.items():
                for target, edge_weight, _relation in self._graph.get(source, []):
                    if edge_weight < min_edge_weight:
                        continue
                    propagated = source_weight * edge_weight * decay
                    if propagated < 1e-6:
                        continue
                    node_scores[target] += propagated
                    next_frontier[target] = max(next_frontier.get(target, 0.0), propagated)
            frontier = next_frontier
            if not frontier:
                break

        plan_scores = self._extract_type_scores(node_scores, "plan")
        rule_scores = self._extract_type_scores(node_scores, "rule")
        drug_scores = self._extract_type_scores(node_scores, "drug")

        ranked_plan_ids = self._rank_ids_by_score(
            ids=plan_scores,
            min_threshold=self._graph_settings["plan_expand_threshold"],
            limit=self._graph_settings["max_candidate_plans"],
            fallback_ids=candidate_plan_ids,
        )
        ranked_rule_ids = self._rank_ids_by_score(
            ids=rule_scores,
            min_threshold=self._graph_settings["rule_expand_threshold"],
            limit=max(len(safety_rule_ids), 6),
            fallback_ids=safety_rule_ids,
        )
        ranked_drug_ids = self._rank_ids_by_score(
            ids=drug_scores,
            min_threshold=self._graph_settings["drug_expand_threshold"],
            limit=self._graph_settings["max_related_drugs"],
            fallback_ids=[],
        )

        return {
            "ranked_plan_ids": ranked_plan_ids,
            "ranked_rule_ids": ranked_rule_ids,
            "ranked_drug_ids": ranked_drug_ids,
            "plan_scores": plan_scores,
            "rule_scores": rule_scores,
            "graph_context": {
                "enabled": True,
                "seed_nodes": sorted(seed_nodes.keys()),
                "top_plan_ids": ranked_plan_ids,
                "top_rule_ids": ranked_rule_ids,
                "top_drug_ids": ranked_drug_ids,
            },
        }

    def _extract_type_scores(self, node_scores: dict[str, float], node_type: str) -> dict[str, float]:
        prefix = f"{node_type}:"
        result: dict[str, float] = {}
        for node_key, score in node_scores.items():
            if not node_key.startswith(prefix):
                continue
            result[node_key.removeprefix(prefix)] = round(float(score), 6)
        return result

    def _rank_ids_by_score(
        self,
        ids: dict[str, float],
        min_threshold: float,
        limit: int,
        fallback_ids: list[str],
    ) -> list[str]:
        ranked = [
            item_id
            for item_id, score in sorted(ids.items(), key=lambda item: item[1], reverse=True)
            if score >= min_threshold
        ]
        for fallback_id in fallback_ids:
            if fallback_id and fallback_id not in ranked:
                ranked.append(fallback_id)
        return ranked[: max(limit, 1)]

    def _build_retrieval_notes(
        self,
        candidate_plans: list[dict[str, Any]],
        matched_rules: list[dict[str, Any]],
        matched_templates: list[dict[str, Any]],
        matched_comorbidity_rules: list[dict[str, Any]],
        matched_contraindications: list[dict[str, Any]],
    ) -> list[str]:
        notes: list[str] = []
        if candidate_plans:
            notes.append(f"已召回 {len(candidate_plans)} 个候选方案。")
        if matched_rules:
            notes.append(f"已命中 {len(matched_rules)} 条安全规则。")
        if matched_templates:
            notes.append(f"已命中 {len(matched_templates)} 个术式模板。")
        if matched_comorbidity_rules:
            notes.append(f"已命中 {len(matched_comorbidity_rules)} 条合并症规则。")
        if matched_contraindications:
            notes.append(f"已匹配 {len(matched_contraindications)} 条禁忌/高风险目录项。")
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

    def _collect_terminology_hits(
        self,
        normalized_patient: dict[str, Any],
        entity_linking: dict[str, Any] | None,
    ) -> dict[str, list[str]]:
        kb_aliases = self.kb.data.get("terminology_aliases", {})
        combined_text = " ".join(
            [
                str(normalized_patient.get("procedure_name") or ""),
                str(normalized_patient.get("procedure_site") or ""),
                str(normalized_patient.get("fasting_status") or ""),
                str(normalized_patient.get("airway_notes") or ""),
                " ".join(normalized_patient.get("comorbidities", [])),
                " ".join(normalized_patient.get("medications", [])),
                " ".join((entity_linking or {}).get("procedure_keywords", [])),
                " ".join((entity_linking or {}).get("risk_keywords", [])),
                " ".join((entity_linking or {}).get("medication_keywords", [])),
            ]
        ).lower()

        hits: dict[str, list[str]] = {}
        for group_name, group_values in kb_aliases.items():
            group_hits: list[str] = []
            for canonical_name, aliases in group_values.items():
                if any(str(alias).lower() in combined_text for alias in aliases):
                    group_hits.append(canonical_name)
            hits[group_name] = group_hits
        return hits

    def _normalize_entity_linking(
        self,
        entity_linking: dict[str, Any] | None,
    ) -> dict[str, list[str]]:
        payload = entity_linking or {}
        keys = [
            "normalized_terms",
            "asa_interpretation",
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
        text = str(value).strip()
        return [text] if text else []

    def _match_procedure_templates(
        self,
        terminology_hits: dict[str, list[str]],
    ) -> list[dict[str, Any]]:
        templates = self.kb.data.get("procedure_templates", [])
        matched: list[dict[str, Any]] = []
        procedure_hits = set(terminology_hits.get("procedure_terms", []))
        site_hits = set(terminology_hits.get("procedure_sites", []))

        for template in templates:
            template_id = template["id"]
            if template_id == "orthopedic_lower_limb_template" and "lower_limb" in site_hits:
                matched.append(template)
            if template_id == "endoscopy_template" and "endoscopy" in procedure_hits:
                matched.append(template)
            if template_id == "laparoscopic_abdominal_template" and "laparoscopy" in procedure_hits:
                matched.append(template)
            if template_id == "cesarean_delivery_template" and "cesarean" in procedure_hits:
                matched.append(template)
        return matched

    def _match_comorbidity_rules(self, normalized_patient: dict[str, Any]) -> list[dict[str, Any]]:
        rules = self.kb.data.get("comorbidity_rules", [])
        combined_text = " ".join(
            normalized_patient.get("comorbidities", []) + normalized_patient.get("medications", [])
        ).lower()
        matched: list[dict[str, Any]] = []

        for rule in rules:
            if any(term.lower() in combined_text for term in rule.get("trigger_terms", [])):
                matched.append(rule)
        return matched

    def _match_contraindications(
        self,
        normalized_patient: dict[str, Any],
        entity_linking: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        contraindications = self.kb.data.get("contraindication_catalog", [])
        combined_text = " ".join(
            normalized_patient.get("comorbidities", [])
            + normalized_patient.get("medications", [])
            + (entity_linking or {}).get("risk_keywords", [])
        ).lower()
        matched: list[dict[str, Any]] = []

        for item in contraindications:
            item_id = item["id"]
            if item_id == "contra_mh" and any(token in combined_text for token in ["恶性高热", "malignant hyperthermia", "mh"]):
                matched.append(item)
            if item_id == "contra_hyperkalemia_risk" and any(
                token in combined_text for token in ["高钾", "burn", "trauma", "失神经", "upper motor neuron"]
            ):
                matched.append(item)
            if item_id == "contra_local_anesthetic_hypersensitivity" and any(
                token.lower() in combined_text for token in ["局麻药过敏", "bupivacaine", "amide allergy"]
            ):
                matched.append(item)
            if item_id == "contra_neuraxial_anticoagulation_risk" and any(
                token in combined_text
                for token in ["warfarin", "heparin", "enoxaparin", "apixaban", "rivaroxaban", "dabigatran", "clopidogrel", "抗凝", "抗血小板"]
            ):
                matched.append(item)
        return matched
