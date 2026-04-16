from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from neurosurgery_patient_mapper_agent import (
    NEUROSURGERY_TASK_MODEL_SELECTION,
    NeurosurgeryKnowledgeBase,
)
from neurosurgery_shared import GenAIChatClient, MODEL_CONFIGS


class NeurosurgeryKnowledgeRetriever:
    def __init__(
        self,
        kb: NeurosurgeryKnowledgeBase,
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
        text = self._build_search_text(normalized_patient, entity_linking)
        syndrome_scores = self._score_syndromes(text)
        red_flags = self._match_red_flags(text)
        risk_rules = self._match_risk_rules(text, normalized_patient)
        graph_ranking = self._graph_rank_entities(
            syndrome_scores=syndrome_scores,
            red_flags=red_flags,
            risk_rules=risk_rules,
        )
        candidate_syndromes = self._build_candidate_syndromes(
            syndrome_scores,
            graph_ranking.get("syndrome_scores", {}),
        )
        selected_workups = self._select_workups(text, candidate_syndromes)
        selected_workups = self._rerank_items_by_graph_score(
            selected_workups,
            graph_ranking.get("workup_scores", {}),
            item_id_key="id",
        )
        complication_watchlist = self._match_complications(text, risk_rules)
        complication_watchlist = self._rerank_items_by_graph_score(
            complication_watchlist,
            graph_ranking.get("complication_scores", {}),
            item_id_key="id",
        )

        return {
            "candidate_syndromes": candidate_syndromes,
            "candidate_syndrome_scores": graph_ranking.get("syndrome_scores", {}),
            "triggered_red_flags": red_flags,
            "triggered_risk_rules": risk_rules,
            "urgency_level": self._estimate_urgency(candidate_syndromes, red_flags, risk_rules),
            "differential_considerations": self._match_differentials(text),
            "recommended_workup": selected_workups,
            "perioperative_focus": self._build_perioperative_focus(risk_rules, normalized_patient),
            "complication_watchlist": complication_watchlist,
            "external_tools": self.kb.list_external_tools(),
            "missing_information": self._identify_missing_information(normalized_patient),
            "retrieval_notes": self._build_retrieval_notes(candidate_syndromes, red_flags, risk_rules),
            "graph_context": graph_ranking.get("graph_context", {}),
        }

    def retrieve_with_api(
        self,
        normalized_patient: dict[str, Any],
        entity_linking: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        prompt = (
            "You are a neurosurgery knowledge-retrieval and triage agent.\n"
            "Review the normalized patient case, optional entity-linking hints, and the neurosurgery KB.\n"
            "Return JSON only with keys:\n"
            "candidate_syndromes, triggered_red_flags, triggered_risk_rules, urgency_level, "
            "differential_considerations, recommended_workup, perioperative_focus, "
            "complication_watchlist, external_tools, missing_information, retrieval_notes, "
            "candidate_syndrome_scores, graph_context.\n"
            "Schema rules:\n"
            "- candidate_syndromes: array of objects with id, name_zh, score, default_urgency, matched_keywords, focus_points, recommended_collaborators.\n"
            "- triggered_red_flags: array of objects with id, name_zh, severity, matched_keywords, rationale_zh, immediate_actions.\n"
            "- triggered_risk_rules: array of objects with id, name_zh, severity, matched_keywords, perioperative_focus, rationale_zh.\n"
            "- urgency_level: one of immediate_emergency, urgent_inpatient_review, expedited_specialist_workup, needs_more_information.\n"
            "- differential_considerations: array of objects with symptom_cluster, matched_keywords, possible_causes, related_tests.\n"
            "- recommended_workup: array of KB workup objects.\n"
            "- perioperative_focus, missing_information, retrieval_notes: arrays of strings.\n"
            "- candidate_syndrome_scores: object mapping syndrome id to numeric score.\n"
            "- graph_context: object summary with seed_nodes and ranked ids.\n"
            "- complication_watchlist: array of objects with id, name_zh, complications, reason.\n"
            "- external_tools: array of KB tool objects.\n"
            "- Prefer KB ids and KB wording when available.\n\n"
            "Normalized patient:\n"
            f"{json.dumps(normalized_patient, ensure_ascii=False, indent=2)}\n\n"
            "Entity linking:\n"
            f"{json.dumps(entity_linking or {}, ensure_ascii=False, indent=2)}\n\n"
            "Neurosurgery KB:\n"
            f"{json.dumps(self.kb.data, ensure_ascii=False, indent=2)}"
        )
        config = MODEL_CONFIGS[NEUROSURGERY_TASK_MODEL_SELECTION["retrieval_reasoning"]]
        response = self.api_client.chat_json(config, prompt)
        payload = self.api_client.extract_json(response)
        return self._normalize_api_retrieval_result(payload)

    def _build_search_text(
        self,
        normalized_patient: dict[str, Any],
        entity_linking: dict[str, Any] | None,
    ) -> str:
        segments: list[str] = []
        for key in [
            "chief_complaint",
            "presenting_illness",
            "planned_procedure",
            "procedure_site",
            "functional_status",
            "seizure_history",
            "trauma_history",
            "asa_hint",
        ]:
            value = normalized_patient.get(key)
            if value:
                segments.append(str(value))

        for key in [
            "suspected_diagnoses",
            "symptoms",
            "neurological_exam",
            "comorbidities",
            "past_history",
            "surgical_history",
            "family_history",
            "medication_history",
            "allergies",
            "headache_features",
            "visual_symptoms",
            "cognitive_symptoms",
            "oncology_history",
            "vascular_history",
            "device_history",
        ]:
            segments.extend([str(item) for item in normalized_patient.get(key, [])])

        for key in ["labs", "vitals", "imaging", "pathology"]:
            value = normalized_patient.get(key, {})
            if value:
                segments.append(json.dumps(value, ensure_ascii=False))

        if entity_linking:
            for value in entity_linking.values():
                if isinstance(value, list):
                    segments.extend([str(item) for item in value])
                elif value:
                    segments.append(str(value))

        return " ".join(segments).lower()

    def _score_syndromes(self, text: str) -> list[dict[str, Any]]:
        scored: list[dict[str, Any]] = []
        for syndrome in self.kb.list_syndromes():
            matched_keywords = [
                keyword
                for keyword in syndrome.get("trigger_keywords", [])
                if self._keyword_present(text, keyword.lower())
            ]
            if not matched_keywords:
                continue
            scored.append(
                {
                    "syndrome": syndrome,
                    "score": len(dict.fromkeys(matched_keywords)),
                    "matched_keywords": list(dict.fromkeys(matched_keywords)),
                }
            )
        return scored

    def _build_candidate_syndromes(
        self,
        scored_syndromes: list[dict[str, Any]],
        graph_syndrome_scores: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        graph_syndrome_scores = graph_syndrome_scores or {}
        enriched: list[dict[str, Any]] = []
        for item in scored_syndromes:
            syndrome = item["syndrome"]
            syndrome_id = syndrome["id"]
            graph_bonus = float(graph_syndrome_scores.get(syndrome_id, 0.0))
            enriched.append(
                {
                    "syndrome": syndrome,
                    "score": float(item["score"]) + graph_bonus,
                    "matched_keywords": item["matched_keywords"],
                }
            )
        candidates = sorted(
            enriched,
            key=lambda item: (-item["score"], item["syndrome"]["name_zh"]),
        )
        result: list[dict[str, Any]] = []
        for item in candidates:
            syndrome = item["syndrome"]
            result.append(
                {
                    "id": syndrome["id"],
                    "name_zh": syndrome["name_zh"],
                    "score": round(float(item["score"]), 6),
                    "default_urgency": syndrome["default_urgency"],
                    "matched_keywords": item["matched_keywords"],
                    "focus_points": syndrome.get("focus_points", []),
                    "recommended_collaborators": syndrome.get("recommended_collaborators", []),
                }
            )
        return result

    def _match_red_flags(self, text: str) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        for rule in self.kb.list_red_flags():
            matched_keywords = [
                keyword
                for keyword in rule.get("if_any_keywords", [])
                if self._keyword_present(text, keyword.lower())
            ]
            if not matched_keywords:
                continue
            matches.append(
                {
                    "id": rule["id"],
                    "name_zh": rule["name_zh"],
                    "severity": rule["severity"],
                    "matched_keywords": list(dict.fromkeys(matched_keywords)),
                    "rationale_zh": rule["rationale_zh"],
                    "immediate_actions": rule.get("immediate_actions", []),
                }
            )
        return matches

    def _match_risk_rules(
        self,
        text: str,
        normalized_patient: dict[str, Any],
    ) -> list[dict[str, Any]]:
        asa_grade = self._extract_asa_grade(normalized_patient.get("asa_hint"))
        age = normalized_patient.get("age")
        matches: list[dict[str, Any]] = []
        for rule in self.kb.list_risk_rules():
            matched_keywords = [
                keyword
                for keyword in rule.get("if_any_keywords", [])
                if self._keyword_present(text, keyword.lower())
            ]
            minimum_matches = int(rule.get("minimum_matches", 0))
            if len(matched_keywords) < minimum_matches:
                continue
            if rule.get("min_age") is not None and not self._at_least(age, rule["min_age"]):
                continue
            if rule.get("asa_at_least") is not None and not self._at_least(
                asa_grade,
                rule["asa_at_least"],
            ):
                continue
            matches.append(
                {
                    "id": rule["id"],
                    "name_zh": rule["name_zh"],
                    "severity": rule["severity"],
                    "matched_keywords": list(dict.fromkeys(matched_keywords)),
                    "perioperative_focus": rule.get("perioperative_focus", []),
                    "rationale_zh": rule["rationale_zh"],
                }
            )
        return matches

    def _estimate_urgency(
        self,
        candidate_syndromes: list[dict[str, Any]],
        red_flags: list[dict[str, Any]],
        risk_rules: list[dict[str, Any]],
    ) -> str:
        if any(rule["severity"] == "high" for rule in red_flags):
            return "immediate_emergency"
        if any(item["default_urgency"] == "urgent_inpatient_review" for item in candidate_syndromes):
            return "urgent_inpatient_review"
        if any(rule["severity"] == "high" for rule in risk_rules):
            return "urgent_inpatient_review"
        if candidate_syndromes:
            return "expedited_specialist_workup"
        return "needs_more_information"

    def _match_differentials(self, text: str) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        for template in self.kb.list_differentials():
            matched_keywords = [
                keyword
                for keyword in template.get("if_any_keywords", [])
                if self._keyword_present(text, keyword.lower())
            ]
            if len(matched_keywords) < int(template.get("minimum_matches", 1)):
                continue
            matches.append(
                {
                    "symptom_cluster": template["symptom_cluster"],
                    "matched_keywords": list(dict.fromkeys(matched_keywords)),
                    "possible_causes": template["possible_causes"],
                    "related_tests": template["related_tests"],
                }
            )
        return matches

    def _select_workups(
        self,
        text: str,
        candidate_syndromes: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        selected: dict[str, dict[str, Any]] = {}
        for item in candidate_syndromes:
            syndrome = next(
                (entry for entry in self.kb.list_syndromes() if entry["id"] == item["id"]),
                None,
            )
            if syndrome is None:
                continue
            for workup_id in syndrome.get("recommended_workup_ids", []):
                workup = self.kb.get_workup(workup_id)
                if workup is not None:
                    selected[workup_id] = workup

        for workup in self.kb.list_workups():
            matched_keywords = [
                keyword
                for keyword in workup.get("if_any_keywords", [])
                if self._keyword_present(text, keyword.lower())
            ]
            if len(matched_keywords) >= int(workup.get("minimum_matches", 1)):
                selected[workup["id"]] = workup

        priority_order = {"high": 0, "medium": 1, "low": 2}
        return sorted(
            selected.values(),
            key=lambda item: (priority_order.get(item.get("priority", "medium"), 1), item["name_zh"]),
        )

    def _build_perioperative_focus(
        self,
        risk_rules: list[dict[str, Any]],
        normalized_patient: dict[str, Any],
    ) -> list[str]:
        focus: list[str] = []
        for rule in risk_rules:
            focus.extend(rule.get("perioperative_focus", []))

        medications = " ".join(normalized_patient.get("medication_history", []))
        if any(token in medications for token in ["阿司匹林", "氯吡格雷", "华法林", "利伐沙班", "阿哌沙班"]):
            focus.append("明确神经外科手术前抗栓药停用时程和再启动计划。")

        if normalized_patient.get("family_history"):
            focus.append("结合家族史评估遗传性癫痫、血管病或麻醉相关风险。")

        return list(dict.fromkeys(focus))

    def _match_complications(
        self,
        text: str,
        risk_rules: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        triggered_risk_ids = {rule["id"] for rule in risk_rules}
        watchlist: list[dict[str, Any]] = []
        for rule in self.kb.list_complications():
            matched_keywords = [
                keyword
                for keyword in rule.get("if_any_keywords", [])
                if self._keyword_present(text, keyword.lower())
            ]
            overlap = triggered_risk_ids.intersection(rule.get("triggered_by_risk_rule_ids", []))
            if not matched_keywords and not overlap:
                continue
            watchlist.append(
                {
                    "id": rule["id"],
                    "name_zh": rule["name_zh"],
                    "complications": rule["complications"],
                    "reason": rule["reason_zh"],
                }
            )
        return watchlist

    def _identify_missing_information(self, normalized_patient: dict[str, Any]) -> list[str]:
        missing: list[str] = []
        if not normalized_patient.get("chief_complaint"):
            missing.append("缺少主诉")
        if not normalized_patient.get("presenting_illness"):
            missing.append("缺少现病史")
        if not normalized_patient.get("suspected_diagnoses"):
            missing.append("缺少初步诊断或疑似病因")
        if not normalized_patient.get("neurological_exam"):
            missing.append("缺少神经系统查体")
        if not normalized_patient.get("medication_history"):
            missing.append("缺少长期用药和抗栓药信息")
        if not normalized_patient.get("family_history"):
            missing.append("缺少家族史/遗传病史")
        if not normalized_patient.get("imaging"):
            missing.append("缺少关键影像学信息")
        if not normalized_patient.get("labs"):
            missing.append("缺少实验室检查摘要")
        if not normalized_patient.get("functional_status"):
            missing.append("缺少功能状态和日常活动能力")
        return missing

    def _build_retrieval_notes(
        self,
        candidate_syndromes: list[dict[str, Any]],
        red_flags: list[dict[str, Any]],
        risk_rules: list[dict[str, Any]],
    ) -> list[str]:
        notes: list[str] = []
        if candidate_syndromes:
            notes.append(f"已识别 {len(candidate_syndromes)} 个神经外科候选问题。")
        if red_flags:
            notes.append(f"已命中 {len(red_flags)} 条神经外科红旗规则。")
        if risk_rules:
            notes.append(f"已识别 {len(risk_rules)} 条围术期风险规则。")
        return notes

    def _normalize_api_retrieval_result(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = {
            "candidate_syndromes": self._normalize_object_list(payload.get("candidate_syndromes")),
            "candidate_syndrome_scores": payload.get("candidate_syndrome_scores", {}),
            "triggered_red_flags": self._normalize_object_list(payload.get("triggered_red_flags")),
            "triggered_risk_rules": self._normalize_object_list(payload.get("triggered_risk_rules")),
            "urgency_level": payload.get("urgency_level", "needs_more_information"),
            "differential_considerations": self._normalize_object_list(payload.get("differential_considerations")),
            "recommended_workup": self._normalize_object_list(payload.get("recommended_workup")),
            "perioperative_focus": self._normalize_string_list(payload.get("perioperative_focus")),
            "complication_watchlist": self._normalize_object_list(payload.get("complication_watchlist")),
            "external_tools": self._normalize_object_list(payload.get("external_tools")) or self.kb.list_external_tools(),
            "missing_information": self._normalize_string_list(payload.get("missing_information")),
            "retrieval_notes": self._normalize_string_list(payload.get("retrieval_notes")),
            "graph_context": payload.get("graph_context", {}),
        }
        return normalized

    def _normalize_object_list(self, value: Any) -> list[dict[str, Any]]:
        if value is None:
            return []
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            return [value]
        return []

    def _normalize_string_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            text = value.replace("，", ",").replace("；", ",").replace("、", ",").replace("\n", ",")
            return [item.strip() for item in text.split(",") if item.strip()]
        if isinstance(value, list):
            result: list[str] = []
            for item in value:
                if item is None:
                    continue
                if isinstance(item, str):
                    stripped = item.strip()
                    if stripped:
                        result.append(stripped)
                else:
                    result.append(str(item).strip())
            return result
        return [str(value).strip()]

    def _load_graph_settings(self) -> dict[str, Any]:
        graph_cfg = self.kb.data.get("knowledge_graph", {})
        retrieval_cfg = graph_cfg.get("retrieval", {})
        edge_weights = graph_cfg.get("edge_weights", {})
        return {
            "enabled": bool(graph_cfg.get("enabled", True)),
            "max_hops": int(retrieval_cfg.get("max_hops", 2)),
            "decay": float(retrieval_cfg.get("decay", 0.72)),
            "min_edge_weight": float(retrieval_cfg.get("min_edge_weight", 0.2)),
            "syndrome_expand_threshold": float(retrieval_cfg.get("syndrome_expand_threshold", 0.2)),
            "workup_expand_threshold": float(retrieval_cfg.get("workup_expand_threshold", 0.2)),
            "complication_expand_threshold": float(retrieval_cfg.get("complication_expand_threshold", 0.2)),
            "seed_weights": {
                "syndrome": float(retrieval_cfg.get("syndrome_seed_weight", 1.0)),
                "red_flag": float(retrieval_cfg.get("red_flag_seed_weight", 1.0)),
                "risk_rule": float(retrieval_cfg.get("risk_rule_seed_weight", 0.95)),
            },
            "edge_weights": {
                "syndrome_to_workup": float(edge_weights.get("syndrome_to_workup", 0.95)),
                "syndrome_to_risk_rule": float(edge_weights.get("syndrome_to_risk_rule", 0.75)),
                "syndrome_to_red_flag": float(edge_weights.get("syndrome_to_red_flag", 0.8)),
                "risk_rule_to_complication": float(edge_weights.get("risk_rule_to_complication", 0.85)),
            },
        }

    def _build_knowledge_graph(self) -> tuple[dict[str, list[tuple[str, float, str]]], dict[str, str]]:
        graph: dict[str, list[tuple[str, float, str]]] = defaultdict(list)
        node_types: dict[str, str] = {}
        edge_weights = self._graph_settings["edge_weights"]

        def node_key(node_type: str, node_id: str) -> str:
            return f"{node_type}:{node_id}"

        def add_edge(left_type: str, left_id: str, right_type: str, right_id: str, weight: float, relation: str) -> None:
            if not left_id or not right_id:
                return
            left = node_key(left_type, left_id)
            right = node_key(right_type, right_id)
            node_types[left] = left_type
            node_types[right] = right_type
            graph[left].append((right, weight, relation))
            graph[right].append((left, weight, relation))

        syndromes = self.kb.list_syndromes()
        red_flags = self.kb.list_red_flags()
        risk_rules = self.kb.list_risk_rules()
        workups = self.kb.list_workups()

        for syndrome in syndromes:
            syndrome_id = str(syndrome.get("id") or "")
            for workup_id in syndrome.get("recommended_workup_ids", []):
                add_edge("syndrome", syndrome_id, "workup", str(workup_id), edge_weights["syndrome_to_workup"], "syndrome_recommends_workup")

            syndrome_keywords = {str(item).lower() for item in syndrome.get("trigger_keywords", [])}
            for risk_rule in risk_rules:
                risk_id = str(risk_rule.get("id") or "")
                risk_keywords = {str(item).lower() for item in risk_rule.get("if_any_keywords", [])}
                if syndrome_keywords.intersection(risk_keywords):
                    add_edge("syndrome", syndrome_id, "risk_rule", risk_id, edge_weights["syndrome_to_risk_rule"], "syndrome_related_risk_rule")

            for red_flag in red_flags:
                red_flag_id = str(red_flag.get("id") or "")
                red_flag_keywords = {str(item).lower() for item in red_flag.get("if_any_keywords", [])}
                if syndrome_keywords.intersection(red_flag_keywords):
                    add_edge("syndrome", syndrome_id, "red_flag", red_flag_id, edge_weights["syndrome_to_red_flag"], "syndrome_related_red_flag")

            for workup in workups:
                workup_id = str(workup.get("id") or "")
                workup_keywords = {str(item).lower() for item in workup.get("if_any_keywords", [])}
                if syndrome_keywords.intersection(workup_keywords):
                    add_edge("syndrome", syndrome_id, "workup", workup_id, edge_weights["syndrome_to_workup"], "keyword_related_workup")

        for complication in self.kb.list_complications():
            comp_id = str(complication.get("id") or "")
            for risk_id in complication.get("triggered_by_risk_rule_ids", []):
                add_edge("risk_rule", str(risk_id), "complication", comp_id, edge_weights["risk_rule_to_complication"], "risk_rule_related_complication")

        return dict(graph), node_types

    def _graph_rank_entities(
        self,
        syndrome_scores: list[dict[str, Any]],
        red_flags: list[dict[str, Any]],
        risk_rules: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not self._graph_settings["enabled"]:
            return {
                "syndrome_scores": {},
                "workup_scores": {},
                "complication_scores": {},
                "graph_context": {"enabled": False},
            }

        seed_nodes: dict[str, float] = {}
        seed_weights = self._graph_settings["seed_weights"]
        for item in syndrome_scores:
            syndrome = item.get("syndrome", {})
            syndrome_id = str(syndrome.get("id") or "")
            if syndrome_id:
                lexical = float(item.get("score", 0.0))
                seed_nodes[f"syndrome:{syndrome_id}"] = max(seed_nodes.get(f"syndrome:{syndrome_id}", 0.0), seed_weights["syndrome"] + lexical * 0.25)
        for item in red_flags:
            item_id = str(item.get("id") or "")
            if item_id:
                seed_nodes[f"red_flag:{item_id}"] = max(seed_nodes.get(f"red_flag:{item_id}", 0.0), seed_weights["red_flag"])
        for item in risk_rules:
            item_id = str(item.get("id") or "")
            if item_id:
                seed_nodes[f"risk_rule:{item_id}"] = max(seed_nodes.get(f"risk_rule:{item_id}", 0.0), seed_weights["risk_rule"])

        node_scores: dict[str, float] = defaultdict(float)
        frontier: dict[str, float] = {}
        for node, weight in seed_nodes.items():
            if node in self._node_types:
                node_scores[node] += weight
                frontier[node] = max(frontier.get(node, 0.0), weight)

        for _ in range(self._graph_settings["max_hops"]):
            next_frontier: dict[str, float] = {}
            for source, source_weight in frontier.items():
                for target, edge_weight, _relation in self._graph.get(source, []):
                    if edge_weight < self._graph_settings["min_edge_weight"]:
                        continue
                    propagated = source_weight * edge_weight * self._graph_settings["decay"]
                    if propagated < 1e-6:
                        continue
                    node_scores[target] += propagated
                    next_frontier[target] = max(next_frontier.get(target, 0.0), propagated)
            frontier = next_frontier
            if not frontier:
                break

        syndrome_score_map = self._extract_type_scores(node_scores, "syndrome", self._graph_settings["syndrome_expand_threshold"])
        workup_score_map = self._extract_type_scores(node_scores, "workup", self._graph_settings["workup_expand_threshold"])
        complication_score_map = self._extract_type_scores(node_scores, "complication", self._graph_settings["complication_expand_threshold"])
        return {
            "syndrome_scores": syndrome_score_map,
            "workup_scores": workup_score_map,
            "complication_scores": complication_score_map,
            "graph_context": {
                "enabled": True,
                "seed_nodes": sorted(seed_nodes.keys()),
                "top_syndrome_ids": [item_id for item_id, _ in sorted(syndrome_score_map.items(), key=lambda item: item[1], reverse=True)[:6]],
                "top_workup_ids": [item_id for item_id, _ in sorted(workup_score_map.items(), key=lambda item: item[1], reverse=True)[:8]],
                "top_complication_ids": [item_id for item_id, _ in sorted(complication_score_map.items(), key=lambda item: item[1], reverse=True)[:6]],
            },
        }

    def _extract_type_scores(self, node_scores: dict[str, float], node_type: str, min_threshold: float) -> dict[str, float]:
        prefix = f"{node_type}:"
        result: dict[str, float] = {}
        for node_key, score in node_scores.items():
            if node_key.startswith(prefix) and score >= min_threshold:
                result[node_key.removeprefix(prefix)] = round(float(score), 6)
        return result

    def _rerank_items_by_graph_score(
        self,
        items: list[dict[str, Any]],
        graph_scores: dict[str, float],
        item_id_key: str,
    ) -> list[dict[str, Any]]:
        if not items:
            return []
        graph_scores = graph_scores or {}
        return sorted(
            items,
            key=lambda item: float(graph_scores.get(str(item.get(item_id_key, "")), 0.0)),
            reverse=True,
        )

    def _keyword_present(self, text: str, keyword: str) -> bool:
        start = 0
        while True:
            index = text.find(keyword, start)
            if index == -1:
                return False
            if not self._is_negated(text, index, len(keyword)):
                return True
            start = index + len(keyword)

    def _is_negated(self, text: str, index: int, keyword_length: int) -> bool:
        left_window = text[max(0, index - 8) : index]
        right_window = text[index + keyword_length : index + keyword_length + 4]
        negation_tokens = [
            "否认",
            "未见",
            "无",
            "未",
            "不伴",
            "排除",
            "除外",
            "未提示",
        ]
        if any(token in left_window for token in negation_tokens):
            return True
        return any(token in right_window for token in ["阴性", "未见"])

    def _extract_asa_grade(self, asa_hint: Any) -> int | None:
        if asa_hint is None:
            return None
        text = str(asa_hint).upper().replace(" ", "")
        if "IV" in text:
            return 4
        if "III" in text:
            return 3
        if "II" in text:
            return 2
        if "I" in text:
            return 1
        return None

    def _at_least(self, value: Any, threshold: float) -> bool:
        try:
            if value is None:
                return False
            return float(value) >= float(threshold)
        except (TypeError, ValueError):
            return False
