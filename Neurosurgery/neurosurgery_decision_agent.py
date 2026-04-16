from __future__ import annotations

import json
from typing import Any

from neurosurgery_patient_mapper_agent import NEUROSURGERY_TASK_MODEL_SELECTION
from neurosurgery_shared import GenAIChatClient, MODEL_CONFIGS


class NeurosurgeryDecisionAgent:
    def __init__(self, api_client: GenAIChatClient | None = None) -> None:
        self.api_client = api_client or GenAIChatClient()

    def decide(
        self,
        normalized_patient: dict[str, Any],
        retrieval_result: dict[str, Any],
    ) -> dict[str, Any]:
        candidate_syndromes = retrieval_result.get("candidate_syndromes", [])
        red_flags = retrieval_result.get("triggered_red_flags", [])
        risk_rules = retrieval_result.get("triggered_risk_rules", [])

        return {
            "patient_summary": self._build_patient_summary(normalized_patient, retrieval_result),
            "urgency_level": retrieval_result.get("urgency_level", "needs_more_information"),
            "surgical_readiness": self._assess_surgical_readiness(candidate_syndromes, red_flags),
            "preliminary_impression": self._build_preliminary_impression(candidate_syndromes),
            "diagnostic_considerations": retrieval_result.get("differential_considerations", []),
            "management_recommendations": self._build_management_recommendations(
                retrieval_result,
                normalized_patient,
            ),
            "recommended_workup": retrieval_result.get("recommended_workup", []),
            "perioperative_considerations": self._build_perioperative_considerations(
                retrieval_result,
                normalized_patient,
            ),
            "complication_watchlist": retrieval_result.get("complication_watchlist", []),
            "recommended_collaboration": self._build_collaboration(candidate_syndromes, red_flags),
            "need_more_info": retrieval_result.get("missing_information", []),
            "reasoning_trace": self._build_reasoning_trace(
                normalized_patient,
                retrieval_result,
            ),
        }

    def decide_with_api(
        self,
        normalized_patient: dict[str, Any],
        retrieval_result: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = (
            "You are a neurosurgery decision-support agent.\n"
            "Think like a careful neurosurgeon reviewing a perioperative case.\n"
            "Consider symptom evolution, neurological findings, imaging clues, comorbidities, "
            "family history, medication history, perioperative complications, and alternative causes.\n"
            "Return JSON only with keys:\n"
            "patient_summary, urgency_level, surgical_readiness, preliminary_impression, "
            "diagnostic_considerations, management_recommendations, recommended_workup, "
            "perioperative_considerations, complication_watchlist, recommended_collaboration, "
            "need_more_info, reasoning_trace.\n\n"
            "Normalized patient:\n"
            f"{json.dumps(normalized_patient, ensure_ascii=False, indent=2)}\n\n"
            "Retrieval result:\n"
            f"{json.dumps(retrieval_result, ensure_ascii=False, indent=2)}"
        )
        config = MODEL_CONFIGS[NEUROSURGERY_TASK_MODEL_SELECTION["final_decision"]]
        response = self.api_client.chat_json(config, prompt)
        return self.api_client.extract_json(response)

    def _build_patient_summary(
        self,
        normalized_patient: dict[str, Any],
        retrieval_result: dict[str, Any],
    ) -> str:
        parts: list[str] = []
        if normalized_patient.get("age") is not None:
            parts.append(f"{normalized_patient['age']}岁")
        if normalized_patient.get("sex"):
            parts.append(str(normalized_patient["sex"]))
        if normalized_patient.get("chief_complaint"):
            parts.append(f"主诉：{normalized_patient['chief_complaint']}")
        if normalized_patient.get("planned_procedure"):
            parts.append(f"拟行：{normalized_patient['planned_procedure']}")
        if normalized_patient.get("suspected_diagnoses"):
            parts.append(f"当前关注：{'、'.join(normalized_patient['suspected_diagnoses'][:3])}")
        parts.append(f"紧急度：{retrieval_result.get('urgency_level', 'needs_more_information')}")
        return "；".join(parts)

    def _assess_surgical_readiness(
        self,
        candidate_syndromes: list[dict[str, Any]],
        red_flags: list[dict[str, Any]],
    ) -> str:
        if any(rule["severity"] == "high" for rule in red_flags):
            return "not_ready_requires_emergency_stabilization"
        if not candidate_syndromes:
            return "undetermined_requires_more_data"
        if any(item["score"] >= 3 for item in candidate_syndromes):
            return "candidate_after_targeted_workup"
        return "needs_more_information_before_surgical_discussion"

    def _build_preliminary_impression(
        self,
        candidate_syndromes: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        impressions: list[dict[str, Any]] = []
        for item in candidate_syndromes[:3]:
            impressions.append(
                {
                    "id": item["id"],
                    "name_zh": item["name_zh"],
                    "likelihood": self._score_to_likelihood(item["score"]),
                    "supporting_evidence": item.get("matched_keywords", []),
                    "focus_points": item.get("focus_points", []),
                }
            )
        return impressions

    def _build_management_recommendations(
        self,
        retrieval_result: dict[str, Any],
        normalized_patient: dict[str, Any],
    ) -> list[str]:
        recommendations: list[str] = []
        urgency = retrieval_result.get("urgency_level")
        if urgency == "immediate_emergency":
            recommendations.append("当前存在神经外科急症线索，应优先进行现场稳定和急诊神经外科评估。")
        elif urgency == "urgent_inpatient_review":
            recommendations.append("建议住院或加速完成评估，避免在门诊路径中延误关键影像和手术窗口。")
        elif urgency == "expedited_specialist_workup":
            recommendations.append("具备较强神经外科评估指征，建议尽快进入专病MDT或手术前评估流程。")
        else:
            recommendations.append("现有证据不足以直接进入手术决策，建议先补足关键病史、查体和影像。")

        candidate_ids = {item["id"] for item in retrieval_result.get("candidate_syndromes", [])}
        if "drug_resistant_epilepsy" in candidate_ids:
            recommendations.append("若已完成充分药物治疗仍控制不佳，应按癫痫外科路径完成电临床定位、MRI和神经心理评估。")
        if "intracranial_mass_effect" in candidate_ids:
            recommendations.append("对头痛、呕吐、视乳头水肿或意识变化病例，应优先排查占位效应和颅压增高。")
        if "ischemic_cerebrovascular_pathway" in candidate_ids:
            recommendations.append("对TIA或反复缺血发作病例，应结合血管成像、危险因素和药物治疗充分性判断是否讨论血运重建。")
        if "hydrocephalus_csf_disorder" in candidate_ids:
            recommendations.append("对脑积水或分流相关病例，应将症状轨迹与脑室影像、分流装置状态和感染迹象一并复核。")
        if "spinal_cord_or_cauda_compression" in candidate_ids:
            recommendations.append("存在进行性无力、尿潴留或鞍区感觉异常时，应尽快完成靶向脊柱MRI并警惕减压时效。")
        if "intracranial_bleeding_or_trauma" in candidate_ids:
            recommendations.append("头部外伤合并抗凝或意识恶化时，应优先完成急诊影像、凝血复核和逆转策略评估。")

        if normalized_patient.get("family_history"):
            recommendations.append("家族史不能忽略，应结合遗传性癫痫、脑血管病及麻醉相关风险做额外甄别。")
        return recommendations

    def _build_perioperative_considerations(
        self,
        retrieval_result: dict[str, Any],
        normalized_patient: dict[str, Any],
    ) -> list[str]:
        considerations = list(retrieval_result.get("perioperative_focus", []))
        if normalized_patient.get("asa_hint"):
            considerations.append(f"结合 ASA 提示 {normalized_patient['asa_hint']} 评估全身耐受性和麻醉准备度。")
        if normalized_patient.get("vitals"):
            considerations.append("术前需复核血压、心率、氧合等生命体征是否稳定，避免在不稳定状态下推进流程。")
        if normalized_patient.get("imaging"):
            considerations.append("神经外科决策高度依赖影像与症状对应关系，必要时应复核原始影像而非仅看报告摘要。")
        return list(dict.fromkeys(considerations))

    def _build_collaboration(
        self,
        candidate_syndromes: list[dict[str, Any]],
        red_flags: list[dict[str, Any]],
    ) -> list[str]:
        collaborators: list[str] = ["麻醉科"]
        for item in candidate_syndromes:
            collaborators.extend(item.get("recommended_collaborators", []))
        if any(rule["severity"] == "high" for rule in red_flags):
            collaborators.extend(["急诊科", "重症医学科"])
        return list(dict.fromkeys(collaborators))

    def _build_reasoning_trace(
        self,
        normalized_patient: dict[str, Any],
        retrieval_result: dict[str, Any],
    ) -> list[str]:
        trace: list[str] = []
        if retrieval_result.get("candidate_syndromes"):
            trace.append("已将主诉、症状演变、神经查体、影像线索和拟行操作综合映射到神经外科候选问题。")
        if retrieval_result.get("triggered_red_flags"):
            trace.append("已优先识别需要即时升级处理的神经外科红旗表现。")
        if normalized_patient.get("medication_history"):
            trace.append("已纳入抗栓药、抗癫痫药等长期用药对围术期风险和手术时机的影响。")
        if normalized_patient.get("family_history"):
            trace.append("已将家族遗传背景纳入鉴别和风险判断。")
        if retrieval_result.get("differential_considerations"):
            trace.append("已保留非单一病因视角，提示需和血管性、肿瘤性、代谢性或感染性病因鉴别。")
        return trace

    def _score_to_likelihood(self, score: int) -> str:
        if score >= 4:
            return "high"
        if score >= 2:
            return "moderate"
        return "possible"
