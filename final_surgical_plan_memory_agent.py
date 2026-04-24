from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from head_doctor.head_doctor_agent import MODEL_CONFIGS, SharedGenAIChatClient


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PLAN_KB_PATH = PROJECT_ROOT / "final_surgical_plan_kb.json"


@dataclass
class SurgicalPlanSession:
    session_id: str
    created_at: float
    updated_at: float
    base_mdt_output: dict[str, Any]
    supplemental_inputs: list[str] = field(default_factory=list)
    output_history: list[dict[str, Any]] = field(default_factory=list)


class FinalSurgicalPlanMemoryAgent:
    """Agent for multi-turn surgical planning with memory.

    Input memory:
    - previous MDT output (workflow or final recommendation)
    - incremental supplemental facts from user

    Output:
    - concrete, executable perioperative surgical plan
    """

    def __init__(
        self,
        plan_kb_path: str | Path = DEFAULT_PLAN_KB_PATH,
        api_client: SharedGenAIChatClient | None = None,
    ) -> None:
        self.plan_kb_path = Path(plan_kb_path)
        self.plan_kb = self._load_plan_kb(self.plan_kb_path)
        self.api_client = api_client or SharedGenAIChatClient()
        self.sessions: dict[str, SurgicalPlanSession] = {}

    def create_session(
        self,
        mdt_output: dict[str, Any],
        session_id: str | None = None,
    ) -> str:
        sid = (session_id or "").strip() or uuid.uuid4().hex
        now = time.time()
        self.sessions[sid] = SurgicalPlanSession(
            session_id=sid,
            created_at=now,
            updated_at=now,
            base_mdt_output=mdt_output,
        )
        return sid

    def append_supplement(self, session_id: str, supplemental_info: str | dict[str, Any]) -> None:
        session = self._get_session(session_id)
        if isinstance(supplemental_info, dict):
            text = json.dumps(supplemental_info, ensure_ascii=False)
        else:
            text = str(supplemental_info).strip()
        if text:
            session.supplemental_inputs.append(text)
            session.updated_at = time.time()

    def update_and_plan(
        self,
        session_id: str,
        supplemental_info: str | dict[str, Any] | None = None,
        use_api: bool = True,
        model_key: str = "gpt_5_2",
    ) -> dict[str, Any]:
        if supplemental_info is not None:
            self.append_supplement(session_id, supplemental_info)
        return self.generate_final_plan(session_id=session_id, use_api=use_api, model_key=model_key)

    def generate_final_plan(
        self,
        session_id: str,
        use_api: bool = True,
        model_key: str = "gpt_5_2",
    ) -> dict[str, Any]:
        session = self._get_session(session_id)
        final_recommendation = self._extract_final_recommendation(session.base_mdt_output)
        status_level = str(final_recommendation.get("current_status_level") or "OPTIMIZE").strip().upper()
        specialty_ids = self._extract_specialty_ids(session.base_mdt_output)
        unresolved = self._unique_list(
            self._as_list(final_recommendation.get("core_constraints"))
            + self._as_list(final_recommendation.get("unresolved_issues"))
        )
        key_risks = self._unique_list(self._as_list(final_recommendation.get("key_risks")))

        seed_plan = self._build_seed_plan(status_level=status_level, specialty_ids=specialty_ids)

        if use_api:
            try:
                result = self._generate_with_api(
                    session=session,
                    status_level=status_level,
                    specialty_ids=specialty_ids,
                    unresolved=unresolved,
                    key_risks=key_risks,
                    seed_plan=seed_plan,
                    model_key=model_key,
                )
            except Exception as exc:
                result = self._generate_locally(
                    session=session,
                    status_level=status_level,
                    specialty_ids=specialty_ids,
                    unresolved=unresolved,
                    key_risks=key_risks,
                    seed_plan=seed_plan,
                )
                result["generation_fallback_reason"] = str(exc)
                result["generated_by"] = "local_rule_fallback"
        else:
            result = self._generate_locally(
                session=session,
                status_level=status_level,
                specialty_ids=specialty_ids,
                unresolved=unresolved,
                key_risks=key_risks,
                seed_plan=seed_plan,
            )
            result["generated_by"] = "local_rule"

        session.output_history.append(result)
        session.updated_at = time.time()
        return result

    def _generate_with_api(
        self,
        session: SurgicalPlanSession,
        status_level: str,
        specialty_ids: list[str],
        unresolved: list[str],
        key_risks: list[str],
        seed_plan: list[dict[str, str]],
        model_key: str,
    ) -> dict[str, Any]:
        if model_key not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model_key: {model_key}")

        prompt = (
            "你是一个术前会诊后的最终手术方案制定 agent。\n"
            "你会收到：上一轮 MDT 输出（作为记忆）、后续补充信息、以及一个模板化 seed plan。\n"
            "请输出“可执行”的最终方案，不要写概述性空话。\n"
            "只返回 JSON，字段固定为：\n"
            "session_id, status_level, surgery_ready, final_surgical_plan, risk_control_points, pending_blockers, notes。\n"
            "- status_level 只能是 READY/OPTIMIZE/CONTRAINDICATED。\n"
            "- surgery_ready 只能是 yes/no。\n"
            "- final_surgical_plan 必须是数组，元素字段为 phase, action, owner, trigger, rationale。\n"
            "- final_surgical_plan 要按术前/术中/术后分层，且每条 action 必须可执行。\n"
            "- 如果补充信息已解决阻塞项，应显式体现在 final_surgical_plan。\n"
            "- pending_blockers 只保留仍然阻碍手术的项。\n\n"
            f"会话ID：{session.session_id}\n"
            f"当前状态等级：{status_level}\n"
            f"涉及专科：{json.dumps(specialty_ids, ensure_ascii=False)}\n"
            f"关键风险：{json.dumps(key_risks, ensure_ascii=False)}\n"
            f"未决阻塞项：{json.dumps(unresolved, ensure_ascii=False)}\n"
            f"新增补充信息：{json.dumps(session.supplemental_inputs, ensure_ascii=False)}\n"
            f"seed plan：{json.dumps(seed_plan, ensure_ascii=False, indent=2)}\n"
            f"上一轮 MDT 输出：{json.dumps(session.base_mdt_output, ensure_ascii=False, indent=2)}"
        )
        response = self.api_client.chat_json(MODEL_CONFIGS[model_key], prompt)
        payload = self.api_client.extract_json(response)
        return self._normalize_output(payload, session.session_id, status_level, seed_plan)

    def _generate_locally(
        self,
        session: SurgicalPlanSession,
        status_level: str,
        specialty_ids: list[str],
        unresolved: list[str],
        key_risks: list[str],
        seed_plan: list[dict[str, str]],
    ) -> dict[str, Any]:
        surgery_ready = "yes" if status_level == "READY" and not unresolved else "no"
        final_surgical_plan: list[dict[str, str]] = []
        for row in seed_plan:
            final_surgical_plan.append(
                {
                    "phase": row["phase"],
                    "action": row["action"],
                    "owner": "MDT联合团队",
                    "trigger": "按当前状态执行",
                    "rationale": "基于MDT结论与补充信息生成",
                }
            )

        if session.supplemental_inputs:
            final_surgical_plan.insert(
                0,
                {
                    "phase": "术前",
                    "action": f"核对新增补充信息并更新术前核查单（共{len(session.supplemental_inputs)}条补充）。",
                    "owner": "head_doctor/病区医生",
                    "trigger": "进入最终排台前",
                    "rationale": "确保最新信息已纳入最终手术决策",
                },
            )

        return {
            "session_id": session.session_id,
            "status_level": status_level,
            "surgery_ready": surgery_ready,
            "final_surgical_plan": final_surgical_plan,
            "risk_control_points": key_risks[:6],
            "pending_blockers": unresolved[:6],
            "notes": "该结果由规则引擎生成，可再调用API模型细化。",
        }

    def _normalize_output(
        self,
        payload: dict[str, Any],
        session_id: str,
        default_status_level: str,
        seed_plan: list[dict[str, str]],
    ) -> dict[str, Any]:
        status_level = str(payload.get("status_level") or default_status_level).strip().upper()
        if status_level not in {"READY", "OPTIMIZE", "CONTRAINDICATED"}:
            status_level = default_status_level

        surgery_ready = str(payload.get("surgery_ready") or "").strip().lower()
        if surgery_ready not in {"yes", "no"}:
            surgery_ready = "yes" if status_level == "READY" else "no"

        plan_rows = payload.get("final_surgical_plan")
        normalized_plan: list[dict[str, str]] = []
        if isinstance(plan_rows, list):
            for item in plan_rows:
                if not isinstance(item, dict):
                    continue
                phase = str(item.get("phase") or "").strip() or "术前"
                action = str(item.get("action") or "").strip()
                if not action:
                    continue
                normalized_plan.append(
                    {
                        "phase": phase,
                        "action": action,
                        "owner": str(item.get("owner") or "MDT联合团队").strip(),
                        "trigger": str(item.get("trigger") or "按计划执行").strip(),
                        "rationale": str(item.get("rationale") or "基于会诊结论").strip(),
                    }
                )
        if not normalized_plan:
            normalized_plan = [
                {
                    "phase": row["phase"],
                    "action": row["action"],
                    "owner": "MDT联合团队",
                    "trigger": "按计划执行",
                    "rationale": "基于seed plan兜底",
                }
                for row in seed_plan
            ]

        return {
            "session_id": session_id,
            "status_level": status_level,
            "surgery_ready": surgery_ready,
            "final_surgical_plan": normalized_plan[:12],
            "risk_control_points": self._as_list(payload.get("risk_control_points"))[:8],
            "pending_blockers": self._as_list(payload.get("pending_blockers"))[:8],
            "notes": str(payload.get("notes") or "").strip(),
            "generated_by": "api_model",
        }

    def _build_seed_plan(self, status_level: str, specialty_ids: list[str]) -> list[dict[str, str]]:
        phase_order = self._as_list(self.plan_kb.get("phase_order")) or ["术前", "术中", "术后"]
        level_defaults = (
            self.plan_kb.get("status_level_defaults", {}).get(status_level)
            or self.plan_kb.get("status_level_defaults", {}).get("OPTIMIZE")
            or {}
        )
        specialty_overrides = self.plan_kb.get("specialty_overrides", {})

        plan_rows: list[dict[str, str]] = []
        for phase in phase_order:
            for action in self._as_list(level_defaults.get(phase)):
                plan_rows.append({"phase": str(phase), "action": str(action)})
            for specialty_id in specialty_ids:
                override = specialty_overrides.get(specialty_id, {})
                for action in self._as_list(override.get(phase)):
                    plan_rows.append({"phase": str(phase), "action": str(action)})

        seen: set[str] = set()
        deduped: list[dict[str, str]] = []
        for row in plan_rows:
            key = f"{row['phase']}::{row['action']}".casefold()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
        return deduped

    def _extract_final_recommendation(self, mdt_output: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(mdt_output, dict):
            return {}
        if isinstance(mdt_output.get("head_doctor_recommendation"), dict):
            return dict(mdt_output.get("head_doctor_recommendation") or {})
        return mdt_output

    def _extract_specialty_ids(self, mdt_output: dict[str, Any]) -> list[str]:
        if not isinstance(mdt_output, dict):
            return []
        opinions = mdt_output.get("specialty_opinions")
        if not isinstance(opinions, dict):
            return []
        ids = [str(k).strip() for k in opinions.keys() if str(k).strip()]
        return self._unique_list(ids)

    def _get_session(self, session_id: str) -> SurgicalPlanSession:
        sid = str(session_id).strip()
        if not sid or sid not in self.sessions:
            raise KeyError(f"Unknown session_id: {session_id}")
        return self.sessions[sid]

    def _load_plan_kb(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Plan KB file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _as_list(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    def _unique_list(self, items: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            text = str(item).strip()
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            result.append(text)
        return result


if __name__ == "__main__":
    # Minimal standalone demo
    demo_output = {
        "head_doctor_recommendation": {
            "current_status_level": "OPTIMIZE",
            "core_constraints": ["凝血结果未回报", "术后ICU床位未确认"],
            "key_risks": ["围术期出血风险", "术后神经功能恶化风险"],
            "final_plan": ["补齐关键检查", "完成麻醉与神外联合复核"],
        },
        "specialty_opinions": {
            "neurosurgery": {},
            "anesthesia": {},
        },
    }
    agent = FinalSurgicalPlanMemoryAgent()
    sid = agent.create_session(demo_output)
    plan_v1 = agent.generate_final_plan(sid, use_api=False)
    print(json.dumps(plan_v1, ensure_ascii=False, indent=2))

    plan_v2 = agent.update_and_plan(
        sid,
        supplemental_info="补充：INR=1.01, APTT正常，ICU床位已确认。",
        use_api=False,
    )
    print(json.dumps(plan_v2, ensure_ascii=False, indent=2))
