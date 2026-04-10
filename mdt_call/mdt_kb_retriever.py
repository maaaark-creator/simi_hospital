from __future__ import annotations

import json
from pathlib import Path
from typing import Any


KB_PATH = Path(__file__).with_name("mdt_call_kb.json")
DEFAULT_KB = {
    "routing_rules": [
        {
            "specialty": "anesthesia",
            "keywords": ["麻醉", "气道", "ASA", "禁食", "镇痛", "围术期", "插管", "苏醒"],
        },
        {
            "specialty": "cardiology",
            "keywords": ["心电图", "胸痛", "心律", "ST", "心衰", "血压", "灌注", "抗凝", "心源性"],
        },
        {
            "specialty": "hepatobiliary",
            "keywords": ["黄疸", "胆道", "肝", "胰", "感染", "ERCP", "引流", "胆红素", "出血"],
        },
        {
            "specialty": "neurosurgery",
            "keywords": ["颅", "脑", "神经", "脊髓", "头痛", "癫痫", "脑膜", "脑出血", "脑积水", "肿瘤"],
        },
    ]
}


class MDTKnowledgeRetriever:
    def __init__(self, path: Path = KB_PATH) -> None:
        self.path = path
        self.data = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return DEFAULT_KB
        content = self.path.read_text(encoding="utf-8-sig").strip()
        if not content:
            return DEFAULT_KB
        return json.loads(content)

    def build_dispatch_plan(self, request_context: dict[str, Any]) -> dict[str, Any]:
        tasks = []
        for item in request_context.get("unresolved_items", []):
            question = str(item.get("question") or "").strip()
            if not question:
                continue
            specialty = self._match_specialty(
                question=question,
                reason=str(item.get("reason") or ""),
                source=str(item.get("source") or ""),
            )
            tasks.append(
                {
                    "specialty": specialty,
                    "question": question,
                    "reason": item.get("reason") or "需要进一步澄清。",
                    "priority": item.get("priority") or "medium",
                }
            )
        return {
            "tasks": tasks,
            "rationale": "依据存疑问题关键词与分诊规则完成二次下派。",
            "priority_summary": self._build_priority_summary(tasks),
        }

    def build_initial_triage_plan(self, request_context: dict[str, Any]) -> dict[str, Any]:
        patient_blob = json.dumps(request_context, ensure_ascii=False)
        tasks = []
        seen: set[str] = set()
        for rule in self.data.get("routing_rules", []):
            specialty = str(rule.get("specialty") or "").strip()
            if not specialty:
                continue
            keywords = [str(keyword) for keyword in rule.get("keywords", [])]
            if any(keyword.lower() in patient_blob.lower() for keyword in keywords):
                if specialty in seen:
                    continue
                seen.add(specialty)
                tasks.append(
                    {
                        "specialty": specialty,
                        "reason": f"患者信息命中 {specialty} 分诊关键词。",
                        "priority": "high",
                    }
                )

        if not tasks:
            for specialty in ("anesthesia", "cardiology", "hepatobiliary", "neurosurgery"):
                tasks.append(
                    {
                        "specialty": specialty,
                        "reason": "未命中特异关键词，按默认 MDT 首轮会诊全量下发。",
                        "priority": "medium",
                    }
                )

        return {
            "tasks": tasks,
            "rationale": "依据患者首轮信息与分诊关键词决定需要先触发的专科。",
            "priority_summary": self._build_priority_summary(tasks),
        }

    def _match_specialty(self, question: str, reason: str, source: str) -> str:
        combined = f"{question} {reason}".lower()
        for rule in self.data.get("routing_rules", []):
            if any(str(keyword).lower() in combined for keyword in rule.get("keywords", [])):
                return str(rule.get("specialty") or "anesthesia")
        return source or "anesthesia"

    def _build_priority_summary(self, tasks: list[dict[str, Any]]) -> dict[str, int]:
        summary = {"high": 0, "medium": 0, "low": 0}
        for task in tasks:
            priority = str(task.get("priority") or "medium")
            summary[priority] = summary.get(priority, 0) + 1
        return summary
