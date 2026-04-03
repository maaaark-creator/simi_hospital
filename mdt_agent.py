#!/usr/bin/env python3
"""MDT 调度 Agent（API 调用版）

功能：
1. 基于病例综合信息自动判定需会诊科室（麻醉科必调）。
2. 通过统一 API 调用各专科 Agent。
3. 汇总各科意见并给出 MDT 总结。
4. 自动加载 knowledge/ 下全部资料并注入提示词进行比对。
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import urllib.error
import urllib.request


DEPARTMENT_LABELS = {
    "cardiology": "心血管内科",
    "cardiac_surgery": "心外科",
    "neuro": "神经内外科",
    "general_surgery": "普外科",
    "orthopedics": "骨科",
    "anesthesiology": "麻醉科",
}


@dataclass
class AgentConfig:
    """单个专科 Agent 的 API 配置。"""

    name: str
    endpoint: str
    api_key: str | None = None


class MDTAgent:
    """MDT 调度 Agent。"""

    def __init__(self, configs: Dict[str, AgentConfig], timeout: int = 90) -> None:
        self.configs = configs
        self.timeout = timeout

    def plan_departments(self, case: Dict[str, Any]) -> List[str]:
        """按病例信息决定需调用的专科，麻醉科始终在内。"""
        text_blobs = self._collect_case_text(case)
        selected = {"anesthesiology"}  # 必调

        def has_any(*keywords: str) -> bool:
            blob = "\n".join(text_blobs)
            return any(k.lower() in blob.lower() for k in keywords)

        # 心血管内科 / 心外科
        if has_any("胸痛", "心衰", "冠心病", "心律失常", "高血压", "st段", "troponin", "heart", "cardiac"):
            selected.add("cardiology")
        if has_any("瓣膜", "搭桥", "主动脉", "先心病", "心脏手术", "cabg", "valve", "aorta"):
            selected.add("cardiac_surgery")

        # 神经内外科
        if has_any("头痛", "偏瘫", "癫痫", "脑梗", "脑出血", "颅", "脊髓", "stroke", "brain", "neuro"):
            selected.add("neuro")

        # 普外科
        if has_any("腹痛", "阑尾", "胆囊", "肝", "胰", "肠梗阻", "疝", "胃肠", "abdomen", "appendix"):
            selected.add("general_surgery")

        # 骨科
        if has_any("骨折", "关节", "脊柱", "创伤", "膝", "髋", "ortho", "fracture", "joint", "spine"):
            selected.add("orthopedics")

        # 高危合并症提升会诊广度
        if has_any("糖尿病", "慢阻肺", "肾衰", "凝血", "妊娠", "高龄", "多器官", "sepsis"):
            selected.update({"cardiology", "general_surgery"})

        # 仅返回可配置的科室，避免未配置报错
        return [d for d in DEPARTMENT_LABELS if d in selected and d in self.configs]

    def run(self, case: Dict[str, Any], knowledge_dir: Path | None = None) -> Dict[str, Any]:
        departments = self.plan_departments(case)
        knowledge_bundle = load_knowledge_bundle(knowledge_dir) if knowledge_dir else ""

        dept_results: Dict[str, Any] = {}
        for dept in departments:
            cfg = self.configs[dept]
            payload = self._build_payload(case, dept, knowledge_bundle)
            dept_results[dept] = self._call_agent(cfg, payload)

        summary = self._synthesize(case, departments, dept_results)
        return {
            "departments_called": [DEPARTMENT_LABELS[d] for d in departments],
            "department_results": dept_results,
            "mdt_summary": summary,
        }

    def _collect_case_text(self, case: Dict[str, Any]) -> List[str]:
        keys = [
            "chief_complaint",
            "history_present_illness",
            "past_medical_history",
            "family_history",
            "physical_exam",
            "possible_complications",
            "differential_diagnosis",
        ]
        text_blobs = []
        for k in keys:
            v = case.get(k)
            if isinstance(v, str):
                text_blobs.append(v)
            elif isinstance(v, list):
                text_blobs.extend(str(x) for x in v)
            elif v is not None:
                text_blobs.append(str(v))
        return text_blobs

    def _build_payload(self, case: Dict[str, Any], dept: str, knowledge_bundle: str) -> Dict[str, Any]:
        system_prompt = (
            f"你是{DEPARTMENT_LABELS[dept]}医生。请像真实临床医生一样，综合主诉、既往史、家族史、"
            "体征、并发症风险和鉴别诊断，输出结构化建议（评估、风险、治疗建议、需补充检查）。"
        )
        if knowledge_bundle:
            system_prompt += "\n请优先参考以下knowledge资料并进行比对：\n" + knowledge_bundle

        return {
            "system": system_prompt,
            "input": {
                "patient_case": case,
                "requirements": {
                    "must_consider": [
                        "病例整体情况",
                        "身体情况",
                        "家族遗传病",
                        "主诉",
                        "既往病史",
                        "可能并发症",
                        "症状可能的多种病因",
                    ]
                },
            },
            "temperature": 0.2,
        }

    def _call_agent(self, cfg: AgentConfig, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(cfg.endpoint, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8", errors="ignore")
                status = resp.status
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"{cfg.name} API HTTP {e.code}: {body[:300]}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"{cfg.name} API 调用失败: {e.reason}") from e

        if not (200 <= status < 300):
            raise RuntimeError(f"{cfg.name} API HTTP {status}: {body[:300]}")

        try:
            return json.loads(body)
        except ValueError:
            return {"raw_text": body}

    def _synthesize(self, case: Dict[str, Any], departments: List[str], results: Dict[str, Any]) -> Dict[str, Any]:
        chief = case.get("chief_complaint", "")
        high_risk_flags: List[str] = []
        for signal in ["高龄", "心衰", "肾衰", "凝血", "休克", "脓毒症"]:
            if signal in json.dumps(case, ensure_ascii=False):
                high_risk_flags.append(signal)

        return {
            "chief_complaint": chief,
            "departments_called": [DEPARTMENT_LABELS[d] for d in departments],
            "mandatory_anesthesiology_called": "anesthesiology" in departments,
            "risk_flags": high_risk_flags,
            "next_step": "建议由 Headmaster Agent 做终决策（手术/非手术路径 + 预后 + 沟通方案）",
            "notes": f"共汇总 {len(results)} 个科室结果。",
        }


def load_knowledge_bundle(knowledge_dir: Path) -> str:
    """读取 knowledge 下所有文本资料并拼接。"""
    if not knowledge_dir.exists() or not knowledge_dir.is_dir():
        return ""

    chunks: List[str] = []
    for p in sorted(knowledge_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".md", ".txt", ".json", ".yaml", ".yml"}:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = p.read_text(encoding="utf-8", errors="ignore")
        text = text.strip()
        if not text:
            continue
        chunks.append(f"\n### 来源: {p.as_posix()}\n{text[:6000]}")

    return "\n".join(chunks)


def load_configs_from_env() -> Dict[str, AgentConfig]:
    """从环境变量加载各科 API 配置。"""
    # 示例：ANESTHESIOLOGY_AGENT_URL / ANESTHESIOLOGY_AGENT_KEY
    mapping: List[Tuple[str, str]] = [
        ("cardiology", "CARDIOLOGY_AGENT_URL"),
        ("cardiac_surgery", "CARDIAC_SURGERY_AGENT_URL"),
        ("neuro", "NEURO_AGENT_URL"),
        ("general_surgery", "GENERAL_SURGERY_AGENT_URL"),
        ("orthopedics", "ORTHOPEDICS_AGENT_URL"),
        ("anesthesiology", "ANESTHESIOLOGY_AGENT_URL"),
    ]

    configs: Dict[str, AgentConfig] = {}
    for dept, url_key in mapping:
        endpoint = os.getenv(url_key)
        if not endpoint:
            continue
        key = os.getenv(url_key.replace("_URL", "_KEY"))
        configs[dept] = AgentConfig(name=dept, endpoint=endpoint, api_key=key)
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description="MDT 调度 Agent")
    parser.add_argument("--case-json", required=True, help="病例 JSON 文件路径")
    parser.add_argument("--knowledge-dir", default="knowledge", help="knowledge 目录")
    args = parser.parse_args()

    case_path = Path(args.case_json)
    case = json.loads(case_path.read_text(encoding="utf-8"))

    configs = load_configs_from_env()
    if "anesthesiology" not in configs:
        raise RuntimeError("必须配置 ANESTHESIOLOGY_AGENT_URL（麻醉科必调）")

    mdt = MDTAgent(configs=configs)
    result = mdt.run(case=case, knowledge_dir=Path(args.knowledge_dir))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
