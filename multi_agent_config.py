from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "multi_agent_config.json"


DEFAULT_CONFIG: dict[str, Any] = {
    "runtime": {
        # Controls whether the patient input itself is treated as local structured data
        # or should first be structured from free text through the API.
        "mode": "local",
        # Controls the first-round specialist conclusion generation.
        # "heuristic" means local retrieval + local decision logic.
        # "api" means specialists use their model-backed decision step.
        "specialist_mode": "heuristic",
        # Controls how mdt_call decides routing/dispatch.
        # This only affects the triage desk itself, not the specialists' replies.
        "mdt_mode": "heuristic",
        # Controls whether head_doctor's final integration is generated locally
        # or synthesized by the model API.
        "final_mode": "heuristic",
        # Whether head_doctor should send unresolved items back to mdt_call.
        "use_mdt_for_uncertainty": True,
        # Whether follow-up questions dispatched by mdt_call should be answered by
        # specialist APIs. Even when mdt_mode is "heuristic", this can still be True.
        "use_api_for_clarification": True,
        # Optional specialty gate. The root demo may use this list to limit which
        # specialty agents are eligible in the workflow.
        "enabled_specialties": [
            "anesthesia",
            "cardiology",
            "hepatobiliary",
        ],
    },
    "patient_case": {
        # "builtin" uses the demo patient embedded in the script.
        # If patient_file is provided, that file is used instead.
        "source": "builtin",
        "patient_file": None,
    },
    "output": {
        "ensure_ascii": False,
        "indent": 2,
    },
    "notes": {
        "description": "Root-level config for the multi-agent MDT demo workflow.",
        "available_modes": {
            "mode": ["local", "api"],
            "specialist_mode": ["heuristic", "api"],
            "mdt_mode": ["heuristic", "api"],
            "final_mode": ["heuristic", "api"],
        },
        "field_guide": {
            "mode": "Controls whether the input patient case is local structured data or API-structured free text.",
            "specialist_mode": "Controls first-round specialist decision generation.",
            "mdt_mode": "Controls mdt_call triage logic only.",
            "final_mode": "Controls head_doctor final synthesis.",
            "use_mdt_for_uncertainty": "If true, unresolved points are sent back to mdt_call for another routing round.",
            "use_api_for_clarification": "Controls whether specialists answer mdt follow-up questions through API.",
            "enabled_specialties": "Optional list of specialty agents allowed in the workflow.",
            "patient_file": "Optional external JSON/TXT case file path.",
        },
        "available_specialties": [
            "anesthesia",
            "cardiology",
            "hepatobiliary",
        ],
    },
}


def deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_project_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        return copy.deepcopy(DEFAULT_CONFIG)

    content = path.read_text(encoding="utf-8-sig").strip()
    if not content:
        return copy.deepcopy(DEFAULT_CONFIG)

    user_config = json.loads(content)
    return deep_merge_dict(DEFAULT_CONFIG, user_config)


if __name__ == "__main__":
    print(json.dumps(DEFAULT_CONFIG, ensure_ascii=False, indent=2))
