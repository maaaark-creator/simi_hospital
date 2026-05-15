from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_GATEWAY_URL = "https://genaiapi.shanghaitech.edu.cn/api/v1/start"


def _load_env_file() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file()


def get_gateway_url() -> str:
    return os.getenv("HOSPITAL_LLM_GATEWAY_URL", DEFAULT_GATEWAY_URL).strip()


def get_default_model_key() -> str:
    return os.getenv("HOSPITAL_LLM_DEFAULT_MODEL_KEY", "deepseek_v3_2").strip()


def get_api_key_for_model_key(model_key: str) -> str:
    normalized_key = {
        "gpt_5_2": "deepseek_v3_2",
    }.get(model_key, model_key)

    mapping = {
        "deepseek_v3_2": os.getenv("HOSPITAL_LLM_API_KEY_DEEPSEEK_V3_2", ""),
        "deepseek_r1": os.getenv("HOSPITAL_LLM_API_KEY_DEEPSEEK_R1", ""),
        "qwen3": os.getenv("HOSPITAL_LLM_API_KEY_QWEN3", ""),
        "qwen3_vl": os.getenv("HOSPITAL_LLM_API_KEY_QWEN3_VL", ""),
        "gpt_5_2": os.getenv("HOSPITAL_LLM_API_KEY_GPT_5_2", ""),
    }
    specific = mapping.get(normalized_key, "").strip()
    if specific:
        return specific
    return os.getenv("HOSPITAL_LLM_API_KEY", "").strip()
