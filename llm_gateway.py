from __future__ import annotations

import os
from pathlib import Path


DEFAULT_LLM_GATEWAY_BASE_URL = "http://hospital-llm-gateway:8080"
LLM_GATEWAY_URL_ENV = "HOSPITAL_LLM_GATEWAY_URL"
LLM_API_KEY_ENV = "HOSPITAL_LLM_API_KEY"
MODEL_ENV_BY_KEY = {
    "gpt_5_2": "HOSPITAL_LLM_MODEL_GPT_5_2",
    "deepseek_v3_2": "HOSPITAL_LLM_MODEL_DEEPSEEK_V3_2",
    "deepseek_r1": "HOSPITAL_LLM_MODEL_DEEPSEEK_R1",
    "qwen3": "HOSPITAL_LLM_MODEL_QWEN3",
    "qwen3_vl": "HOSPITAL_LLM_MODEL_QWEN3_VL",
}


class MissingLLMApiKeyError(RuntimeError):
    """Raised only when API mode actually tries to call the LLM gateway."""


def _load_local_env_file() -> None:
    env_path = Path(__file__).resolve().with_name(".env")
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


def get_llm_gateway_url() -> str:
    _load_local_env_file()
    base_url = os.environ.get(LLM_GATEWAY_URL_ENV, DEFAULT_LLM_GATEWAY_BASE_URL).strip()
    if not base_url:
        base_url = DEFAULT_LLM_GATEWAY_BASE_URL
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1/chat/completions") or normalized.endswith("/api/v1/start"):
        return normalized
    return f"{normalized}/v1/chat/completions"


def get_llm_api_key() -> str:
    _load_local_env_file()
    api_key = os.environ.get(LLM_API_KEY_ENV, "").strip()
    if not api_key:
        raise MissingLLMApiKeyError(
            "未配置 HOSPITAL_LLM_API_KEY，请配置后再启用 API 模式。"
            "如果只是本地演示，请关闭 API 模式。"
        )
    return api_key


def build_auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {get_llm_api_key()}"}


def get_llm_model_id(model_key: str, default: str) -> str:
    _load_local_env_file()
    env_name = MODEL_ENV_BY_KEY.get(model_key)
    if not env_name:
        return default
    return os.environ.get(env_name, default).strip() or default


DEFAULT_LLM_GATEWAY_URL = get_llm_gateway_url()
