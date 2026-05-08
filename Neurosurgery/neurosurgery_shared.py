from __future__ import annotations

import json
import sys
import re
import socket
import ssl
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

_LLM_HELPER_ROOT = Path(__file__).resolve().parent.parent
if str(_LLM_HELPER_ROOT) not in sys.path:
    sys.path.insert(0, str(_LLM_HELPER_ROOT))

from llm_gateway import DEFAULT_LLM_GATEWAY_URL, build_auth_headers, get_llm_model_id


GENAI_API_URL = DEFAULT_LLM_GATEWAY_URL


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model: str


MODEL_CONFIGS = {
    "gpt_5_2": ModelConfig(
        name="gpt-4o",
        model=get_llm_model_id("gpt_5_2", "gpt-4o"),
    ),
    "deepseek_v3_2": ModelConfig(
        name="deepseek-chat",
        model=get_llm_model_id("deepseek_r1", "deepseek-chat"),
    ),
    "deepseek_r1": ModelConfig(
        name="deepseek-chat",
        model=get_llm_model_id("deepseek_r1", "deepseek-chat"),
    ),
    "qwen3": ModelConfig(
        name="qwen-max",
        model=get_llm_model_id("qwen3", "qwen-max"),
    ),
}


class GenAIChatClient:
    def __init__(self, api_url: str = GENAI_API_URL) -> None:
        self.api_url = api_url

    def chat_json(
        self,
        config: ModelConfig,
        prompt: str,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        payload = {
            "model": config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ],
                }
            ],
            "temperature": temperature,
            "n": 1,
            "stream": False,
            "presence_penalty": 0,
            "frequency_penalty": 0,
        }

        req = request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "accept": "application/json",
                **build_auth_headers(),
                "Content-Type": "application/json",
            },
            method="POST",
        )

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with request.urlopen(req, timeout=60) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except (error.URLError, ssl.SSLError, TimeoutError, socket.timeout) as exc:
                last_error = exc
                if attempt == 2:
                    break
                time.sleep(1.5 * (attempt + 1))

        raise RuntimeError(
            f"Neurosurgery API request failed after retries: {last_error}"
        ) from last_error

    def extract_text(self, response: dict[str, Any]) -> str:
        choices = response.get("choices", [])
        if not choices:
            raise ValueError(f"API response does not contain choices. Response excerpt: {json.dumps(response, ensure_ascii=False)[:500]}")

        message = choices[0].get("message", {})
        content = message.get("content")

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
            return "".join(text_parts).strip()

        raise ValueError("Unsupported API response format.")

    def extract_json(self, response: dict[str, Any]) -> dict[str, Any]:
        return self._parse_json_text(self.extract_text(response))

    def _parse_json_text(self, text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char not in "[{":
                continue
            try:
                payload, _ = decoder.raw_decode(text[index:])
                if isinstance(payload, dict):
                    return payload
            except json.JSONDecodeError:
                continue

        raise ValueError(f"Model did not return valid JSON. Raw output excerpt: {text[:500]}")


