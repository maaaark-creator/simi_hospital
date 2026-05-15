from __future__ import annotations

import json
import os
import re
import socket
import ssl
import time
from dataclasses import dataclass
from typing import Any
from urllib import error, request

from llm_runtime import get_api_key_for_model_key, get_gateway_url


GENAI_API_URL = get_gateway_url()


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model: str


MODEL_CONFIGS = {
    "gpt_5_2": ModelConfig(
        name="deepseek-v3:671b",
        model="deepseek-v3:671b",
    ),
    "deepseek_v3_2": ModelConfig(
        name="deepseek-v3:671b",
        model="deepseek-v3:671b",
    ),
    "deepseek_r1": ModelConfig(
        name="deepseek-r1:671b",
        model="deepseek-r1:671b",
    ),
    "qwen3": ModelConfig(
        name="deepseek-v3:671b",
        model="deepseek-v3:671b",
    ),
}


class GenAIChatClient:
    def __init__(self, api_url: str = GENAI_API_URL, api_key: str | None = None) -> None:
        self.api_url = api_url
        self.api_key = str(api_key or "").strip()

    def chat_json(
        self,
        config: ModelConfig,
        prompt: str,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        api_key = self.api_key or get_api_key_for_model_key(
            next((key for key, candidate in MODEL_CONFIGS.items() if candidate == config), "deepseek_v3_2")
        )
        if not api_key:
            raise ValueError("Missing DeepSeek API key environment variable for Neurosurgery.")
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
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with request.urlopen(req, timeout=60) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(
                    f"Neurosurgery API HTTP {exc.code}: {exc.reason}. Response: {body[:1200]}"
                )
                if exc.code == 429 and attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                break
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
            error_payload = response.get("error")
            if isinstance(error_payload, dict):
                code = str(error_payload.get("code") or "").strip()
                message = str(error_payload.get("message") or error_payload).strip()
                raise ValueError(
                    f"API response does not contain choices. error_code={code or 'unknown'} message={message}"
                )
            raise ValueError(f"API response does not contain choices. keys={list(response.keys())[:8]}")

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
