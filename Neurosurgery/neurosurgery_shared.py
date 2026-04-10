from __future__ import annotations

import json
import re
import socket
import ssl
import time
from dataclasses import dataclass
from typing import Any
from urllib import error, request


GENAI_API_URL = "https://genaiapi.shanghaitech.edu.cn/api/v1/start"


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model: str
    api_key: str


MODEL_CONFIGS = {
    "gpt_5_2": ModelConfig(
        name="GPT-5.2",
        model="GPT-5.2",
        api_key="bb336cff66f54e7a9d6f48b3dba97657",
    ),
    "deepseek_v3_2": ModelConfig(
        name="deepseek-v3.2",
        model="deepseek-v3:671b",
        api_key="277149ebe53440a190ee02bd66673cd1",
    ),
    "deepseek_r1": ModelConfig(
        name="deepseek-r1",
        model="deepseek-r1:671b",
        api_key="e693397f5e1e41259f8e3bef4e502ca4",
    ),
    "qwen3": ModelConfig(
        name="Qwen3",
        model="qwen-instruct",
        api_key="791e88f506f441ba8185adb3a8a9f98a",
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
                "Authorization": f"Bearer {config.api_key}",
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
            raise ValueError("API response does not contain choices.")

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
