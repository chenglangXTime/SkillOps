"""DeepSeek API client — OpenAI-compatible, zero third-party deps, usage-tracked.

密钥加载顺序：环境变量 DEEPSEEK_API_KEY → ~/.env → ./.env。
密钥永不打印、永不写入仓库或 manifest（manifest 只记录模型名与 token 数）。
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"


class CredentialError(RuntimeError):
    pass


def load_api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key
    for p in (Path.home() / ".env", Path.cwd() / ".env"):
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("DEEPSEEK_API_KEY="):
                return line.split("=", 1)[1].strip().strip("'\"")
    raise CredentialError(
        "DEEPSEEK_API_KEY not found. Add it to ~/.env "
        "(see https://platform.deepseek.com/api_keys)."
    )


class DeepSeekClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = 120,
        max_retries: int = 3,
        temperature: float = 0.3,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self._key = load_api_key()
        self.usage = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def chat(self, messages: list[dict], max_tokens: int = 1024,
             temperature: float | None = None) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._key}",
            },
            method="POST",
        )
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                u = body.get("usage", {})
                self.usage["calls"] += 1
                self.usage["prompt_tokens"] += u.get("prompt_tokens", 0)
                self.usage["completion_tokens"] += u.get("completion_tokens", 0)
                self.usage["total_tokens"] += u.get("total_tokens", 0)
                return body["choices"][0]["message"]["content"]
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                    KeyError, json.JSONDecodeError) as e:
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"DeepSeek API failed after {self.max_retries} tries: {last_err}")
