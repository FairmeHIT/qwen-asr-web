from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Optional


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_TIMEOUT = 120


@dataclass(frozen=True)
class SummaryResult:
    provider: str
    base_url: str
    model: str
    summary: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


class LLMService:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL") or DEFAULT_MODEL
        self.provider = provider or os.environ.get("LLM_PROVIDER") or "deepseek"
        self.timeout = int(os.environ.get("LLM_TIMEOUT", str(timeout)))

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def summarize(
        self,
        text: str,
        instruction: str = "",
        max_input_chars: int = 60000,
    ) -> SummaryResult:
        text = (text or "").strip()
        if not text:
            raise ValueError("Text is empty.")
        if not self.api_key:
            raise RuntimeError("LLM_API_KEY or DEEPSEEK_API_KEY is not configured.")

        clipped = text[:max_input_chars]
        if len(text) > max_input_chars:
            clipped += "\n\n[文本过长，已截断用于要点提炼。]"

        user_instruction = instruction.strip() or (
            "请基于以下转录文本提炼要点。要求：\n"
            "1. 用中文输出。\n"
            "2. 先给出 5-10 条核心要点。\n"
            "3. 再列出待办事项、关键结论和可能需要复核的信息。\n"
            "4. 保持客观，不添加原文没有的信息。"
        )

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是严谨的会议纪要和音视频转写整理助手。",
                },
                {
                    "role": "user",
                    "content": f"{user_instruction}\n\n转录文本：\n{clipped}",
                },
            ],
            "temperature": 0.2,
        }

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM request failed: HTTP {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc.reason}") from exc

        obj: dict[str, Any] = json.loads(body)
        choices = obj.get("choices") or []
        if not choices:
            raise RuntimeError("LLM response has no choices.")
        message = choices[0].get("message") or {}
        summary = str(message.get("content") or "").strip()
        if not summary:
            raise RuntimeError("LLM response is empty.")

        return SummaryResult(
            provider=self.provider,
            base_url=self.base_url,
            model=self.model,
            summary=summary,
        )
