"""Pluggable LLM root-cause classification.

Bring your own model: pass any callable ``fn(system, text, image_paths) ->
str`` to ``diagnose``, or use the built-in Anthropic / OpenAI adapters. The
response is parsed into a ranked list of taxonomy categories.
"""

from __future__ import annotations

import base64
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from .prompt import build_prompt
from .taxonomy import DEFAULT_TAXONOMY

LLMFn = Callable[[str, str, Sequence[str]], str]


@dataclass
class Diagnosis:
    ranked: list[tuple[int, str]]  # [(category_number, category_name), ...] best first
    raw_response: str

    @property
    def top1(self) -> tuple[int, str] | None:
        return self.ranked[0] if self.ranked else None


def parse_ranked_categories(text: str, categories: list[str]) -> list[tuple[int, str]]:
    """Extract ``Rank k: <number>. <name>`` style conclusions from a response.

    Falls back to any ``Category N`` / leading ``N.`` mentions in order of
    appearance. Category numbers are 1-based indices into ``categories``.
    """
    ranked: list[tuple[int, str]] = []
    seen: set[int] = set()

    def add(num: int) -> None:
        if 1 <= num <= len(categories) and num not in seen:
            seen.add(num)
            ranked.append((num, categories[num - 1]))

    for m in re.finditer(r"Rank\s*(\d)\s*[:\-–]\s*.*?(\d{1,2})", text, re.IGNORECASE):
        add(int(m.group(2)))
    if not ranked:
        for m in re.finditer(r"(?:Category|Cat\.?)\s*#?(\d{1,2})", text, re.IGNORECASE):
            add(int(m.group(1)))
    if not ranked:
        for m in re.finditer(r"^\s*(\d{1,2})[.):]", text, re.MULTILINE):
            add(int(m.group(1)))
    return ranked[:3]


def diagnose(
    llm: LLMFn,
    image_paths: Sequence[str | Path],
    categories: list[str] | None = None,
    rubric: str | None = None,
    data_description: str | None = None,
    extra_context: str = "",
) -> Diagnosis:
    """Run one root-cause classification query.

    ``llm`` is called as ``llm(system_prompt, user_text, image_paths)`` and
    must return the model's text response.
    """
    cats = categories if categories is not None else DEFAULT_TAXONOMY
    system, user = build_prompt(categories, rubric, data_description)
    if extra_context:
        user = f"{user}\n\nAdditional context:\n{extra_context}"
    response = llm(system, user, [str(p) for p in image_paths])
    return Diagnosis(ranked=parse_ranked_categories(response, cats), raw_response=response)


# ---------------------------------------------------------------------------
# Built-in adapters
# ---------------------------------------------------------------------------


def _read_image_b64(path: str) -> tuple[str, str]:
    media = mimetypes.guess_type(path)[0] or "image/png"
    return media, base64.standard_b64encode(Path(path).read_bytes()).decode()


def anthropic_llm(model: str = "claude-sonnet-5", max_tokens: int = 2048) -> LLMFn:
    """Adapter for the Anthropic API (``pip install anthropic``,
    ``ANTHROPIC_API_KEY`` in the environment)."""
    import anthropic

    client = anthropic.Anthropic()

    def fn(system: str, text: str, image_paths: Sequence[str]) -> str:
        content: list[dict] = []
        for p in image_paths:
            media, data = _read_image_b64(p)
            content.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media, "data": data},
                }
            )
        content.append({"type": "text", "text": text})
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": content}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")

    return fn


def openai_llm(model: str = "gpt-5.2", max_tokens: int = 2048) -> LLMFn:
    """Adapter for the OpenAI API (``pip install openai``, ``OPENAI_API_KEY``
    in the environment)."""
    from openai import OpenAI

    client = OpenAI()

    def fn(system: str, text: str, image_paths: Sequence[str]) -> str:
        content: list[dict] = []
        for p in image_paths:
            media, data = _read_image_b64(p)
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:{media};base64,{data}"}}
            )
        content.append({"type": "text", "text": text})
        resp = client.chat.completions.create(
            model=model,
            max_completion_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
        )
        return resp.choices[0].message.content or ""

    return fn
