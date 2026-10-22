"""Thin provider-agnostic LLM layer (Anthropic / OpenAI) with tool use + vision.

The agent works on a neutral message format:
    {"role": "user", "content": str}
    {"role": "assistant", "text": str, "tool_calls": [{"id", "name", "args"}]}
    {"role": "tool", "tool_call_id": str, "name": str, "content": str}
and each backend converts to its own wire format. No framework: the loop is
~40 lines in agent.py and easy to inspect, test and explain.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
from dataclasses import dataclass, field

from . import config


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)


def _b64(path: str) -> tuple[str, str]:
    mt = mimetypes.guess_type(path)[0] or "image/png"
    return mt, base64.b64encode(open(path, "rb").read()).decode()


class AnthropicLLM:
    name = "anthropic"

    def __init__(self, model: str = config.ANTHROPIC_MODEL):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    def _convert(self, msgs):
        out = []
        for m in msgs:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                blocks = [{"type": "text", "text": m["text"]}] if m.get("text") else []
                blocks += [{"type": "tool_use", "id": t["id"], "name": t["name"], "input": t["args"]}
                           for t in m.get("tool_calls", [])]
                out.append({"role": "assistant", "content": blocks})
            elif m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
        return out

    def chat(self, system: str, messages: list[dict], tools: list[dict], max_tokens: int = 2000) -> LLMResponse:
        r = self.client.messages.create(
            model=self.model, system=system, max_tokens=max_tokens, temperature=0,
            messages=self._convert(messages),
            tools=[{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                   for t in tools])
        text = "".join(b.text for b in r.content if b.type == "text")
        calls = [{"id": b.id, "name": b.name, "args": b.input} for b in r.content if b.type == "tool_use"]
        return LLMResponse(text, calls, {"in": r.usage.input_tokens, "out": r.usage.output_tokens})

    def vision(self, image_path: str, prompt: str, max_tokens: int = 700) -> str:
        mt, data = _b64(image_path)
        r = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, temperature=0,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": mt, "data": data}},
                {"type": "text", "text": prompt}]}])
        return "".join(b.text for b in r.content if b.type == "text")


class OpenAILLM:
    """OpenAI API, or any OpenAI-compatible server (Ollama, Gemini, vLLM ...) via OPENAI_BASE_URL."""
    name = "openai"

    def __init__(self, model: str | None = None):
        from openai import OpenAI
        self.client = OpenAI(timeout=600)  # local models can be slow
        self.model = model or os.environ.get("OPENAI_MODEL", config.OPENAI_MODEL)
        if os.environ.get("OPENAI_BASE_URL"):
            self.name = "local" if "localhost" in os.environ["OPENAI_BASE_URL"] else "openai-compatible"

    def _convert(self, system, msgs):
        out = [{"role": "system", "content": system}]
        for m in msgs:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                d = {"role": "assistant", "content": m.get("text") or None}
                if m.get("tool_calls"):
                    d["tool_calls"] = [{"id": t["id"], "type": "function",
                                        "function": {"name": t["name"], "arguments": json.dumps(t["args"])}}
                                       for t in m["tool_calls"]]
                out.append(d)
            elif m["role"] == "tool":
                out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
        return out

    def chat(self, system, messages, tools, max_tokens: int = 2000) -> LLMResponse:
        kw = {}
        if tools:  # an empty tools list is rejected by some OpenAI-compatible servers
            kw["tools"] = [{"type": "function", "function": t} for t in tools]
        r = self.client.chat.completions.create(
            model=self.model, temperature=0, max_tokens=max_tokens,
            messages=self._convert(system, messages), **kw)
        msg = r.choices[0].message
        calls = []
        for i, c in enumerate(msg.tool_calls or []):
            raw = c.function.arguments or "{}"
            try:  # small local models sometimes emit malformed JSON arguments
                args = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except Exception:
                args = {}
            calls.append({"id": c.id or f"call_{i}", "name": c.function.name, "args": args})
        u = r.usage
        return LLMResponse(msg.content or "", calls,
                           {"in": getattr(u, "prompt_tokens", 0) or 0, "out": getattr(u, "completion_tokens", 0) or 0})

    def vision(self, image_path, prompt, max_tokens: int = 700) -> str:
        mt, data = _b64(image_path)
        r = self.client.chat.completions.create(
            model=self.model, max_tokens=max_tokens, temperature=0,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mt};base64,{data}"}},
                {"type": "text", "text": prompt}]}])
        return r.choices[0].message.content or ""


def get_llm(provider: str = config.LLM_PROVIDER):
    """Returns an LLM backend, or None for offline mode."""
    if provider == "offline":
        return None
    if provider in ("auto", "anthropic") and os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicLLM()
    if provider in ("auto", "openai") and os.environ.get("OPENAI_API_KEY"):
        return OpenAILLM()
    if provider != "auto":
        raise RuntimeError(f"provider {provider!r} requested but no API key set")
    return None
