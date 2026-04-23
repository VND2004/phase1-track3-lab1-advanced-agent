from __future__ import annotations

import json
import os
from dataclasses import dataclass
from socket import timeout as socket_timeout
from time import perf_counter
from typing import Literal
from urllib import error
from urllib import request

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import JudgeResult, QAExample, ReflectionEntry
from .utils import normalize_answer

if load_dotenv is not None:
    load_dotenv()

FIRST_ATTEMPT_WRONG = {"hp2": "London", "hp4": "Atlantic Ocean", "hp6": "Red Sea", "hp8": "Andes"}
FAILURE_MODE_BY_QID = {
    "hp2": "incomplete_multi_hop",
    "hp4": "wrong_final_answer",
    "hp6": "entity_drift",
    "hp8": "entity_drift",
}


@dataclass(frozen=True)
class RuntimeConfig:
    mode: Literal["mock", "real"] = "mock"
    provider: Literal["auto", "openai", "ollama"] = "auto"
    model: str = "gpt-4o-mini"
    base_url: str | None = None
    api_key: str | None = None
    timeout_s: int = 120
    max_retries: int = 2
    temperature: float = 0.0


def _effective_config(runtime: RuntimeConfig | None) -> RuntimeConfig:
    if runtime is not None:
        return runtime
    return RuntimeConfig(
        mode=os.getenv("REFLEXION_MODE", "mock"),
        provider=os.getenv("LLM_PROVIDER", "auto"),
        model=os.getenv("AGENT_MODEL") or os.getenv("LLM_MODEL") or "gpt-4o-mini",
        base_url=(
            os.getenv("NVIDIA_BASE_URL")
            or os.getenv("LLM_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("OLLAMA_HOST")
        ),
        api_key=os.getenv("NVIDIA_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY"),
        timeout_s=int(os.getenv("LLM_TIMEOUT_S", "120")),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
    )


def _json_post(url: str, payload: dict, headers: dict[str, str], timeout_s: int, max_retries: int) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(url=url, data=body, headers=headers, method="POST")
    attempt = 0
    while True:
        try:
            with request.urlopen(req, timeout=timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (socket_timeout, TimeoutError, error.URLError) as exc:
            if attempt >= max_retries:
                raise RuntimeError(
                    f"LLM request timed out after {attempt + 1} attempt(s). "
                    f"url={url}, timeout_s={timeout_s}. "
                    "Increase --timeout-s or reduce model latency."
                ) from exc
            attempt += 1


def _resolve_provider(cfg: RuntimeConfig) -> Literal["openai", "ollama"]:
    if cfg.provider != "auto":
        return cfg.provider
    base_url = (cfg.base_url or "").lower()
    if "11434" in base_url or "ollama" in base_url or os.getenv("OLLAMA_HOST"):
        return "ollama"
    if cfg.api_key:
        return "openai"
    return "ollama"


def _chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _call_llm(messages: list[dict[str, str]], cfg: RuntimeConfig, json_mode: bool = False) -> tuple[str, int, int]:
    provider = _resolve_provider(cfg)
    started = perf_counter()

    if provider == "openai":
        base = (
            cfg.base_url
            or os.getenv("NVIDIA_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com"
        )
        url = _chat_completions_url(base)
        headers = {"Content-Type": "application/json"}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
        elif "api.openai.com" in base:
            raise RuntimeError("OPENAI_API_KEY is required when using api.openai.com")

        payload: dict = {
            "model": cfg.model,
            "messages": messages,
            "temperature": cfg.temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        data = _json_post(url, payload, headers, cfg.timeout_s, cfg.max_retries)
        content = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {})
        token_usage = int(usage.get("total_tokens", 0))
        latency_ms = int((perf_counter() - started) * 1000)
        return content, token_usage, latency_ms

    base = (cfg.base_url or os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
    url = f"{base}/api/chat"
    headers = {"Content-Type": "application/json"}
    payload: dict = {
        "model": cfg.model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": cfg.temperature},
    }
    if json_mode:
        payload["format"] = "json"

    data = _json_post(url, payload, headers, cfg.timeout_s, cfg.max_retries)
    content = data.get("message", {}).get("content", "").strip()
    token_usage = int(data.get("prompt_eval_count", 0)) + int(data.get("eval_count", 0))
    latency_ms = int(data.get("total_duration", 0) / 1_000_000) or int((perf_counter() - started) * 1000)
    return content, token_usage, latency_ms


def _extract_json_object(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _actor_user_prompt(example: QAExample, reflection_memory: list[str]) -> str:
    context_text = "\n".join(f"- {chunk.title}: {chunk.text}" for chunk in example.context)
    reflection_text = "\n".join(f"- {item}" for item in reflection_memory) if reflection_memory else "- (none)"
    return (
        f"Question:\n{example.question}\n\n"
        f"Context:\n{context_text}\n\n"
        f"Reflection memory:\n{reflection_text}\n\n"
        "Return only final answer text."
    )


def actor_answer(
    example: QAExample,
    attempt_id: int,
    agent_type: str,
    reflection_memory: list[str],
    runtime: RuntimeConfig | None = None,
) -> tuple[str, int, int]:
    cfg = _effective_config(runtime)
    if cfg.mode == "mock":
        if example.qid not in FIRST_ATTEMPT_WRONG:
            return example.gold_answer, 0, 0
        if agent_type == "react":
            return FIRST_ATTEMPT_WRONG[example.qid], 0, 0
        if attempt_id == 1 and not reflection_memory:
            return FIRST_ATTEMPT_WRONG[example.qid], 0, 0
        return example.gold_answer, 0, 0

    messages = [
        {"role": "system", "content": ACTOR_SYSTEM.strip()},
        {"role": "user", "content": _actor_user_prompt(example, reflection_memory)},
    ]
    answer, tokens, latency_ms = _call_llm(messages, cfg, json_mode=False)
    return answer.strip(), tokens, latency_ms


def evaluator(example: QAExample, answer: str, runtime: RuntimeConfig | None = None) -> JudgeResult:
    cfg = _effective_config(runtime)
    if cfg.mode == "mock":
        if normalize_answer(example.gold_answer) == normalize_answer(answer):
            return JudgeResult(score=1, reason="Final answer matches the gold answer after normalization.")
        if normalize_answer(answer) == "london":
            return JudgeResult(
                score=0,
                reason="The answer stopped at the birthplace city and never completed the second hop to the river.",
                missing_evidence=["Need to identify the river that flows through London."],
                spurious_claims=[],
            )
        return JudgeResult(
            score=0,
            reason="The final answer selected the wrong second-hop entity.",
            missing_evidence=["Need to ground the answer in the second paragraph."],
            spurious_claims=[answer],
        )

    context_text = "\n".join(f"- {chunk.title}: {chunk.text}" for chunk in example.context)
    user_prompt = (
        f"Question:\n{example.question}\n\n"
        f"Gold answer:\n{example.gold_answer}\n\n"
        f"Predicted answer:\n{answer}\n\n"
        f"Context:\n{context_text}\n"
    )
    messages = [
        {"role": "system", "content": EVALUATOR_SYSTEM.strip()},
        {"role": "user", "content": user_prompt},
    ]
    raw_text, tokens, latency_ms = _call_llm(messages, cfg, json_mode=True)

    try:
        parsed = _extract_json_object(raw_text)
    except json.JSONDecodeError:
        parsed = {
            "score": 1 if normalize_answer(example.gold_answer) == normalize_answer(answer) else 0,
            "reason": "Fallback heuristic evaluation because evaluator JSON parsing failed.",
            "missing_evidence": [],
            "spurious_claims": [] if normalize_answer(example.gold_answer) == normalize_answer(answer) else [answer],
        }

    parsed_score = 1 if int(parsed.get("score", 0)) == 1 else 0
    return JudgeResult(
        score=parsed_score,
        reason=str(parsed.get("reason", "No reason provided.")),
        missing_evidence=[str(x) for x in parsed.get("missing_evidence", [])],
        spurious_claims=[str(x) for x in parsed.get("spurious_claims", [])],
        token_usage=tokens,
        latency_ms=latency_ms,
    )


def reflector(
    example: QAExample,
    attempt_id: int,
    judge: JudgeResult,
    runtime: RuntimeConfig | None = None,
) -> ReflectionEntry:
    cfg = _effective_config(runtime)
    if cfg.mode == "mock":
        strategy = (
            "Do the second hop explicitly: birthplace city -> river through that city."
            if example.qid == "hp2"
            else "Verify the final entity against the second paragraph before answering."
        )
        return ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=judge.reason,
            lesson="A partial first-hop answer is not enough; the final answer must complete all hops.",
            next_strategy=strategy,
        )

    user_prompt = (
        f"Question:\n{example.question}\n\n"
        f"Last failure reason:\n{judge.reason}\n\n"
        f"Missing evidence:\n{json.dumps(judge.missing_evidence, ensure_ascii=True)}\n\n"
        f"Spurious claims:\n{json.dumps(judge.spurious_claims, ensure_ascii=True)}\n"
    )
    messages = [
        {"role": "system", "content": REFLECTOR_SYSTEM.strip()},
        {"role": "user", "content": user_prompt},
    ]
    raw_text, tokens, latency_ms = _call_llm(messages, cfg, json_mode=True)

    try:
        parsed = _extract_json_object(raw_text)
    except json.JSONDecodeError:
        parsed = {
            "lesson": "Ground each hop in context and avoid stopping at intermediate entities.",
            "next_strategy": "Re-read context chunk-by-chunk and verify final entity before answering.",
        }

    return ReflectionEntry(
        attempt_id=attempt_id,
        failure_reason=judge.reason,
        lesson=str(parsed.get("lesson", "No lesson provided.")),
        next_strategy=str(parsed.get("next_strategy", "No strategy provided.")),
        token_usage=tokens,
        latency_ms=latency_ms,
    )
