#!/usr/bin/env python3
"""Competition inference script with strict structured logs."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

try:
    from .server.pii_redaction_env_environment import RedactionEnvironment
    from .models import ActionType, RedactionAction
except ImportError:
    from server.pii_redaction_env_environment import RedactionEnvironment
    from models import ActionType, RedactionAction
from dotenv import load_dotenv
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
BENCHMARK = os.getenv("BENCHMARK") or "pii-redaction-env"

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
OPENAI_SEED = int(os.getenv("OPENAI_SEED", "42"))
INFERENCE_MAX_STEPS = int(os.getenv("INFERENCE_MAX_STEPS", "100"))
REQUEST_TIMEOUT_S = float(os.getenv("REQUEST_TIMEOUT_S", "90"))

TASKS = ["gdpr_contract_easy", "hipaa_medical_medium", "security_logs_hard"]

AUTH_BLOCK_REASON: Optional[str] = None
BACKEND_BLOCK_REASON: Optional[str] = None

LABEL_PATTERNS: Dict[str, List[str]] = {
    "EMAIL": [r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"],
    "PHONE": [
        r"\b(\+?1[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}\b",
        r"\b\d{3}[-.]\d{4}\b",
    ],
    "SSN": [r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b"],
    "DOB": [
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{2}/\d{2}/\d{4}\b",
        r"\b\d{2}-\d{2}-\d{4}\b",
    ],
    "NAME": [],
    "ADDRESS": [
        r"\d+\s+[A-Za-z]+(?:\s+[A-Za-z]+){1,4}(?:\s+(?:St|Ave|Blvd|Rd|Dr|Ln|Way|Ct|Terrace|Court|Place|Drive|Road|Street|Lane|Boulevard|Avenue)\.?)*(?:,\s*(?:Apt|Suite|Unit|Floor|Fl)\.?\s*[\w-]+)?",
    ],
}

PII_REGEX_PATTERNS = [pattern for patterns in LABEL_PATTERNS.values() for pattern in patterns]


def _is_gemma4_nvidia_model() -> bool:
    return MODEL_NAME.strip().lower() == "google/gemma-4-31b-it"


def _chat_completion_kwargs(prompt: str, use_json_mode: bool) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "seed": OPENAI_SEED,
        "timeout": REQUEST_TIMEOUT_S,
    }

    if _is_gemma4_nvidia_model():
        kwargs["max_tokens"] = int(os.getenv("MAX_TOKENS", "512"))
        kwargs["top_p"] = float(os.getenv("TOP_P", "0.95"))
        kwargs["temperature"] = float(os.getenv("TEMPERATURE", "0"))
        enable_thinking = os.getenv("ENABLE_THINKING", "0") in {"1", "true", "True"}
        kwargs["extra_body"] = {
            "chat_template_kwargs": {
                "enable_thinking": enable_thinking,
            }
        }
    elif use_json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    return kwargs


def _sanitize_api_key(value: Optional[str]) -> str:
    token = (value or "").strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    return token


def _is_auth_error(message: str) -> bool:
    low = message.lower()
    return "401" in low or "unauthorized" in low or "authentication failed" in low


def _is_server_error(message: str) -> bool:
    low = message.lower()
    return "500" in low or "internal server error" in low or "enginecore" in low


def _is_timeout_error(message: str) -> bool:
    low = message.lower()
    return "timeout" in low or "timed out" in low or "read timeout" in low


def _is_degraded_error(message: str) -> bool:
    low = message.lower()
    return "degraded" in low and "cannot be invoked" in low


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    done_val = str(done).lower()
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def _clamp_reward(raw_reward: float) -> float:
    """Clamp raw reward from [-1, 1] into [0, 1] for competition logs."""
    return round(max(0.0, min(1.0, (raw_reward + 1.0) / 2.0)), 4)


def _build_prompt(obs) -> str:
    return f"""
You are a GDPR/HIPAA compliance analyst performing PII redaction on a document.

Task: identify and redact ALL personally identifiable information using precise
absolute character indices. PII types: EMAIL, PHONE, SSN, NAME, ADDRESS, DOB.

Label guidance:
- a word or full name of a person -> NAME
- a date in any format -> DOB
- a street address -> ADDRESS
- an email address -> EMAIL
- a phone number -> PHONE
- a social security number -> SSN
- NAME detection: scan for full names near contextual anchors such as member,
  patient, customer, contractor, tenant, applicant, owner, user, caller, she,
  he, her, and his. Redact the NAME before associated contact details.

IMPORTANT — Obfuscated PII patterns to watch for:
- Emails: "john dot smith at gmail dot com", "j_doe at mail dot net"
- Phones: "five five five-0199", "area code four-one-five then 555-0187"
- SSNs: "three four five dash six seven dash eight nine zero one"
- DOBs: "nineteen eighty-five march twelve", "oh-three slash fifteen slash seventy-eight"

IMPORTANT — Span boundaries for obfuscated PII:
When PII is written out in words, the span MUST start at the FIRST word of the
obfuscated phrase and end at the LAST character of the value, including all
written-out connectors. Do NOT start the span at the first digit.

WRONG: "area code four-one-five, then 555-0187" -> start at "555"
RIGHT: "area code four-one-five, then 555-0187" -> start at "area"

WRONG: "john dot smith at gmail dot com" -> start at "smith"
RIGHT: "john dot smith at gmail dot com" -> start at "john"

DO NOT REDACT: Company names (Apple, Google, Amazon, Microsoft, Samsung, Meta),
product codes, project codenames (Falcon, Orion), infrastructure terms, or
organizational references. Only redact personal identifiers.

Current window (chars {obs.cursor_position}-{obs.cursor_position + len(obs.visible_text)}):
---
{obs.visible_text}
---
Progress: {obs.progress_pct:.0%} | Already redacted: {obs.redacted_spans}
Previous actions: {obs.previous_actions}

Worked example for absolute indexing:
1. cursor_position = 50
2. visible_text = "Patient John Smith lived"
3. The word "John" starts at relative offset 8 in visible_text
4. absolute_start = 50 + 8 = 58
5. If "John" is the target span, its absolute indices are start=58 and end=62

ADDRESS example:
1. visible_text = "...resides at 90 Birchwood Lane. Patient..."
2. cursor_position = 0
3. The address starts at "90" and ends after "Lane".
4. Count from the first digit of the street number and stop before punctuation.

Return strict JSON only, one of:
{{"action": "REDACT", "start": <abs_start>, "end": <abs_end>, "label": "<PII_TYPE>"}}
{{"action": "NEXT_CHUNK"}}
{{"action": "SKIP"}}
{{"action": "FINISH"}}

Rules:
- REDACT: mark one PII span using absolute offsets and always include its label.
  Formula: absolute_index = cursor_position + relative_offset_in_visible_text.
- Example: if cursor_position=50, visible_text="Patient John Smith lived", and "John" starts at relative offset 8,
    then absolute_start = 50 + 8 = 58 and absolute_end = 62.
- Use the label guidance above when choosing REDACT labels; wrong labels are penalized.
- NEXT_CHUNK: advance to next unseen section. Use after all PII in current window is handled.
- SKIP: no PII found in current window, equivalent to acknowledging and waiting.
- FINISH: only when ALL of the following are true:
    1. progress_pct >= 1.0 (you have scrolled through the entire document), AND
    2. remaining_entities == 0 in the last reward signal, AND
    3. You have explicitly looked for every PII type: NAME, DOB, EMAIL, PHONE, SSN, ADDRESS.
    Do NOT call FINISH just because no new PII is visible in the current window.
- Do NOT re-redact spans that appear in already_redacted — move forward instead.
- Prefer tight, precise spans over broad ones.
- If uncertain whether something is PII, use NEXT_CHUNK for more context.
""".strip()


def _action_to_string(action: RedactionAction) -> str:
    if action.action_type == ActionType.REDACT:
        return f"REDACT({action.start},{action.end})"
    return action.action_type.value


def _label_from_text(text: str) -> str:
    candidate = (text or "").strip()
    if re.fullmatch(LABEL_PATTERNS["EMAIL"][0], candidate, flags=re.IGNORECASE):
        return "EMAIL"
    if re.fullmatch(LABEL_PATTERNS["SSN"][0], candidate, flags=re.IGNORECASE):
        return "SSN"
    if re.fullmatch(LABEL_PATTERNS["PHONE"][0], candidate, flags=re.IGNORECASE):
        return "PHONE"
    if re.fullmatch(LABEL_PATTERNS["DOB"][0], candidate, flags=re.IGNORECASE):
        return "DOB"
    return "NAME"


def _coerce_action(payload: Dict[str, Any], obs) -> Tuple[RedactionAction, Optional[str]]:
    if "action_type" not in payload and "action" in payload:
        payload = {**payload, "action_type": payload["action"]}
    if "action_type" not in payload:
        return RedactionAction(action_type=ActionType.NEXT_CHUNK), "missing_action_type"

    action_value = str(payload.get("action_type", "")).upper()
    if action_value == ActionType.REDACT.value:
        try:
            raw_start = int(payload.get("start"))
            raw_end = int(payload.get("end"))
            label_value = payload.get("label")
            snapped_start, snapped_end = _snap_redact_span(
                obs,
                raw_start,
                raw_end,
                label=str(label_value) if label_value else None,
            )
            if not label_value:
                rel_start = max(0, snapped_start - int(obs.cursor_position))
                rel_end = max(rel_start + 1, snapped_end - int(obs.cursor_position))
                label_value = _label_from_text((obs.visible_text or "")[rel_start:rel_end])
            payload = {**payload, "label": label_value, "start": snapped_start, "end": snapped_end}
        except Exception:
            pass

    try:
        return RedactionAction(**payload), None
    except Exception as exc:
        return RedactionAction(action_type=ActionType.NEXT_CHUNK), str(exc)


def _snap_redact_span(obs, raw_start: int, raw_end: int, label: Optional[str] = None) -> Tuple[int, int]:
    text = obs.visible_text or ""
    if not text:
        return raw_start, raw_end

    patterns = LABEL_PATTERNS.get(label, []) if label else []
    if not patterns:
        patterns = PII_REGEX_PATTERNS

    window_start = int(obs.cursor_position)
    window_end = window_start + len(text)

    rel_start = raw_start - window_start
    rel_end = raw_end - window_start

    scan_left = max(0, min(rel_start, rel_end) - 20)
    scan_right = min(len(text), max(rel_start, rel_end) + 20)
    if scan_right <= scan_left:
        return raw_start, raw_end

    scan_text = text[scan_left:scan_right]
    start_candidates: List[int] = []
    end_candidates: List[int] = []

    for pattern in patterns:
        for match in re.finditer(pattern, scan_text, flags=re.IGNORECASE):
            start_candidates.append(window_start + scan_left + match.start())
            end_candidates.append(window_start + scan_left + match.end())

    snapped_start = raw_start
    snapped_end = raw_end

    if start_candidates:
        snapped_start = min(start_candidates, key=lambda candidate: abs(candidate - raw_start))
    if end_candidates:
        snapped_end = min(end_candidates, key=lambda candidate: abs(candidate - raw_end))

    # Keep span sane and inside the document bounds.
    snapped_start = max(0, min(snapped_start, window_end - 1))
    snapped_end = max(snapped_start + 1, min(snapped_end, window_end))

    return snapped_start, snapped_end


def _extract_json_object(text: str) -> Optional[str]:
    candidate = (text or "").strip()
    if not candidate:
        return None

    # Handle fenced markdown blocks.
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate, flags=re.IGNORECASE)
        candidate = candidate.strip()

    # Fast path if already a JSON object.
    if candidate.startswith("{") and candidate.endswith("}"):
        return candidate

    # Extract first balanced {...} block from verbose output.
    start = candidate.find("{")
    if start == -1:
        return None

    depth = 0
    for idx in range(start, len(candidate)):
        ch = candidate[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return candidate[start : idx + 1]
    return None


def _parse_action_payload(raw_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    json_text = _extract_json_object(raw_text)
    if not json_text:
        return None, "no_json_object"
    try:
        payload = json.loads(json_text)
        if not isinstance(payload, dict):
            return None, "json_not_object"
        return payload, None
    except Exception as exc:
        return None, str(exc)


def _first_match(pattern: str, text: str) -> Optional[Tuple[int, int]]:
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if not m:
        return None
    return m.start(), m.end()


def _fallback_action(obs) -> RedactionAction:
    # Deterministic local policy used when provider auth/config is invalid.
    for label, patterns in LABEL_PATTERNS.items():
        if not patterns:
            continue
        for pattern in patterns:
            rel_span = _first_match(pattern, obs.visible_text)
            if rel_span is not None:
                start = obs.cursor_position + rel_span[0]
                end = obs.cursor_position + rel_span[1]
                return RedactionAction(
                    action_type=ActionType.REDACT,
                    start=start,
                    end=end,
                    label=label,
                )

    if obs.progress_pct >= 0.98:
        return RedactionAction(action_type=ActionType.FINISH)
    return RedactionAction(action_type=ActionType.NEXT_CHUNK)


def _next_action(client: OpenAI, obs) -> Tuple[RedactionAction, Optional[str]]:
    global AUTH_BLOCK_REASON, BACKEND_BLOCK_REASON

    if AUTH_BLOCK_REASON is not None:
        return _fallback_action(obs), AUTH_BLOCK_REASON
    if BACKEND_BLOCK_REASON is not None:
        return _fallback_action(obs), BACKEND_BLOCK_REASON

    trailing_next_chunks = 0
    for previous_action in reversed(obs.previous_actions or []):
        if previous_action == "NEXT_CHUNK":
            trailing_next_chunks += 1
        else:
            break

    if trailing_next_chunks >= 3 and obs.cursor_position >= obs.document_length - len(obs.visible_text):
        return RedactionAction(action_type=ActionType.FINISH), "cursor_clamped"

    prompt = _build_prompt(obs)

    # First attempt with strict JSON mode.
    try:
        response = client.chat.completions.create(**_chat_completion_kwargs(prompt, use_json_mode=True))
        text = response.choices[0].message.content or ""
        payload, parse_error = _parse_action_payload(text)
        if payload is None:
            return _fallback_action(obs), parse_error
        return _coerce_action(payload, obs)
    except Exception as exc:
        msg = str(exc)
        if _is_auth_error(msg):
            AUTH_BLOCK_REASON = "authentication_failed"
            return _fallback_action(obs), AUTH_BLOCK_REASON
        if _is_degraded_error(msg):
            BACKEND_BLOCK_REASON = "backend_degraded"
            return _fallback_action(obs), BACKEND_BLOCK_REASON
        if _is_server_error(msg):
            # Retry once without response_format for providers that fail on JSON mode.
            try:
                response = client.chat.completions.create(**_chat_completion_kwargs(prompt, use_json_mode=False))
                text = response.choices[0].message.content or ""
                payload, parse_error = _parse_action_payload(text)
                if payload is None:
                    return _fallback_action(obs), parse_error
                return _coerce_action(payload, obs)
            except Exception as retry_exc:
                retry_msg = str(retry_exc)
                if _is_auth_error(retry_msg):
                    AUTH_BLOCK_REASON = "authentication_failed"
                    return _fallback_action(obs), AUTH_BLOCK_REASON
                if _is_degraded_error(retry_msg):
                    BACKEND_BLOCK_REASON = "backend_degraded"
                    return _fallback_action(obs), BACKEND_BLOCK_REASON
                if _is_server_error(retry_msg):
                    BACKEND_BLOCK_REASON = "backend_500"
                    return _fallback_action(obs), BACKEND_BLOCK_REASON
                return _fallback_action(obs), retry_msg
        if _is_timeout_error(msg):
            # Retry once without JSON mode to reduce parse overhead.
            try:
                response = client.chat.completions.create(**_chat_completion_kwargs(prompt, use_json_mode=False))
                text = response.choices[0].message.content or ""
                payload, parse_error = _parse_action_payload(text)
                if payload is not None:
                    return _coerce_action(payload, obs)
            except Exception:
                pass
        return _fallback_action(obs), msg


def run_task(client: OpenAI, task_id: str) -> None:
    env = RedactionEnvironment(task_id=task_id)
    rewards: List[float] = []
    steps = 0
    success = False
    score = 0.0

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset()
        for step in range(1, INFERENCE_MAX_STEPS + 1):
            if obs.done:
                break

            action, action_error = _next_action(client, obs)
            action_str = _action_to_string(action)

            obs, reward, done, info = env.step(action)
            steps = step
            clamped_reward = _clamp_reward(reward.raw_total)
            rewards.append(clamped_reward)

            error_value: Optional[str] = action_error
            if info.get("invalid_action"):
                error_value = "invalid_action"

            log_step(step=step, action=action_str, reward=clamped_reward, done=done, error=error_value)

            if done:
                break

        grade = env.grade()
        score = max(0.0, min(1.0, float(grade.score)))
        success = bool(grade.success)
    except Exception:
        success = False
    finally:
        log_end(success=success, steps=steps, score=score, rewards=rewards)


def main() -> None:
    global AUTH_BLOCK_REASON, BACKEND_BLOCK_REASON

    resolved_token = _sanitize_api_key(HF_TOKEN)
    if not resolved_token:
        raise RuntimeError("HF_TOKEN (or OPENAI_API_KEY) is required.")

    AUTH_BLOCK_REASON = None
    BACKEND_BLOCK_REASON = None
    client = OpenAI(base_url=API_BASE_URL, api_key=resolved_token)
    for task_id in TASKS:
        run_task(client, task_id)


if __name__ == "__main__":
    main()
