#!/usr/bin/env python3
"""Reproducible baseline inference runner for pii-redaction-env."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from openai import OpenAI

from src.environment import RedactionEnvironment
from src.models import ActionType, RedactionAction
from src.tasks import TASKS

ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")
API_BASE_URL = os.getenv("API_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta/llama-3.1-70b-instruct")
BENCHMARK = os.getenv("BENCHMARK", "emvlight")
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
DEFAULT_SEED = int(os.getenv("OPENAI_SEED", "42"))


def _build_prompt(obs) -> str:
    return f"""
You are a compliance analyst redacting PII under GDPR/HIPAA.

Task: redact all personal identifiers (EMAIL, PHONE, SSN, NAME, ADDRESS, DOB)
with precise absolute character indices in the full document.

Current window ({obs.cursor_position}-{obs.cursor_position + len(obs.visible_text)}):
---
{obs.visible_text}
---
Already redacted: {obs.redacted_spans}
Previous actions: {obs.previous_actions}

Return strict JSON only, in one of these forms:
{{"action": "REDACT", "start": 120, "end": 135}}
{{"action": "NEXT_CHUNK"}}
{{"action": "FINISH"}}

Rules:
- Use absolute indices against full document text.
- Prefer precision over broad spans.
- If uncertain, use NEXT_CHUNK.
""".strip()


def _coerce_action(payload: Dict[str, Any]) -> RedactionAction:
    if "action_type" not in payload and "action" in payload:
        payload = {**payload, "action_type": payload["action"]}
    if "action_type" not in payload:
        return RedactionAction(action_type=ActionType.NEXT_CHUNK)

    try:
        return RedactionAction(**payload)
    except Exception:
        return RedactionAction(action_type=ActionType.NEXT_CHUNK)


def run_baseline(task_id: str, model: str = MODEL_NAME) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to run inference baseline.")

    client = OpenAI(base_url=API_BASE_URL, api_key=api_key)
    env = RedactionEnvironment(task_id=task_id)

    obs = env.reset()
    rewards = []

    while not obs.done:
        prompt = _build_prompt(obs)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=DEFAULT_TEMPERATURE,
                seed=DEFAULT_SEED,
            )
            text = response.choices[0].message.content or "{}"
            action_payload = json.loads(text)
            action = _coerce_action(action_payload)
        except Exception:
            action = RedactionAction(action_type=ActionType.NEXT_CHUNK)

        obs, reward, done, _ = env.step(action)
        rewards.append(reward.total)

        if done:
            break

    grade = env.grade()
    result = {
        "task_id": task_id,
        "score": grade.score,
        "f1": grade.f1_final,
        "precision": grade.precision,
        "recall": grade.recall,
        "utility": grade.utility_score,
        "success": grade.success,
        "success_threshold": TASKS[task_id]["success_threshold"],
        "total_reward": round(sum(rewards), 4),
        "steps": len(rewards),
    }
    return result


def main() -> None:
    task_ids = ["gdpr_contract_easy", "hipaa_medical_medium", "security_logs_hard"]
    all_results = [run_baseline(task_id=t) for t in task_ids]

    print("=== Baseline Results ===")
    for row in all_results:
        print(
            f"{row['task_id']}: score={row['score']:.3f} "
            f"f1={row['f1']:.3f} precision={row['precision']:.3f} "
            f"recall={row['recall']:.3f} utility={row['utility']:.3f} "
            f"success={row['success']} threshold={row['success_threshold']:.2f} "
            f"steps={row['steps']} total_reward={row['total_reward']:.2f}"
        )


if __name__ == "__main__":
    main()
