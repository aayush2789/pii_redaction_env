#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path
from typing import Dict

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.environment import RedactionEnvironment
from src.models import ActionType, RedactionAction
from src.tasks import TASKS

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _to_action_payload(action_data: Dict) -> Dict:
    if "action_type" in action_data:
        return action_data
    if "action" in action_data:
        action_data["action_type"] = action_data.pop("action")
    return action_data


def run_baseline(task_id: str, model: str = "gpt-4o-mini"):
    env = RedactionEnvironment(task_id=task_id)
    obs = env.reset()
    total_reward = 0.0

    while not obs.done:
        prompt = f"""
You are a GDPR compliance officer reviewing documents for PII.
Your task: Redact all Personally Identifiable Information (emails, phones, names, addresses, SSNs, DOBs).

Current document view (character positions {obs.cursor_position}-{obs.cursor_position + len(obs.visible_text)}):
---
{obs.visible_text}
---
Already redacted regions: {obs.redacted_spans}

Available actions:
1. REDACT(start, end) - Redact character span from current view (absolute indices)
2. NEXT_CHUNK() - Move to next section of document
3. FINISH() - Document review complete

Guidelines:
- Use absolute character indices from the full document
- Be precise: do not redact company names, only person names and explicit personal identifiers
- When uncertain, use NEXT_CHUNK to see more context

Respond in JSON format:
{{"action": "REDACT", "start": 120, "end": 135, "reason": "Email address"}}
or
{{"action": "NEXT_CHUNK"}}
"""

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            action_data = json.loads(response.choices[0].message.content)
            payload = _to_action_payload(action_data)
            action = RedactionAction(**payload)
        except Exception as exc:
            print(f"Error parsing action: {exc}, defaulting to NEXT_CHUNK")
            action = RedactionAction(action_type=ActionType.NEXT_CHUNK)

        obs, reward, done, info = env.step(action)
        total_reward += reward.total
        print(f"Step {info.get('step', '?')}: {action.action_type.value}, Reward: {reward.total:.2f}")

        if done:
            break

    grade = env.grade()
    threshold = TASKS[task_id]["success_threshold"]
    print(f"\n=== {task_id} Results ===")
    print(f"Score: {grade.score:.3f} (Threshold: {threshold:.2f})")
    print(
        f"F1: {grade.f1_final:.3f}, Precision: {grade.precision:.3f}, "
        f"Recall: {grade.recall:.3f}, Utility: {grade.utility_score:.3f}"
    )
    print(f"Success: {grade.success}, Total Reward: {total_reward:.2f}")
    return grade


if __name__ == "__main__":
    for task in ["gdpr_contract_easy", "hipaa_medical_medium", "security_logs_hard"]:
        run_baseline(task)
