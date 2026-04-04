from __future__ import annotations

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient
from openenv.core.env_server.types import State

from src.models import RedactionAction, RedactionObservation


class PIIRedactionEnv(EnvClient[RedactionAction, RedactionObservation, State]):
    """HTTP/WebSocket client for the PII redaction environment."""

    def _step_payload(self, action: RedactionAction) -> Dict[str, Any]:
        return action.model_dump(mode="json")

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[RedactionObservation]:
        obs_data = payload.get("observation", {})
        observation = RedactionObservation(
            task_id=obs_data.get("task_id", ""),
            document_id=obs_data.get("document_id", ""),
            visible_text=obs_data.get("visible_text", ""),
            cursor_position=obs_data.get("cursor_position", 0),
            document_length=obs_data.get("document_length", 0),
            redacted_spans=[tuple(span) for span in obs_data.get("redacted_spans", [])],
            progress_pct=obs_data.get("progress_pct", 0.0),
            previous_actions=obs_data.get("previous_actions", []),
            done=payload.get("done", obs_data.get("done", False)),
            reward=payload.get("reward", obs_data.get("reward")),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
