# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pii Redaction Env client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import RedactionAction, RedactionObservation
except ImportError:
    from models import RedactionAction, RedactionObservation


class RedactionEnv(EnvClient[RedactionAction, RedactionObservation, State]):
    """
    Client for the Pii Redaction Env environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with RedactionEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.visible_text)
        ...
        ...     result = client.step(RedactionAction(action_type="SKIP"))
        ...     print(result.observation.progress_pct)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = RedactionEnv.from_docker_image("pii_redaction_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(RedactionAction(action_type="NEXT_CHUNK"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: RedactionAction) -> Dict:
        """
        Convert RedactionAction to JSON payload for step message.

        Args:
            action: RedactionAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[RedactionObservation]:
        """
        Parse server response into StepResult[RedactionObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with RedactionObservation
        """
        obs_data = payload.get("observation", {})
        observation = RedactionObservation(**obs_data)

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )


# Backward-compatible alias.
PiiRedactionEnv = RedactionEnv
