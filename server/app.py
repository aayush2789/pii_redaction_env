# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Pii Redaction Env Environment.

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 7860

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 7860 --workers 4

    # Or run directly:
    python -m server.app
"""

from __future__ import annotations

import os
from openenv.core.env_server.http_server import create_app

try:
    from ..models import RedactionAction, RedactionObservation
except ImportError:
    from models import RedactionAction, RedactionObservation
from .pii_redaction_env_environment import RedactionEnvironment


class ServerRedactionEnvironment(RedactionEnvironment):
    """Server adapter: exposes observation-only step return for OpenEnv HTTP server."""

    def step(self, action: RedactionAction) -> RedactionObservation:  # type: ignore[override]
        observation, reward, done, info = super().step(action)
        observation.done = done
        observation.reward = reward.raw_total
        observation.metadata = {
            **(observation.metadata or {}),
            "reward_total": reward.total,
            "reward_raw_total": reward.raw_total,
            "reward_components": reward.components,
            "remaining_entities": reward.remaining_entities,
            "invalid_action": bool(info.get("invalid_action", False)),
        }
        return observation


def create_redaction_environment() -> ServerRedactionEnvironment:
    """Factory function that creates RedactionEnvironment with config."""

    task_id = os.getenv("PII_TASK_ID")
    window_size = int(os.getenv("PII_WINDOW_SIZE", "200"))
    max_steps = int(os.getenv("PII_MAX_STEPS", "100"))
    return ServerRedactionEnvironment(
        task_id=task_id,
        window_size=window_size,
        max_steps=max_steps,
    )


app = create_app(
    create_redaction_environment,
    RedactionAction,
    RedactionObservation,
    env_name="pii_redaction_env",
)

def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
