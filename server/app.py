# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Pii Redaction Env Environment.

This module creates an HTTP server that exposes the PiiRedactionEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

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


def create_redaction_environment() -> RedactionEnvironment:
    """Factory function that creates RedactionEnvironment with config."""

    task_id = os.getenv("PII_TASK_ID")
    window_size = int(os.getenv("PII_WINDOW_SIZE", "500"))
    max_steps = int(os.getenv("PII_MAX_STEPS", "100"))
    return RedactionEnvironment(
        task_id=task_id,
        window_size=window_size,
        max_steps=max_steps,
    )


app = create_app(
    create_redaction_environment,
    RedactionAction,
    RedactionObservation,
    env_name="pii-redaction-env",
)


@app.post("/grade")
async def grade() -> dict:
    """Explicitly expose the grade method of the environment."""
    # Note: create_app stores the env factory/instance.
    # We can get the environment from the app state if needed,
    # but the simplest way is to use the environment associated with the request session.
    # For now, we'll assume a single-session model or use the env_registry.
    from openenv.core.env_server.http_server import get_env_registry

    registry = get_env_registry(app)
    # Get the first active env (for single-session local testing)
    envs = list(registry.envs.values())
    if not envs:
        return {"error": "No active environment session"}
    env = envs[0]
    grade_obj = env.grade()
    # TaskGrade is a pydantic model, so we can use model_dump()
    return grade_obj.model_dump()


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
