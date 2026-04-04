from __future__ import annotations

import os
from openenv.core.env_server.http_server import create_app
from src.models import RedactionAction, RedactionObservation
from src.environment import RedactionEnvironment

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

def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
