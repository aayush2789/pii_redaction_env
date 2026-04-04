from fastapi import FastAPI
import uvicorn

from src.environment import RedactionEnvironment

app = FastAPI(title="PII Redaction OpenEnv Server")


@app.get("/health")
def health() -> dict:
    env = RedactionEnvironment(task_id="gdpr_contract_easy")
    obs = env.reset()
    return {
        "status": "ok",
        "task": obs.task_id,
        "document_id": obs.document_id,
    }


def main() -> None:
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
