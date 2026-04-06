---
title: Pii Redaction Env
emoji: 🛡️
colorFrom: gray
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Pii Redaction Env

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi)
![OpenEnv](https://img.shields.io/badge/OpenEnv-runtime-111827)
![Docker](https://img.shields.io/badge/Docker-container-2496ED?logo=docker&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-spaces-FFD21E?logo=huggingface&logoColor=000)

Pii Redaction Env is a document redaction environment for evaluating agents on labeled PII detection. The agent scans a text document with a sliding window, identifies personally identifiable information, and redacts spans using absolute character offsets and a PII label.

The environment is designed to test three things at once:

1. Span localization: can the agent find the exact character range?
2. Type classification: can it assign the right label?
3. Coverage discipline: can it finish the document without over-redacting unrelated text?

## Tasks

The bundled tasks come from synthetic documents with different difficulty profiles:

| Task | Difficulty | Objective | Success threshold | Max steps |
| --- | --- | --- | --- | --- |
| `gdpr_contract_easy` | Easy | Redact obvious PII in business contracts: emails, phone numbers, and addresses. | 0.90 | 40 |
| `hipaa_medical_medium` | Medium | Redact contextual patient identifiers in medical notes: names, DOBs, phones, and addresses. | 0.85 | 60 |
| `security_logs_hard` | Hard | Detect obfuscated and ambiguous PII in support logs without over-redacting non-PII entities. | 0.75 | 80 |

The difficulty rises from regex-friendly text to contextual and obfuscated PII that requires broader document understanding.

## Environment Contract

## State Space

The agent receives a structured `RedactionObservation` at every step. This is the state that the LLM reasons over in the sequential redaction loop.

The observation contains:

- `visible_text`: the current sliding window of document text, with previously redacted regions masked as `[REDACTED]`.
- `cursor_position`: the absolute character offset where the current window begins.
- `document_length`: the full document length in characters.
- `redacted_spans`: every span the agent has already chosen to redact, represented as `(start, end)` offsets.
- `progress_pct`: the current coverage fraction of the document.
- `previous_actions`: the most recent actions, kept as short-term memory to reduce looping.
- `done`: whether the episode has ended.

This is intentionally not a one-shot extraction setup. The model only sees a local window and must decide whether to redact, move forward, move back, skip, or finish.

### Actions

`RedactionAction` supports the following action types:

- `REDACT`: redact one span using `start`, `end`, and `label`.
- `SKIP`: keep the current window unchanged and continue reasoning.
- `NEXT_CHUNK`: move forward by half a window.
- `PREV_CHUNK`: move backward by half a window.
- `FINISH`: end the episode.

For `REDACT`, both the span and the label are required. Valid labels are `EMAIL`, `PHONE`, `SSN`, `NAME`, `ADDRESS`, and `DOB`.

Action behavior:

- `REDACT(start, end, label)` uses absolute document offsets, not offsets relative to the visible window.
- `NEXT_CHUNK` advances the cursor by half the window size, which creates overlap so entities split across boundaries are still visible.
- `PREV_CHUNK` moves the cursor backward by half the window size.
- `SKIP` consumes a step but leaves the cursor unchanged.
- `FINISH` ends the episode once the agent believes the document has been fully reviewed.

### Role of the LLM

The LLM is used as an autonomous sequential decision-maker rather than a direct span extractor.

- At each step, it receives the observation JSON and the recent action history.
- It inspects the visible window, identifies likely PII spans, and emits a structured JSON action.
- If it finds a span, it should return `REDACT` with absolute indices and a label.
- If the current window is exhausted, it should move the cursor with `NEXT_CHUNK` or `PREV_CHUNK`.
- If no PII is visible, it can `SKIP` and continue scanning.
- It should only `FINISH` once the full document has been reviewed and no PII remains.

This makes the task suitable for both evaluation and training loops that want dense step-by-step feedback, including PPO-style and GRPO-style workflows.

### Observations

Each observation contains:

- `task_id`: active task name.
- `document_id`: selected document ID.
- `visible_text`: the current masked window.
- `cursor_position`: absolute offset of the current window.
- `document_length`: total document length.
- `redacted_spans`: spans already redacted.
- `progress_pct`: current coverage fraction.
- `previous_actions`: recent action history.
- `done`: whether the episode has ended.

## Reward Method

The reward is built around potential-based reward shaping (PBRS) so that the agent gets dense feedback without losing the structure of the final task objective.

The potential function is:

$$
\Phi(s) = 0.55 \cdot F1 + 0.15 \cdot LabelAccuracy + 0.30 \cdot Utility
$$

The shaping term is:

$$
r_{shape} = \gamma \Phi(s') - \Phi(s), \quad \gamma = 0.99
$$

That gives the agent a reward signal for improving the current state, not just for ending the episode well.

### Why this works

- `F1` measures whether the agent is finding the right spans.
- `LabelAccuracy` rewards choosing the right PII type once a span is matched.
- `Utility` keeps the agent from redacting too much of the document. The utility stays at `1.0` until redaction exceeds 25% of the text, then decays linearly.
- `gamma` makes future improvements count slightly less than immediate ones, which stabilizes learning while preserving the long-term objective.

The environment also adds small direct terms to make the policy harder to exploit:

- true-positive bonuses for plausible span matches,
- false-positive penalties for redacting unrelated text,
- duplicate penalties for re-redacting the same area,
- invalid-action penalties for malformed spans.

At the step level, the environment computes:

$$
r_t = (\gamma \Phi(s_{t+1}) - \Phi(s_t)) + r_{tp} + r_{fp} + r_{dup} + r_{invalid}
$$

where:

- `r_tp` is a small bonus when a `REDACT` span overlaps a true entity strongly enough to count as a match.
- `r_fp` is a penalty for redacting text that does not meaningfully match any entity.
- `r_dup` discourages repeated redaction of the same area.
- `r_invalid` penalizes malformed spans or out-of-bounds actions.

The final step reward is normalized into `[0, 1]`, while the raw shaped signal is also exposed for debugging.

### Terminal Scoring

The end-of-episode grade is computed separately by the grader. It combines:

- `F1`: how well the detected spans match ground truth.
- `label_accuracy`: how often the label on a matched span is correct.
- `utility_score`: how much readable text the agent preserved.

The final score uses the weighted formula:

$$
score = 0.55 \cdot F1 + 0.15 \cdot LabelAccuracy + 0.30 \cdot Utility
$$

That means the agent is rewarded for being precise, correctly typed, and conservative enough to preserve readability.

### Intuition

The design is trying to balance two failure modes:

- under-redaction, where the agent misses sensitive entities and gets low recall,
- over-redaction, where the agent deletes large portions of the document and destroys utility.

PBRS gives the model a gradient-like signal after every step, while the terminal grade makes sure the final document-level quality still matters.

## Baseline Scores

The table below uses a simple local regex-and-anchor heuristic over the bundled documents and the current grader. It is meant as a transparent reference point, not a production benchmark.

| Task | Mean score | Mean F1 | Success rate |
| --- | --- | --- | --- |
| `gdpr_contract_easy` | 0.9307 | 0.8900 | 60% |
| `hipaa_medical_medium` | 0.9246 | 0.8628 | 100% |
| `security_logs_hard` | 0.6384 | 0.3971 | 30% |

## Setup

Use the project venv or `uv` from the repository root.

```bash
uv sync
```

Run the test suite:

```bash
uv run pytest
```

## Run Locally

Start the FastAPI server:

```bash
uv run uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

Or use the packaged entry point:

```bash
uv run server
```

## Docker

Build the container from the project root:

```bash
docker build -t pii_redaction_env-env:latest -f server/Dockerfile .
```

Run it:

```bash
docker run --rm -p 8000:8000 pii_redaction_env-env:latest
```

## Hugging Face Spaces

The repo is configured for OpenEnv + Docker Spaces through `openenv.yaml` and `server/app.py`.

To deploy:

```bash
openenv push
```

The deployed space exposes the web UI at `/web`, with the API served from the same container on port `8000`.

## Project Layout

```text
pii_redaction_env/
├── client.py
├── inference.py
├── models.py
├── openenv.yaml
├── pyproject.toml
├── README.md
├── sample_inference_script.py
├── server/
│   ├── app.py
│   ├── Dockerfile
│   ├── graders.py
│   ├── pii_redaction_env_environment.py
│   ├── tasks.py
│   └── data/
└── tests/
```

## Reference Agent

`inference.py` contains the reference loop used for benchmarking. It builds a label-aware prompt, snaps model outputs to valid spans when possible, retries on transient backend failures, and falls back to a deterministic local policy when the provider is unavailable.
