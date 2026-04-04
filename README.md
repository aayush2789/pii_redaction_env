# PII Redaction Assistant OpenEnv

A complete OpenEnv-compliant environment for training and evaluating agents that redact personally identifiable information (PII) in realistic GDPR/HIPAA document workflows.

## Motivation

Automated PII redaction is a high-impact compliance need:
- GDPR violations can incur fines up to 4% of global annual revenue.
- HIPAA violations can trigger major penalties, legal exposure, and mandatory breach notifications.
- Real review pipelines require sequential decisions with limited context windows, not one-shot extraction.

This environment simulates that real process: agents inspect a sliding window, redact spans, advance through the document, and trade off recall versus over-redaction utility.

## Environment Design

### Observation Space
`RedactionObservation` includes:
- `visible_text`: Sliding text window (`window_size`, default 500 chars)
- `cursor_position`: Absolute offset in full document
- `document_length`: Total characters
- `redacted_spans`: All absolute redaction spans so far
- `progress_pct`: Fraction of current document window covered
- `previous_actions`: Last 5 actions for policy context
- `done`: Episode completion flag

Redacted regions are masked in subsequent windows as `[REDACTED]`.

### Action Space
`RedactionAction` supports:
- `REDACT(start, end)`: redact absolute char span
- `NEXT_CHUNK()`: advance cursor by 50% overlap (`window_size / 2`)
- `SKIP()`: no-op action
- `FINISH()`: terminate episode

Examples:
- `{"action_type": "REDACT", "start": 120, "end": 136}`
- `{"action_type": "NEXT_CHUNK"}`
- `{"action_type": "FINISH"}`

### Reward Function
Dense reward balances compliance and utility:
- Immediate feedback:
- `progress_bonus = +0.1` on `NEXT_CHUNK`
- True-positive redaction bonus scaled by IOU if `IOU > 0.6`
- False-positive penalty `-0.5`
- End-of-episode penalties:
- Missed entities: `-2.0 * missed`
- Utility penalty if over-redaction ratio exceeds 25%

Final task score combines quality and usefulness:
- `Final Score = 0.7 * F1 + 0.3 * utility_score`
- Utility decreases when too much text is redacted.

## Tasks

### Easy: `gdpr_contract_easy`
Pattern-heavy synthetic contracts with obvious emails, phones, addresses.
- Regex-based approaches can perform well.
- Objective: redact obvious PII with high precision and minimal over-redaction.
- Success threshold: 0.90

### Medium: `hipaa_medical_medium`
Medical-style notes with contextual entities (names, DOB, phone/address references).
- Requires context-aware extraction.
- Objective: identify contextual patient identifiers and redact them without removing clinical text.
- Success threshold: 0.85

### Hard: `security_logs_hard`
Adversarial support logs with obfuscation and ambiguity.
- Example: `john dot smith at gmail dot com`
- Ambiguous references and nested mentions.
- Objective: catch obfuscated PII while avoiding false positives on company names and ordinary references.
- Success threshold: 0.75

Each task uses a deterministic grader with a normalized score in the inclusive range $[0, 1]$.
The final score is a weighted blend of exactness and utility, so an agent must balance recall against over-redaction.

## Project Structure

```text
pii-redaction-env/
├── openenv.yaml
├── README.md
├── Dockerfile
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── models.py
│   ├── environment.py
│   ├── tasks.py
│   ├── graders.py
│   └── data/
│       ├── __init__.py
│       ├── easy_docs.json
│       ├── medium_docs.json
│       └── hard_docs.json
├── scripts/
│   └── baseline.py
└── tests/
    ├── test_env.py
    └── test_graders.py
```

## Setup

1. Activate your virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set API key for baseline:

```bash
set OPENAI_API_KEY=your_key_here
```

## Usage

Run baseline across all tasks:

```bash
python scripts/baseline.py
```

Programmatic usage:

```python
from src.environment import RedactionEnvironment
from src.models import RedactionAction, ActionType

env = RedactionEnvironment(task_id="gdpr_contract_easy")
obs = env.reset()

obs, reward, done, info = env.step(
    RedactionAction(action_type=ActionType.NEXT_CHUNK)
)

if not done:
    obs, reward, done, info = env.step(
        RedactionAction(action_type=ActionType.FINISH)
    )

grade = env.grade()
print(grade.dict())
```

## Baseline Expectations (GPT-4o-mini)

Expected approximate outcomes:
- Easy: ~0.92
- Medium: ~0.78
- Hard: ~0.65

These are directional benchmarks and may vary by model version and prompt strategy.

## Testing

Run all tests:

```bash
pytest tests/ -v
```

Included tests validate:
- Observation reset integrity
- Masking behavior after redaction
- Cursor movement for overlapping windows
- Invalid redaction penalties
- Grade/F1 correctness
- Episode termination on finish and max steps

## OpenEnv Metadata

Environment metadata is defined in `openenv.yaml` with:
- Entrypoint: `src.environment:RedactionEnvironment`
- Typed observation/action models from `src.models`
- Task list and reward range for evaluator integrations
