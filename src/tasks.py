import json
from pathlib import Path
from typing import Dict, List


TASKS = {
    "gdpr_contract_easy": {
        "difficulty": "easy",
        "objective": "Redact obvious PII in business contracts: emails, phone numbers, and addresses.",
        "description": "Synthetic business contracts with obvious emails, phone numbers, and standard addresses. Regex-able patterns.",
        "success_threshold": 0.90,
        "max_steps": 40,
        "docs_file": "easy_docs.json",
        "grading_focus": ["EMAIL", "PHONE", "ADDRESS"],
    },
    "hipaa_medical_medium": {
        "difficulty": "medium",
        "objective": "Redact contextual patient identifiers in medical notes: names, DOBs, phones, and addresses.",
        "description": "Medical discharge summaries with contextual PII. Names embedded in sentences, dates of birth, medical record numbers. Requires context understanding.",
        "success_threshold": 0.85,
        "max_steps": 60,
        "docs_file": "medium_docs.json",
        "grading_focus": ["NAME", "DOB", "PHONE", "ADDRESS"],
    },
    "security_logs_hard": {
        "difficulty": "hard",
        "objective": "Detect obfuscated and ambiguous PII in support logs without over-redacting non-PII entities.",
        "description": "Adversarial support chat logs with obfuscated PII (e.g., 'john dot smith at gmail dot com'), nested references ('my wife Sarah'), ambiguous entities ('Contact Apple' vs 'Contact Apple Inc').",
        "success_threshold": 0.75,
        "max_steps": 80,
        "docs_file": "hard_docs.json",
        "grading_focus": ["EMAIL", "PHONE", "NAME"],
    },
}

_DATA_DIR = Path(__file__).resolve().parent / "data"


def get_task(task_id: str) -> Dict:
    if task_id not in TASKS:
        raise ValueError(f"Unknown task_id: {task_id}")
    return TASKS[task_id]


def load_documents(task_id: str) -> List[Dict]:
    task = get_task(task_id)
    docs_file = _DATA_DIR / task["docs_file"]
    with docs_file.open("r", encoding="utf-8") as f:
        return json.load(f)
