"""Compatibility package for the legacy server.src import path."""

from server.graders import compute_grade
from server.pii_redaction_env_environment import RedactionEnvironment
from server.tasks import TASKS, get_task, load_documents

__all__ = [
    "RedactionEnvironment",
    "compute_grade",
    "TASKS",
    "get_task",
    "load_documents",
]
