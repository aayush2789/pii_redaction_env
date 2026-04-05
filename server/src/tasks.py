"""Legacy import shim for server.src.tasks."""

from server.tasks import TASKS, get_task, load_documents

__all__ = ["TASKS", "get_task", "load_documents"]
