# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Pii Redaction Env Environment.

The pii_redaction_env environment is a simple test environment that echoes back messages.
"""

from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator


PII_LABELS = ("EMAIL", "PHONE", "SSN", "NAME", "ADDRESS", "DOB")


class ActionType(str, Enum):
    REDACT = "REDACT"
    SKIP = "SKIP"
    NEXT_CHUNK = "NEXT_CHUNK"
    PREV_CHUNK = "PREV_CHUNK"
    FINISH = "FINISH"


class PIIEntity(BaseModel):
    label: Literal["EMAIL", "PHONE", "SSN", "NAME", "ADDRESS", "DOB"]
    start: int
    end: int
    text: str

    @field_validator("end")
    @classmethod
    def end_after_start(cls, v: int, info):
        start = info.data.get("start")
        if start is not None and v <= start:
            raise ValueError("end must be > start")
        return v


class RedactionAction(BaseModel):
    action_type: ActionType
    start: Optional[int] = None
    end: Optional[int] = None
    label: Optional[Literal["EMAIL", "PHONE", "SSN", "NAME", "ADDRESS", "DOB"]] = None
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    justification: Optional[str] = None

    @model_validator(mode="after")
    def require_fields_for_redact(self):
        if self.action_type == ActionType.REDACT and (
            self.start is None or self.end is None
        ):
            raise ValueError("REDACT requires start/end")
        if self.action_type == ActionType.REDACT and self.label is None:
            raise ValueError("REDACT requires label")
        return self


class RedactionObservation(BaseModel):
    task_id: str
    document_id: str
    visible_text: str
    cursor_position: int
    document_length: int
    redacted_spans: List[Tuple[int, int]]
    progress_pct: float
    previous_actions: List[str]
    done: bool


class RedactionReward(BaseModel):
    total: float
    raw_total: float
    components: Dict[str, float]
    f1_score_current: float
    remaining_entities: int


class TaskGrade(BaseModel):
    task_id: str
    score: float
    f1_final: float
    precision: float
    recall: float
    utility_score: float
    label_accuracy: float
    success: bool
    components: Dict[str, float]
