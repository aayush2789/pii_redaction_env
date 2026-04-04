from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator


class ActionType(str, Enum):
    REDACT = "REDACT"
    SKIP = "SKIP"
    NEXT_CHUNK = "NEXT_CHUNK"
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
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def require_span_for_redact(self):
        if self.action_type == ActionType.REDACT and (
            self.start is None or self.end is None
        ):
            raise ValueError("REDACT requires start/end")
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
    success: bool
    components: Dict[str, float]
