import math
import random
from collections import deque
from typing import Dict, List, Optional, Tuple

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from .graders import compute_grade
try:
    from ..models import (
        ActionType,
        PIIEntity,
        RedactionAction,
        RedactionObservation,
        RedactionReward,
        TaskGrade,
    )
except ImportError:
    from models import (
        ActionType,
        PIIEntity,
        RedactionAction,
        RedactionObservation,
        RedactionReward,
        TaskGrade,
    )
from .tasks import TASKS, get_task, load_documents


class RedactionEnvironment(Environment):
    def __init__(self, task_id: Optional[str] = None, window_size: int = 500, max_steps: int = 100):
        super().__init__()
        self._state = State(episode_id=None, step_count=0)
        self.window_size = window_size
        self.current_task = None
        self.current_doc = None
        self.cursor = 0
        self.redacted_spans = []
        self.step_count = 0
        self.max_steps = max_steps
        self._custom_max_steps = max_steps != 100
        self.ground_truth = []
        self.detected_entities = []

        self.done = False
        self.previous_actions: List[str] = []
        self.total_redacted_chars = 0
        self.detected_spans: List[Tuple[int, int]] = []
        self._recent_redact_spans: deque = deque(maxlen=10)
        self.annotations: List[Dict] = []  # {start, end, label, correct}
        self._seed = 0

        if task_id is not None:
            task = get_task(task_id)
            self.current_task = {"task_id": task_id, **task}
            if not self._custom_max_steps:
                self.max_steps = task.get("max_steps", max_steps)
            self._state = State(episode_id=task_id, step_count=0)

    def reset(self, task_id: Optional[str] = None, seed: Optional[int] = None) -> RedactionObservation:
        if task_id is not None:
            task = get_task(task_id)
            self.current_task = {"task_id": task_id, **task}
            if not self._custom_max_steps:
                self.max_steps = task.get("max_steps", self.max_steps)
            self._state = State(episode_id=task_id, step_count=0)
        elif self.current_task is None:
            default_task_id = next(iter(TASKS.keys()))
            task = get_task(default_task_id)
            self.current_task = {"task_id": default_task_id, **task}
            if not self._custom_max_steps:
                self.max_steps = task.get("max_steps", self.max_steps)
            self._state = State(episode_id=default_task_id, step_count=0)

        docs = load_documents(self.current_task["task_id"])
        effective_seed = seed if seed is not None else self._seed
        self.current_doc = random.Random(effective_seed).choice(docs)
        if seed is None:
            self._seed += 1
        self.ground_truth = [PIIEntity(**entity) for entity in self.current_doc.get("entities", [])]

        self.cursor = 0
        self.redacted_spans = []
        self.detected_spans = []
        self.detected_entities = []
        self.step_count = 0
        self.done = False
        self.previous_actions = []
        self.total_redacted_chars = 0
        self._recent_redact_spans = deque(maxlen=10)
        self.annotations = []
        self._state.step_count = 0

        return self._build_observation()

    def step(self, action: RedactionAction) -> Tuple[RedactionObservation, RedactionReward, bool, dict]:
        if self.current_doc is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        old_state = self.state
        invalid_action = False

        self.step_count += 1
        self._state.step_count = self.step_count

        if action.action_type == ActionType.NEXT_CHUNK:
            step_size = max(1, self.window_size // 2)
            max_cursor = max(0, len(self.current_doc["text"]) - 1)
            self.cursor = min(max_cursor, self.cursor + step_size)

        elif action.action_type == ActionType.PREV_CHUNK:
            step_size = max(1, self.window_size // 2)
            self.cursor = max(0, self.cursor - step_size)

        elif action.action_type == ActionType.REDACT:
            start = action.start if action.start is not None else -1
            end = action.end if action.end is not None else -1
            if start < 0 or end <= start or end > len(self.current_doc["text"]):
                invalid_action = True
            else:
                self.redacted_spans.append((start, end))
                self.detected_spans.append((start, end))
                self.total_redacted_chars = self._merged_span_length(self.redacted_spans)

                entity_text = self.current_doc["text"][start:end]
                # Use agent-provided label if present, otherwise auto-assign
                label = action.label if action.label else self._best_label(start, end)
                self.detected_entities.append(
                    PIIEntity(label=label, start=start, end=end, text=entity_text)
                )

        elif action.action_type == ActionType.ANNOTATE:
            start = action.start if action.start is not None else -1
            end = action.end if action.end is not None else -1
            agent_label = action.label
            if start < 0 or end <= start or end > len(self.current_doc["text"]) or agent_label is None:
                invalid_action = True
            else:
                # ANNOTATE = REDACT + explicit label assignment
                self.redacted_spans.append((start, end))
                self.detected_spans.append((start, end))
                self.total_redacted_chars = self._merged_span_length(self.redacted_spans)

                entity_text = self.current_doc["text"][start:end]
                gt_label = self._best_label(start, end)
                label_correct = (agent_label == gt_label)
                self.annotations.append({
                    "start": start, "end": end,
                    "agent_label": agent_label, "gt_label": gt_label,
                    "correct": label_correct,
                })
                # Store the agent's chosen label in detected entities
                self.detected_entities.append(
                    PIIEntity(label=agent_label, start=start, end=end, text=entity_text)
                )

        elif action.action_type == ActionType.FINISH:
            self.done = True

        elif action.action_type == ActionType.SKIP:
            pass

        self.previous_actions.append(self._action_to_string(action))
        self.previous_actions = self.previous_actions[-5:]

        if self.step_count >= self.max_steps:
            self.done = True

        reward = self.compute_reward(action, old_state)
        if invalid_action:
            reward.components["invalid_action_penalty"] = -1.0
            reward.raw_total = round(reward.raw_total - 1.0, 4)
            reward.total = round(max(0.0, min(1.0, (reward.raw_total + 1.0) / 2.0)), 4)

        observation = self._build_observation()
        info = {
            "step": self.step_count,
            "invalid_action": invalid_action,
            "task_id": self.current_task["task_id"],
        }
        return observation, reward, self.done, info

    @property
    def state(self) -> State:
        return self._state

    def grade(self) -> TaskGrade:
        if self.current_doc is None or self.current_task is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        return compute_grade(
            detected=self.detected_entities,
            ground_truth=self.ground_truth,
            document_length=len(self.current_doc["text"]),
            total_redacted_chars=self.total_redacted_chars,
            task_id=self.current_task["task_id"],
            success_threshold=self.current_task["success_threshold"],
        )

    def compute_reward(self, action: RedactionAction, old_state: State) -> RedactionReward:
        del old_state
        components: Dict[str, float] = {}

        # Fix 7: Progress-scaled positive reward for NEXT_CHUNK
        if action.action_type == ActionType.NEXT_CHUNK:
            progress = self._current_progress()
            components["progress_bonus"] = round(0.05 * (1.0 - progress), 4)
        elif action.action_type == ActionType.PREV_CHUNK:
            components["progress_bonus"] = -0.05
        else:
            components["progress_bonus"] = 0.0

        tp_bonus = 0.0
        fp_penalty = 0.0
        duplicate_penalty = 0.0
        annotation_bonus = 0.0
        calibration_penalty = 0.0
        explainability_bonus = 0.0

        is_span_action = (
            action.action_type in (ActionType.REDACT, ActionType.ANNOTATE)
            and action.start is not None and action.end is not None
        )

        if is_span_action:
            span = (action.start, action.end)

            # Escalating duplicate-action penalty
            dup_count = sum(
                1 for prev in self._recent_redact_spans
                if self._iou(span, prev) > 0.8
            )
            if dup_count > 0:
                duplicate_penalty = -0.2 * dup_count
            self._recent_redact_spans.append(span)

            best_iou = (
                max(self._iou(span, (gt.start, gt.end)) for gt in self.ground_truth)
                if self.ground_truth
                else 0.0
            )

            if best_iou > 0.6:
                tp_bonus = best_iou
            else:
                fp_penalty = -0.3

            calibration_penalty = -0.1 * abs(action.confidence - best_iou)

            # ANNOTATE label-accuracy bonus: reward correct PII type classification
            if action.action_type == ActionType.ANNOTATE and action.label and best_iou > 0.6:
                gt_label = self._best_label(action.start, action.end)
                if action.label == gt_label:
                    annotation_bonus = 0.15  # correct label
                else:
                    annotation_bonus = -0.1  # wrong label on valid span

            if action.justification:
                label = self._best_label(action.start, action.end)
                keyword_map = {
                    "EMAIL": ["email", "contact", "mail"],
                    "PHONE": ["phone", "number", "call", "mobile"],
                    "NAME": ["name", "person", "patient"],
                    "ADDRESS": ["address", "street", "location", "residence"],
                    "DOB": ["dob", "birth", "date of birth"],
                    "SSN": ["ssn", "social security"],
                }
                text = action.justification.lower()
                if any(k in text for k in keyword_map.get(label, [])):
                    explainability_bonus = 0.1

        finish_bonus = 0.0
        if action.action_type == ActionType.FINISH:
            running_f1 = self._compute_running_f1()
            finish_bonus = min(0.7, 0.3 + (0.4 * running_f1))

        components["tp_bonus"] = round(tp_bonus, 4)
        components["fp_penalty"] = round(fp_penalty, 4)
        components["duplicate_penalty"] = round(duplicate_penalty, 4)
        components["annotation_bonus"] = round(annotation_bonus, 4)
        components["calibration_penalty"] = round(calibration_penalty, 4)
        components["explainability_bonus"] = round(explainability_bonus, 4)
        components["finish_bonus"] = round(finish_bonus, 4)

        raw_total = round(sum(components.values()), 4)
        # Fix 2: Linear clamping instead of sigmoid for better discriminability
        total = round(max(0.0, min(1.0, (raw_total + 1.0) / 2.0)), 4)

        matched_gt = 0
        for gt in self.ground_truth:
            if any(
                self._iou((gt.start, gt.end), (det.start, det.end)) > 0.6
                for det in self.detected_entities
            ):
                matched_gt += 1

        remaining_entities = max(0, len(self.ground_truth) - matched_gt)

        return RedactionReward(
            total=total,
            raw_total=raw_total,
            components=components,
            f1_score_current=self._compute_running_f1(),
            remaining_entities=remaining_entities,
        )

    def _current_progress(self) -> float:
        """Return fraction of document covered by current window end."""
        if self.current_doc is None:
            return 0.0
        text_len = len(self.current_doc["text"])
        window_end = min(text_len, self.cursor + self.window_size)
        return window_end / text_len if text_len > 0 else 1.0

    def _build_observation(self) -> RedactionObservation:
        text = self.current_doc["text"]
        window_end = min(len(text), self.cursor + self.window_size)

        return RedactionObservation(
            task_id=self.current_task["task_id"],
            document_id=self.current_doc["id"],
            visible_text=self._masked_window(text, self.cursor, window_end),
            cursor_position=self.cursor,
            document_length=len(text),
            redacted_spans=list(self.redacted_spans),
            annotations=list(self.annotations),
            progress_pct=1.0 if self.done else round(window_end / len(text), 4),
            previous_actions=list(self.previous_actions[-5:]),
            done=self.done,
        )

    def _masked_window(self, text: str, start: int, end: int) -> str:
        parts: List[str] = []
        i = start
        while i < end:
            covering = [
                span for span in self.redacted_spans
                if span[0] <= i < span[1] and span[0] < end and span[1] > start
            ]

            if covering:
                cover_end = min(end, max(span[1] for span in covering))
                span_len = cover_end - i
                mask_str = "[REDACTED]"
                if span_len <= len(mask_str):
                    parts.append(mask_str[:span_len])
                else:
                    parts.append(mask_str.ljust(span_len, "█"))
                i = cover_end
                continue

            next_starts = [span[0] for span in self.redacted_spans if i < span[0] < end]
            next_boundary = min(next_starts) if next_starts else end
            if next_boundary > i:
                parts.append(text[i:next_boundary])
            i = next_boundary

        return "".join(parts)

    def _best_label(self, start: int, end: int) -> str:
        if not self.ground_truth:
            return "NAME"

        span = (start, end)
        best = max(self.ground_truth, key=lambda gt: self._iou(span, (gt.start, gt.end)))
        if self._iou(span, (best.start, best.end)) > 0.6:
            return best.label
        return "NAME"

    def _merged_span_length(self, spans: List[Tuple[int, int]]) -> int:
        if not spans:
            return 0

        merged = sorted(spans)
        compact = [list(merged[0])]
        for s, e in merged[1:]:
            prev = compact[-1]
            if s <= prev[1]:
                prev[1] = max(prev[1], e)
            else:
                compact.append([s, e])
        return sum(e - s for s, e in compact)

    def _compute_running_f1(self) -> float:
        if not self.ground_truth:
            return 0.0

        matched_gt = set()
        tp = 0
        for det in self.detected_entities:
            best_idx = None
            best_iou = 0.0
            for idx, gt in enumerate(self.ground_truth):
                if idx in matched_gt:
                    continue
                # Fix 5: Relaxed matching — IoU-only, no label requirement
                # Prevents double-penalizing near-misses with auto-assigned labels
                score = self._iou((det.start, det.end), (gt.start, gt.end))
                if score > 0.6 and score > best_iou:
                    best_iou = score
                    best_idx = idx
            if best_idx is not None:
                matched_gt.add(best_idx)
                tp += 1

        fp = max(0, len(self.detected_entities) - tp)
        fn = max(0, len(self.ground_truth) - tp)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            return 0.0
        return round(2 * precision * recall / (precision + recall), 4)

    @staticmethod
    def _iou(span_a: Tuple[int, int], span_b: Tuple[int, int]) -> float:
        a0, a1 = span_a
        b0, b1 = span_b
        intersection = max(0, min(a1, b1) - max(a0, b0))
        union = max(a1, b1) - min(a0, b0)
        return (intersection / union) if union > 0 else 0.0

    @staticmethod
    def _action_to_string(action: RedactionAction) -> str:
        if action.action_type == ActionType.REDACT:
            lbl = f",{action.label}" if action.label else ""
            return f"REDACT({action.start},{action.end}{lbl})"
        if action.action_type == ActionType.ANNOTATE:
            return f"ANNOTATE({action.start},{action.end},{action.label})"
        if action.action_type == ActionType.PREV_CHUNK:
            return "PREV_CHUNK"
        return action.action_type.value
