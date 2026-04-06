import random
import re
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
    from ..utils import iou as _iou_fn
except ImportError:
    from models import (
        ActionType,
        PIIEntity,
        RedactionAction,
        RedactionObservation,
        RedactionReward,
        TaskGrade,
    )
    from utils import iou as _iou_fn
from .tasks import TASKS, get_task, load_documents


class RedactionEnvironment(Environment):
    def __init__(self, task_id: Optional[str] = None, window_size: int = 200, max_steps: int = 100):
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
        self._cached_tp = 0
        self._cached_fp = 0
        self._cached_fn = 0
        self._cached_label_correct = 0
        self._cached_label_total = 0
        self._matched_gt_indices: set[int] = set()
        self._seed = 0

        if task_id is not None:
            task = get_task(task_id)
            self.current_task = {"task_id": task_id, **task}
            if not self._custom_max_steps:
                self.max_steps = task.get("max_steps", max_steps)
            self._state = State(episode_id=task_id, step_count=0)

    def reset(
        self, task_id: Optional[str] = None, seed: Optional[int] = None
    ) -> RedactionObservation:
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
        self.ground_truth = [
            PIIEntity(**entity) for entity in self.current_doc.get("entities", [])
        ]

        self.cursor = 0
        self.redacted_spans = []
        self.detected_spans = []
        self.detected_entities = []
        self.step_count = 0
        self.done = False
        self.previous_actions = []
        self.total_redacted_chars = 0
        self._recent_redact_spans = deque(maxlen=10)
        self._cached_tp = 0
        self._cached_fp = 0
        self._cached_fn = len(self.ground_truth)
        self._cached_label_correct = 0
        self._cached_label_total = 0
        self._matched_gt_indices = set()
        self._state.step_count = 0

        return self._build_observation()

    def step(
        self, action: RedactionAction
    ) -> Tuple[RedactionObservation, RedactionReward, bool, dict]:
        if self.current_doc is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        old_potential = self._calculate_potential()
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
                already_detected = any(
                    _iou_fn((start, end), (detected.start, detected.end)) > 0.8
                    for detected in self.detected_entities
                )
                if already_detected:
                    invalid_action = True
                else:
                    self.redacted_spans.append((start, end))
                    self.detected_spans.append((start, end))
                    self.total_redacted_chars = self._merged_span_length(
                        self.redacted_spans
                    )

                    entity_text = self.current_doc["text"][start:end]
                    detected = PIIEntity(
                        label=action.label, start=start, end=end, text=entity_text
                    )
                    self.detected_entities.append(detected)
                    self._update_cached_metrics_for_detection(detected)

        elif action.action_type == ActionType.FINISH:
            self.done = True

        elif action.action_type == ActionType.SKIP:
            pass

        self.previous_actions.append(self._action_to_string(action))
        self.previous_actions = self.previous_actions[-5:]

        if self.step_count >= self.max_steps:
            self.done = True

        reward = self.compute_reward(
            action, old_potential, invalid_action=invalid_action
        )

        observation = self._build_observation()
        info = {
            "step": self.step_count,
            "invalid_action": invalid_action,
            "task_id": self.current_task["task_id"],
        }
        return observation, reward, self.done, info

    def _update_cached_metrics_for_detection(self, det: PIIEntity) -> None:
        best_idx = None
        best_iou = 0.0
        for idx, gt in enumerate(self.ground_truth):
            if idx in self._matched_gt_indices:
                continue
            score = _iou_fn((det.start, det.end), (gt.start, gt.end))
            if score > 0.6 and score > best_iou:
                best_iou = score
                best_idx = idx

        if best_idx is not None:
            self._matched_gt_indices.add(best_idx)
            self._cached_tp += 1
            self._cached_fn = max(0, self._cached_fn - 1)
            self._cached_label_total += 1
            if det.label == self.ground_truth[best_idx].label:
                self._cached_label_correct += 1
        else:
            self._cached_fp += 1

    def _calculate_potential(self) -> float:
        """
        Calculate the Potential Function Phi(s) based on the current grading metric.
        Phi(s) = 0.55 * F1 + 0.15 * LabelAccuracy + 0.30 * Utility
        """
        if self.current_doc is None:
            return 0.0

        precision = (
            self._cached_tp / (self._cached_tp + self._cached_fp)
            if (self._cached_tp + self._cached_fp)
            else 0.0
        )
        recall = (
            self._cached_tp / (self._cached_tp + self._cached_fn)
            if (self._cached_tp + self._cached_fn)
            else 0.0
        )
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall)
            else 0.0
        )
        label_accuracy = (
            (self._cached_label_correct / self._cached_label_total)
            if self._cached_label_total
            else 0.0
        )

        doc_len = len(self.current_doc["text"])
        over_redaction_ratio = (self.total_redacted_chars / doc_len) if doc_len else 0.0
        if over_redaction_ratio > 0.25:
            utility_score = max(0.0, 1 - 2 * (over_redaction_ratio - 0.25))
        else:
            utility_score = 1.0

        utility_weight = 0.30
        return (0.55 * f1) + (0.15 * label_accuracy) + (utility_weight * utility_score)

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

    def compute_reward(
        self,
        action: RedactionAction,
        old_potential: float,
        invalid_action: bool = False,
    ) -> RedactionReward:
        components: Dict[str, float] = {}

        # 1. Shaping Reward (PBRS): Delta Phi
        new_potential = self._calculate_potential()
        gamma = 0.99
        shaping_reward = (gamma * new_potential) - old_potential
        components["shaping_potential"] = round(shaping_reward, 4)

        # 2. Base Rewards (Direct penalties for undesirable behavior)
        tp_bonus = 0.0
        fp_penalty = 0.0
        duplicate_penalty = 0.0
        invalid_penalty = -1.0 if invalid_action else 0.0

        is_span_action = (
            action.action_type == ActionType.REDACT
            and action.start is not None
            and action.end is not None
        )

        if is_span_action:
            span = (action.start, action.end)

            # Duplicate action penalty
            dup_count = sum(
                1 for prev in self._recent_redact_spans if _iou_fn(span, prev) > 0.8
            )
            if dup_count > 0:
                duplicate_penalty = -0.2 * dup_count
            self._recent_redact_spans.append(span)

            # Direct FP penalty (for spans with near-zero IoU)
            best_iou = (
                max(_iou_fn(span, (gt.start, gt.end)) for gt in self.ground_truth)
                if self.ground_truth
                else 0.0
            )
            if best_iou > 0.6:
                tp_bonus = 0.1 * best_iou
                fp_penalty = 0.0
            else:
                fp_penalty = -0.2 * (1 - best_iou)

        components["tp_bonus"] = round(tp_bonus, 4)
        components["fp_penalty"] = round(fp_penalty, 4)
        components["duplicate_penalty"] = round(duplicate_penalty, 4)
        components["invalid_penalty"] = round(invalid_penalty, 4)

        raw_total = round(
            shaping_reward
            + tp_bonus
            + fp_penalty
            + duplicate_penalty
            + invalid_penalty,
            4,
        )
        # Keep environment reward unnormalized. Competition-facing clamping is
        # applied in inference logging/output.
        total = raw_total

        remaining_entities = max(0, self._cached_fn)

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
            progress_pct=1.0 if self.done else round(window_end / len(text), 4),
            previous_actions=list(self.previous_actions[-5:]),
            done=self.done,
        )

    def _masked_window(self, text: str, start: int, end: int) -> str:
        parts: List[str] = []
        i = start
        while i < end:
            covering = [
                span
                for span in self.redacted_spans
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
            return self._regex_label(start, end)

        span = (start, end)
        best = max(self.ground_truth, key=lambda gt: _iou_fn(span, (gt.start, gt.end)))
        if _iou_fn(span, (best.start, best.end)) > 0.6:
            return best.label
        return self._regex_label(start, end)

    def _regex_label(self, start: int, end: int) -> str:
        """Heuristic label based on span text when no GT match exists."""
        text = (
            self.current_doc["text"][start:end].strip()
            if self.current_doc is not None
            else ""
        )
        if re.fullmatch(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", text):
            return "EMAIL"
        if re.fullmatch(r"\d{3}[-\s]\d{2}[-\s]\d{4}", text):
            return "SSN"
        if re.fullmatch(r"(\+?1[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}", text):
            return "PHONE"
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4}", text):
            return "DOB"
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
        precision = (
            self._cached_tp / (self._cached_tp + self._cached_fp)
            if (self._cached_tp + self._cached_fp)
            else 0.0
        )
        recall = (
            self._cached_tp / (self._cached_tp + self._cached_fn)
            if (self._cached_tp + self._cached_fn)
            else 0.0
        )
        if precision + recall == 0:
            return 0.0
        return round(2 * precision * recall / (precision + recall), 4)

    @staticmethod
    def _action_to_string(action: RedactionAction) -> str:
        if action.action_type == ActionType.REDACT:
            lbl = f",{action.label}" if action.label else ""
            return f"REDACT({action.start},{action.end}{lbl})"
        if action.action_type == ActionType.PREV_CHUNK:
            return "PREV_CHUNK"
        return action.action_type.value
