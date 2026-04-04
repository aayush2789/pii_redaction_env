from typing import Dict, List, Optional, Tuple

from .graders import compute_grade
from .models import (
    ActionType,
    PIIEntity,
    RedactionAction,
    RedactionObservation,
    RedactionReward,
    TaskGrade,
)
from .tasks import TASKS, get_task, load_documents


class RedactionEnvironment:
    def __init__(self, task_id: Optional[str] = None, window_size: int = 500, max_steps: int = 100):
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

        if task_id is not None:
            task = get_task(task_id)
            self.current_task = {"task_id": task_id, **task}
            if not self._custom_max_steps:
                self.max_steps = task.get("max_steps", max_steps)

    def reset(self, task_id: Optional[str] = None) -> RedactionObservation:
        if task_id is not None:
            task = get_task(task_id)
            self.current_task = {"task_id": task_id, **task}
            if not self._custom_max_steps:
                self.max_steps = task.get("max_steps", self.max_steps)
        elif self.current_task is None:
            default_task_id = next(iter(TASKS.keys()))
            task = get_task(default_task_id)
            self.current_task = {"task_id": default_task_id, **task}
            if not self._custom_max_steps:
                self.max_steps = task.get("max_steps", self.max_steps)

        docs = load_documents(self.current_task["task_id"])
        self.current_doc = docs[0]
        self.ground_truth = [PIIEntity(**entity) for entity in self.current_doc.get("entities", [])]

        self.cursor = 0
        self.redacted_spans = []
        self.detected_spans = []
        self.detected_entities = []
        self.step_count = 0
        self.done = False
        self.previous_actions = []
        self.total_redacted_chars = 0

        return self._build_observation()

    def step(self, action: RedactionAction) -> Tuple[RedactionObservation, RedactionReward, bool, dict]:
        if self.current_doc is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        old_state = self.state()
        invalid_action = False

        self.step_count += 1

        if action.action_type == ActionType.NEXT_CHUNK:
            step_size = max(1, self.window_size // 2)
            max_cursor = max(0, len(self.current_doc["text"]) - 1)
            self.cursor = min(max_cursor, self.cursor + step_size)

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
                label = self._best_label(start, end)
                self.detected_entities.append(
                    PIIEntity(label=label, start=start, end=end, text=entity_text)
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
            reward.total = round(reward.total - 1.0, 4)

        observation = self._build_observation()
        info = {
            "step": self.step_count,
            "invalid_action": invalid_action,
            "task_id": self.current_task["task_id"],
        }
        return observation, reward, self.done, info

    def state(self) -> dict:
        return {
            "task_id": self.current_task["task_id"] if self.current_task else None,
            "document_id": self.current_doc["id"] if self.current_doc else None,
            "cursor": self.cursor,
            "step_count": self.step_count,
            "done": self.done,
            "redacted_spans": list(self.redacted_spans),
            "total_redacted_chars": self.total_redacted_chars,
        }

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

    def compute_reward(self, action: RedactionAction, old_state: dict) -> RedactionReward:
        del old_state
        components: Dict[str, float] = {}

        if action.action_type == ActionType.NEXT_CHUNK:
            components["progress_bonus"] = 0.1
        else:
            components["progress_bonus"] = 0.0

        tp_bonus = 0.0
        fp_penalty = 0.0

        if action.action_type == ActionType.REDACT and action.start is not None and action.end is not None:
            span = (action.start, action.end)
            best_iou = (
                max(self._iou(span, (gt.start, gt.end)) for gt in self.ground_truth)
                if self.ground_truth
                else 0.0
            )

            if best_iou > 0.6:
                tp_bonus = 1.0 * best_iou
                if any(self._iou(span, existing) > 0.8 for existing in self.redacted_spans[:-1]):
                    tp_bonus = 0.1
            else:
                fp_penalty = -0.5

        components["tp_bonus"] = round(tp_bonus, 4)
        components["fp_penalty"] = round(fp_penalty, 4)

        fn_penalty = 0.0
        utility_penalty = 0.0
        if self.done:
            missed = len(
                [
                    gt
                    for gt in self.ground_truth
                    if not any(
                        self._iou((gt.start, gt.end), det) > 0.6
                        for det in self.detected_spans
                    )
                ]
            )
            fn_penalty = -2.0 * missed

            over_ratio = self.total_redacted_chars / len(self.current_doc["text"])
            if over_ratio > 0.25:
                utility_penalty = -3.0 * (over_ratio - 0.25)

        components["fn_penalty"] = round(fn_penalty, 4)
        components["utility_penalty"] = round(utility_penalty, 4)

        total = round(sum(components.values()), 4)

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
            components=components,
            f1_score_current=self._compute_running_f1(),
            remaining_entities=remaining_entities,
        )

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
            matching = [span for span in self.redacted_spans if span[0] < end and span[1] > i]
            if not matching:
                parts.append(text[i:end])
                break

            next_span = min(matching, key=lambda s: s[0])
            s0, s1 = max(start, next_span[0]), min(end, next_span[1])

            if i < s0:
                parts.append(text[i:s0])
            parts.append("[REDACTED]")
            i = s1

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
                if det.label != gt.label:
                    continue
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
            return f"REDACT({action.start},{action.end})"
        return action.action_type.value
