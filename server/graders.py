from typing import Dict, List, Tuple

try:
    from ..models import PIIEntity, TaskGrade
except ImportError:
    from models import PIIEntity, TaskGrade


def _iou(span_a: Tuple[int, int], span_b: Tuple[int, int]) -> float:
    a0, a1 = span_a
    b0, b1 = span_b
    intersection = max(0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return (intersection / union) if union > 0 else 0.0


def _match_counts(
    detected: List[PIIEntity], ground_truth: List[PIIEntity]
) -> Tuple[int, int, int, int, int]:
    """Returns (tp, fp, fn, label_correct, label_total).

    Matching is IoU-only (relaxed). For each matched pair, we also check
    whether the detected label equals the ground-truth label to compute
    a separate label-accuracy metric.
    """
    matched_gt = set()
    tp = 0
    label_correct = 0
    label_total = 0

    for det in detected:
        best_idx = None
        best_iou = 0.0
        for idx, gt in enumerate(ground_truth):
            if idx in matched_gt:
                continue
            # Relaxed matching: IoU-only, no label requirement
            score = _iou((det.start, det.end), (gt.start, gt.end))
            if score > 0.6 and score > best_iou:
                best_iou = score
                best_idx = idx

        if best_idx is not None:
            matched_gt.add(best_idx)
            tp += 1
            # Track label accuracy for matched spans
            label_total += 1
            if det.label == ground_truth[best_idx].label:
                label_correct += 1

    fp = max(0, len(detected) - tp)
    fn = max(0, len(ground_truth) - tp)
    return tp, fp, fn, label_correct, label_total


def compute_grade(
    detected: List[PIIEntity],
    ground_truth: List[PIIEntity],
    document_length: int,
    total_redacted_chars: int,
    task_id: str = "unknown",
    success_threshold: float = 0.0,
) -> TaskGrade:
    tp, fp, fn, label_correct, label_total = _match_counts(detected, ground_truth)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    # Label accuracy: fraction of matched spans with correct PII type label
    label_accuracy = (label_correct / label_total) if label_total > 0 else 0.0

    over_redaction_ratio = (total_redacted_chars / document_length) if document_length else 0.0
    if over_redaction_ratio > 0.25:
        utility_score = max(0.0, 1 - 2 * (over_redaction_ratio - 0.25))
    else:
        utility_score = 1.0

    # Weighted score: 55% span detection, 15% label accuracy, 30% utility
    score = max(0.0, min(1.0, 0.55 * f1 + 0.15 * label_accuracy + 0.30 * utility_score))
    success = score >= success_threshold

    components: Dict[str, float] = {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "label_correct": float(label_correct),
        "label_total": float(label_total),
        "over_redaction_ratio": round(over_redaction_ratio, 6),
        "success_threshold": success_threshold,
    }

    return TaskGrade(
        task_id=task_id,
        score=round(score, 4),
        f1_final=round(max(0.0, min(1.0, f1)), 4),
        precision=round(max(0.0, min(1.0, precision)), 4),
        recall=round(max(0.0, min(1.0, recall)), 4),
        utility_score=round(max(0.0, min(1.0, utility_score)), 4),
        label_accuracy=round(max(0.0, min(1.0, label_accuracy)), 4),
        success=success,
        components=components,
    )
