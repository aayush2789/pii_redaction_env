from src.graders import compute_grade
from src.models import PIIEntity


def test_compute_grade_matches_expected_f1():
    gt = [PIIEntity(label="EMAIL", start=10, end=28, text="john@example.com")]
    detected = [PIIEntity(label="EMAIL", start=10, end=28, text="john@example.com")]

    grade = compute_grade(
        detected=detected,
        ground_truth=gt,
        document_length=100,
        total_redacted_chars=18,
        task_id="gdpr_contract_easy",
        success_threshold=0.9,
    )

    assert grade.f1_final == 1.0
    assert grade.utility_score == 1.0
    assert grade.score == 1.0
    assert grade.success is True


def test_compute_grade_utility_penalty_triggers():
    gt = [PIIEntity(label="EMAIL", start=10, end=28, text="john@example.com")]
    detected = [PIIEntity(label="EMAIL", start=10, end=28, text="john@example.com")]

    grade = compute_grade(
        detected=detected,
        ground_truth=gt,
        document_length=100,
        total_redacted_chars=60,
        task_id="gdpr_contract_easy",
        success_threshold=0.9,
    )

    assert grade.utility_score < 1.0
    assert grade.score < 1.0
