from src.graders import compute_grade
from src.models import PIIEntity
from src.tasks import TASKS


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


def test_compute_grade_score_is_bounded():
    gt = [PIIEntity(label="EMAIL", start=0, end=10, text="a@b.com")]
    detected = [PIIEntity(label="EMAIL", start=0, end=10, text="a@b.com")]

    grade = compute_grade(
        detected=detected,
        ground_truth=gt,
        document_length=20,
        total_redacted_chars=20,
        task_id="gdpr_contract_easy",
        success_threshold=0.9,
    )

    assert 0.0 <= grade.score <= 1.0


def test_tasks_define_concrete_objectives():
    assert TASKS["gdpr_contract_easy"]["objective"]
    assert TASKS["hipaa_medical_medium"]["objective"]
    assert TASKS["security_logs_hard"]["objective"]
