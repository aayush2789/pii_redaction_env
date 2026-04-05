from server.pii_redaction_env_environment import RedactionEnvironment
from models import ActionType, RedactionAction


def test_reset_returns_valid_observation():
    env = RedactionEnvironment(task_id="gdpr_contract_easy")
    obs = env.reset()

    assert obs.task_id == "gdpr_contract_easy"
    assert obs.document_id
    assert obs.cursor_position == 0
    assert obs.document_length > 0
    assert obs.done is False


def test_redact_action_masks_text():
    env = RedactionEnvironment(task_id="gdpr_contract_easy")
    env.reset()

    email = next(entity for entity in env.ground_truth if entity.label == "EMAIL")
    obs, reward, done, _ = env.step(
        RedactionAction(action_type=ActionType.REDACT, start=email.start, end=email.end)
    )

    assert "[REDACTED]" in obs.visible_text or "[REDACT" in obs.visible_text
    assert 0.0 <= reward.total <= 1.0
    assert reward.raw_total > 0.0
    assert done is False


def test_next_chunk_advances_cursor():
    env = RedactionEnvironment(task_id="gdpr_contract_easy", window_size=20)
    env.reset()
    old_cursor = env.cursor

    env.step(RedactionAction(action_type=ActionType.NEXT_CHUNK))

    assert env.cursor > old_cursor


def test_prev_chunk_rewinds_cursor():
    env = RedactionEnvironment(task_id="gdpr_contract_easy", window_size=40)
    env.reset()
    env.step(RedactionAction(action_type=ActionType.NEXT_CHUNK))
    moved_cursor = env.cursor

    env.step(RedactionAction(action_type=ActionType.PREV_CHUNK))

    assert env.cursor < moved_cursor
    assert env.cursor >= 0


def test_invalid_action_penalty():
    env = RedactionEnvironment(task_id="gdpr_contract_easy")
    env.reset()

    _, reward, _, info = env.step(
        RedactionAction(action_type=ActionType.REDACT, start=-1, end=99999)
    )

    assert info["invalid_action"] is True
    assert reward.components["invalid_action_penalty"] == -1.0
    assert reward.raw_total < 0.0
    assert 0.0 <= reward.total <= 1.0


def test_grade_computes_f1_correctly():
    env = RedactionEnvironment(task_id="gdpr_contract_easy")
    env.reset()

    for entity in env.ground_truth:
        env.step(
            RedactionAction(action_type=ActionType.REDACT, start=entity.start, end=entity.end)
        )

    env.step(RedactionAction(action_type=ActionType.FINISH))
    grade = env.grade()

    assert grade.f1_final == 1.0
    assert grade.precision == 1.0
    assert grade.recall == 1.0


def test_episode_terminates_on_finish():
    env = RedactionEnvironment(task_id="gdpr_contract_easy")
    env.reset()

    obs, _, done, _ = env.step(RedactionAction(action_type=ActionType.FINISH))

    assert done is True
    assert obs.done is True


def test_episode_terminates_on_max_steps():
    env = RedactionEnvironment(task_id="gdpr_contract_easy", max_steps=1)
    env.reset()

    _, _, done, _ = env.step(RedactionAction(action_type=ActionType.SKIP))

    assert done is True


def test_masked_window_shows_multiple_redactions():
    env = RedactionEnvironment(task_id="gdpr_contract_easy", window_size=500)
    obs = env.reset(seed=1)
    first = env.ground_truth[0]
    second = env.ground_truth[1] if len(env.ground_truth) > 1 else first

    env.step(RedactionAction(action_type=ActionType.REDACT, start=first.start, end=first.end))
    obs, _, _, _ = env.step(
        RedactionAction(action_type=ActionType.REDACT, start=second.start, end=second.end)
    )

    assert len(obs.redacted_spans) >= 2
    # Length-preserving mask means [REDACTED] or truncated version appears
    assert "[REDACT" in obs.visible_text


def test_reset_seed_is_reproducible():
    env = RedactionEnvironment(task_id="gdpr_contract_easy")
    obs_a = env.reset(seed=123)
    obs_b = env.reset(seed=123)

    assert obs_a.document_id == obs_b.document_id


def test_explainability_bonus_applies_for_matching_justification():
    env = RedactionEnvironment(task_id="gdpr_contract_easy")
    env.reset(seed=0)
    entity = env.ground_truth[0]
    label = entity.label.lower()
    # Build a justification that contains a keyword matching the entity label
    justification_map = {
        "email": "This is an email contact",
        "phone": "This is a phone number",
        "name": "This is a person name",
        "address": "This is a street address",
        "dob": "This is a date of birth",
        "ssn": "This is a social security number",
    }
    justification = justification_map.get(label, f"This is a {label} field")

    _, reward, _, _ = env.step(
        RedactionAction(
            action_type=ActionType.REDACT,
            start=entity.start,
            end=entity.end,
            confidence=0.9,
            justification=justification,
        )
    )

    assert reward.components["explainability_bonus"] == 0.1


def test_navigation_rewards_are_applied():
    env = RedactionEnvironment(task_id="gdpr_contract_easy", window_size=30)
    env.reset(seed=2)

    _, next_reward, _, _ = env.step(RedactionAction(action_type=ActionType.NEXT_CHUNK))
    _, prev_reward, _, _ = env.step(RedactionAction(action_type=ActionType.PREV_CHUNK))

    # Fix 7: NEXT_CHUNK now gives positive progress-scaled reward
    assert next_reward.components["progress_bonus"] >= 0.0
    assert prev_reward.components["progress_bonus"] == -0.05
    assert 0.0 <= next_reward.total <= 1.0
    assert 0.0 <= prev_reward.total <= 1.0


def test_finish_bonus_is_present_and_bounded():
    env = RedactionEnvironment(task_id="gdpr_contract_easy")
    env.reset(seed=0)

    entity = env.ground_truth[0]
    label = entity.label.lower()
    justification_map = {
        "email": "email field",
        "phone": "phone number",
        "name": "person name",
        "address": "address field",
        "dob": "date of birth",
        "ssn": "social security",
    }
    env.step(
        RedactionAction(
            action_type=ActionType.REDACT,
            start=entity.start,
            end=entity.end,
            confidence=0.9,
            justification=justification_map.get(label, label),
        )
    )

    _, reward, done, _ = env.step(RedactionAction(action_type=ActionType.FINISH))

    assert done is True
    assert 0.0 <= reward.total <= 1.0
    assert 0.3 <= reward.components["finish_bonus"] <= 0.7


def test_duplicate_redaction_escalating_penalty():
    """Fix 1: Redacting the same span repeatedly yields escalating penalties."""
    env = RedactionEnvironment(task_id="gdpr_contract_easy")
    env.reset()

    span_start, span_end = 10, 20
    penalties = []
    for _ in range(3):
        _, reward, _, _ = env.step(
            RedactionAction(action_type=ActionType.REDACT, start=span_start, end=span_end)
        )
        penalties.append(reward.components.get("duplicate_penalty", 0.0))

    # First action has no duplicate, second has -0.2, third has -0.4
    assert penalties[0] == 0.0
    assert penalties[1] < 0.0
    assert penalties[2] < penalties[1]


def test_reward_discriminability():
    """Fix 2: TP reward must be meaningfully higher than FP reward."""
    env = RedactionEnvironment(task_id="gdpr_contract_easy")
    env.reset()

    email = next(entity for entity in env.ground_truth if entity.label == "EMAIL")
    _, tp_reward, _, _ = env.step(
        RedactionAction(action_type=ActionType.REDACT, start=email.start, end=email.end)
    )

    env2 = RedactionEnvironment(task_id="gdpr_contract_easy")
    env2.reset()
    _, fp_reward, _, _ = env2.step(
        RedactionAction(action_type=ActionType.REDACT, start=0, end=5)
    )

    assert tp_reward.total - fp_reward.total > 0.15


def test_all_entity_types_present():
    """Fix 4: Every declared entity type should appear in at least one document."""
    import json
    from pathlib import Path

    data_dir = Path(__file__).resolve().parents[1] / "server" / "data"
    all_labels = set()
    for json_file in data_dir.glob("*.json"):
        with open(json_file) as f:
            docs = json.load(f)
            for doc in docs:
                for entity in doc.get("entities", []):
                    all_labels.add(entity["label"])

    expected = {"EMAIL", "PHONE", "SSN", "NAME", "ADDRESS", "DOB"}
    assert expected.issubset(all_labels), f"Missing: {expected - all_labels}"


def test_annotate_action_behavior_and_rewards():
    """Verify ANNOTATE action assigns labels and gives annotation_bonus properly."""
    env = RedactionEnvironment(task_id="gdpr_contract_easy")
    env.reset(seed=0)
    entity = env.ground_truth[0]
    
    # Correct annotation
    obs, reward_correct, _, _ = env.step(
        RedactionAction(
            action_type=ActionType.ANNOTATE,
            start=entity.start,
            end=entity.end,
            label=entity.label
        )
    )
    assert reward_correct.components.get("annotation_bonus", 0.0) == 0.15
    assert len(obs.annotations) == 1
    assert obs.annotations[0]["correct"] is True
    
    # Wrong annotation (on a valid span)
    env.reset(seed=0)
    wrong_label = "DOB" if entity.label != "DOB" else "EMAIL"
    obs, reward_wrong, _, _ = env.step(
        RedactionAction(
            action_type=ActionType.ANNOTATE,
            start=entity.start,
            end=entity.end,
            label=wrong_label
        )
    )
    assert reward_wrong.components.get("annotation_bonus", 0.0) == -0.1
    assert len(obs.annotations) == 1
    assert obs.annotations[0]["correct"] is False
