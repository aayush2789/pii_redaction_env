from src.environment import RedactionEnvironment
from src.models import ActionType, RedactionAction


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

    assert "[REDACTED]" in obs.visible_text
    assert reward.total > 0
    assert done is False


def test_next_chunk_advances_cursor():
    env = RedactionEnvironment(task_id="gdpr_contract_easy", window_size=20)
    env.reset()
    old_cursor = env.cursor

    env.step(RedactionAction(action_type=ActionType.NEXT_CHUNK))

    assert env.cursor > old_cursor


def test_invalid_action_penalty():
    env = RedactionEnvironment(task_id="gdpr_contract_easy")
    env.reset()

    _, reward, _, info = env.step(
        RedactionAction(action_type=ActionType.REDACT, start=-1, end=99999)
    )

    assert info["invalid_action"] is True
    assert reward.components["invalid_action_penalty"] == -1.0


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
