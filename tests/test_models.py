from src.models import ActionType, RedactionAction


def test_prev_chunk_action_validates_without_span():
    action = RedactionAction(action_type=ActionType.PREV_CHUNK)

    assert action.action_type == ActionType.PREV_CHUNK
    assert action.start is None
    assert action.end is None
