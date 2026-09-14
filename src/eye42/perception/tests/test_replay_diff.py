from eye42.perception.replay_diff import diff_play_sequences


def _play(hand_index: int, player: int, high: int, low: int) -> dict:
    return {"hand_index": hand_index, "player": player, "tile": {"high": high, "low": low}}


def test_diff_play_sequences_reports_full_match_for_identical_sequences():
    plays = [_play(0, 0, 6, 6), _play(0, 1, 5, 3), _play(0, 2, 4, 4)]
    result = diff_play_sequences(plays, plays)
    assert result == {"matched": 3, "only_in_original": [], "only_in_replay": []}


def test_diff_play_sequences_flags_a_duplicate_only_present_in_the_replay():
    original = [_play(0, 0, 6, 6), _play(0, 1, 5, 3)]
    replay = [_play(0, 0, 6, 6), _play(0, 0, 1, 1), _play(0, 1, 5, 3)]
    result = diff_play_sequences(original, replay)
    assert result["matched"] == 2
    assert result["only_in_original"] == []
    assert result["only_in_replay"] == [(0, 0, 1, 1)]


def test_diff_play_sequences_flags_a_play_missing_from_the_replay():
    original = [_play(0, 0, 6, 6), _play(0, 1, 5, 3), _play(0, 2, 4, 4)]
    replay = [_play(0, 0, 6, 6), _play(0, 2, 4, 4)]
    result = diff_play_sequences(original, replay)
    assert result["matched"] == 2
    assert result["only_in_original"] == [(0, 1, 5, 3)]
    assert result["only_in_replay"] == []


def test_diff_play_sequences_does_not_falsely_match_the_same_tile_across_different_hands():
    """A double-six set has exactly one of each tile per hand, so an
    identical (player, tile) pair legitimately recurs across hands -- the
    diff must not let a real miss in one hand hide behind a coincidental
    match against a different hand's play of the same tile."""
    original = [_play(0, 0, 6, 6), _play(1, 0, 6, 6)]
    replay = [_play(0, 0, 6, 6)]
    result = diff_play_sequences(original, replay)
    assert result["matched"] == 1
    assert result["only_in_original"] == [(1, 0, 6, 6)]
    assert result["only_in_replay"] == []
