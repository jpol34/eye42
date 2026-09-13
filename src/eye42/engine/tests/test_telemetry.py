from __future__ import annotations

from eye42.engine.bidding import BidKind
from eye42.engine.events import BidMade, Passed, TilePlayed, TrumpCalled
from eye42.engine.game import GameState
from eye42.engine.hand import HandState
from eye42.engine.repair import Irregularity, IrregularityKind
from eye42.engine.tiles import Tile
from eye42.telemetry import EventStore


def test_event_store_round_trip(tmp_path):
    store = EventStore(tmp_path / "events.db")
    store.log_event(BidMade(player=1, amount=30), session_id="s1", hand_index=0)
    store.log_irregularity(
        Irregularity(kind=IrregularityKind.OUT_OF_TURN, reason="test", player=2),
        session_id="s1",
        hand_index=0,
    )

    events = store.events_for_session("s1")
    assert len(events) == 1
    assert events[0]["kind"] == "BidMade"

    tail = store.recent("s1", limit=10)
    kinds = {row["kind"] for row in tail}
    assert kinds == {"BidMade", "OUT_OF_TURN"}


def test_ingest_dispatches_and_logs_every_event(tmp_path):
    store = EventStore(tmp_path / "events.db")
    hand = HandState(dealer=3, store=store, session_id="s1", hand_index=0)

    for seat in (0, 1, 2):
        hand.ingest(Passed(player=seat))
    hand.ingest(BidMade(player=3, amount=30))

    assert hand.contract is not None
    assert hand.contract.bidder == 3
    assert hand.contract.kind == BidKind.POINTS

    hand.ingest(TrumpCalled(caller=3, trump=5))
    assert hand.trump_tracker.confirmed == 5

    hand.ingest(TilePlayed(player=3, tile=Tile.of(5, 5)))
    assert hand.played_tiles == {Tile.of(5, 5)}

    logged_kinds = [row["kind"] for row in store.events_for_session("s1")]
    assert logged_kinds == ["Passed", "Passed", "Passed", "BidMade", "TrumpCalled", "TilePlayed"]


def test_ingest_rejects_unknown_event_type(tmp_path):
    hand = HandState(dealer=0)
    try:
        hand.ingest(object())
    except Exception as exc:  # HandError
        assert "unrecognized event type" in str(exc)
    else:
        raise AssertionError("expected ingest to reject an unrecognized event type")


def test_irregularity_still_logged_via_direct_call(tmp_path):
    """The chokepoint for irregularities is RepairLog itself, so a direct
    method call (not routed through ingest) still logs -- unlike normal
    events, which are only captured via ingest()."""
    store = EventStore(tmp_path / "events.db")
    hand = HandState(dealer=0, store=store, session_id="s1", hand_index=0)

    hand.bid(player=1, amount=30)  # out-of-turn: dealer 0's rotation starts at seat 1... use wrong seat
    hand.bid(player=1, amount=200)  # invalid amount -> rejected bid, logged as an irregularity

    kinds = [row["kind"] for row in store.recent("s1", limit=10)]
    assert "BID_NOT_LEGAL" in kinds


def test_game_new_hand_threads_store_and_session(tmp_path):
    store = EventStore(tmp_path / "events.db")
    game = GameState(store=store, session_id="s1")
    hand = game.new_hand()
    assert hand.store is store
    assert hand.session_id == "s1"
    assert hand.hand_index == 0
