from __future__ import annotations

from eye42.engine.bidding import BiddingRound
from eye42.engine.events import BidMade, Passed
from eye42.speech.bid_parser import (
    BidCandidate,
    BidCandidateScorer,
    TrumpCueResolver,
    Utterance,
    _extract_number,
)


def _utt(text: str, confidence: float = 0.9) -> Utterance:
    return Utterance(text=text, confidence=confidence, start_time=0.0, end_time=1.0)


# ---------------------------------------------------------------------------
# _extract_number
# ---------------------------------------------------------------------------

def test_extract_number_from_digits():
    assert _extract_number("31") == 31
    assert _extract_number("thirty-one points") == 31  # digit-free but hyphenated word form


def test_extract_number_from_spoken_words():
    assert _extract_number("thirty one") == 31
    assert _extract_number("forty two") == 42
    assert _extract_number("thirty") == 30
    assert _extract_number("two marks") == 2


def test_extract_number_returns_none_for_no_number():
    assert _extract_number("nice shot") is None


# ---------------------------------------------------------------------------
# BidCandidateScorer
# ---------------------------------------------------------------------------

def _bidding(dealer: int = 3) -> BiddingRound:
    return BiddingRound(dealer=dealer)  # rotation starts at seat 0


def test_scores_a_legal_point_bid():
    bidding = _bidding()
    scorer = BidCandidateScorer(bidding)
    candidate = scorer.score(_utt("I'll go thirty"))
    assert candidate is not None
    assert candidate.player_guess == 0
    assert candidate.amount == 30
    assert candidate.marks == 0
    assert not candidate.is_pass
    assert candidate.plausibility == 1.0


def test_scores_a_legal_marks_bid():
    bidding = _bidding()
    scorer = BidCandidateScorer(bidding)
    candidate = scorer.score(_utt("We're going two marks"))
    assert candidate is not None
    assert candidate.marks == 2
    assert candidate.amount is None


def test_a_tens_word_or_multi_digit_number_before_marks_is_rejected():
    """A marks count is always a single small number, never a tens+units
    pair a point bid can be -- a stray "thirty"/"30" landing next to "marks"
    (misheard cross-talk, plausible per RESEARCH.md) must not be read as a
    30-mark bid, which would wildly outrank every real bid in the game."""
    bidding = _bidding()
    scorer = BidCandidateScorer(bidding)
    assert scorer.score(_utt("no, thirty, marks")) is None
    assert scorer.score(_utt("30 marks")) is None


def test_scores_a_pass():
    bidding = _bidding()
    scorer = BidCandidateScorer(bidding)
    for text in ("I had to pass", "That's the pass", "To pass also"):
        candidate = scorer.score(_utt(text))
        assert candidate is not None, text
        assert candidate.is_pass is True, text


def test_pass_embedded_mid_utterance_is_not_a_pass():
    """A real risk a bare substring match would miss: "pass" used as an
    ordinary word well before the end of the utterance, not as a bid
    declaration -- every genuine pass found in real transcripts lands at or
    within one word of the end (RESEARCH.md)."""
    bidding = _bidding()
    scorer = BidCandidateScorer(bidding)
    assert scorer.score(_utt("I'll pass the salt please")) is None
    assert scorer.score(_utt("let's just pass on this one and move along")) is None


def test_a_bid_shaped_number_in_a_long_sentence_is_not_a_bid():
    """The point-bid equivalent of the marks-bid false positive documented
    in RESEARCH.md: a legal bid amount mentioned in ordinary conversation,
    not as an actual bid declaration, must not score as one."""
    bidding = _bidding()
    scorer = BidCandidateScorer(bidding)
    assert scorer.score(_utt("I think you need thirty one to beat him")) is None


def test_ordinary_table_talk_with_a_number_is_not_a_bid():
    """The whole reason plausibility scoring exists: raw vocab-match
    acceptance would treat this as a legal bid, which real table talk
    absolutely contains."""
    bidding = _bidding()
    scorer = BidCandidateScorer(bidding)
    # "thirty" is technically extractable, but the current bidder hasn't
    # been established as speaking here -- this alone isn't the guard
    # (turn-order is); the guard is a NUMBER that isn't vocabulary-shaped.
    assert scorer.score(_utt("that was a close one, nice")) is None


def test_player_guess_always_comes_from_current_bidder_not_the_utterance():
    """Turn-order attribution, never true diarization: whatever seat is
    expected to bid next is who a legal-shaped utterance gets attributed to,
    since a single shared table mic can't otherwise say who spoke."""
    bidding = _bidding()  # seat 0 bids first
    scorer = BidCandidateScorer(bidding)
    candidate = scorer.score(Utterance(text="thirty", confidence=0.9, start_time=5.0, end_time=6.0))
    assert candidate is not None
    assert candidate.player_guess == 0


def test_a_bid_below_the_current_high_bid_is_rejected():
    bidding = _bidding()
    scorer = BidCandidateScorer(bidding)
    first = scorer.score(_utt("thirty one"))
    assert first is not None
    bidding.record_bid(BidMade(player=0, amount=31))

    # Seat 1 now tries a lower bid than the standing 31 -- not legal.
    second = scorer.score(_utt("thirty"))
    assert second is None


def test_score_returns_none_once_bidding_is_done():
    bidding = BiddingRound(dealer=3)
    for seat in (0, 1, 2, 3):
        bidding.record_pass(Passed(player=seat))
    assert bidding.is_done  # forced-30

    scorer = BidCandidateScorer(bidding)
    assert scorer.score(_utt("thirty")) is None


def test_combined_score_multiplies_confidence_and_plausibility():
    candidate = BidCandidate(
        utterance=_utt("thirty", confidence=0.7), player_guess=0, amount=30, marks=0, plausibility=1.0,
    )
    assert candidate.combined_score == 0.7


# ---------------------------------------------------------------------------
# TrumpCueResolver
# ---------------------------------------------------------------------------

def test_resolves_low_end_cue():
    assert TrumpCueResolver().resolve("the low end", speaker_guess=0) == "low"


def test_resolves_high_end_cue():
    assert TrumpCueResolver().resolve("HIGH END", speaker_guess=0) == "high"


def test_non_cue_text_resolves_to_none():
    assert TrumpCueResolver().resolve("nice play", speaker_guess=0) is None


def test_cue_phrase_embedded_in_a_long_unrelated_sentence_is_not_a_cue():
    """The same false-positive shape as a long bid-shaped sentence
    (RESEARCH.md): ordinary conversation can contain "low end"/"high end"
    as a plain substring without being anywhere near a trump call."""
    resolver = TrumpCueResolver()
    assert resolver.resolve("that's the low end of the market for sure", speaker_guess=0) is None
