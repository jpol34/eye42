"""Phase 3 (not implemented yet — depends on real recorded audio to design and
validate against). Interface stubs matching the project plan so real STT/parsing
logic has a slot to drop into.

Pipeline shape (see plan): continuous local VAD + rolling buffer -> Whisper STT
(no cloud API) -> closed-vocabulary bid/trump candidate matching, scored by
bid-rotation plausibility x ASR confidence (NOT raw vocab-match acceptance,
which the plan's re-review flagged as too permissive against ordinary table
talk) -> turn-order speaker attribution, cross-checked against the CV-observed
trick leader (splash/plunge means the leader is the bidder's PARTNER, which is
positive evidence for that bid type, not a mismatch to flag).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from eye42.engine.bidding import BiddingRound

# Trump is an int 0-6 only per confirmed house rules -- no "doubles"/"no-trump"
# vocabulary, so it isn't modeled and shouldn't be added without a house-rule
# change.
CLOSED_VOCAB_BIDS = list(range(30, 43))  # point bids: 30-42, per BiddingRound's own legality check
# Marks bids ("two marks," "four marks") have no fixed numeric ceiling in the
# engine's own bidding legality (BiddingRound._bid_kind_and_amount only
# requires >=1), unlike point bids' fixed 30-42 range -- so there's no
# analogous closed numeric set to define here. What distinguishes a marks bid
# from a point bid in speech is the word itself.
CLOSED_VOCAB_MARKS_WORDS = ("mark", "marks")
CLOSED_VOCAB_TRUMP_CUES = ("low end", "high end")


@dataclass(frozen=True)
class Utterance:
    text: str
    confidence: float  # ASR confidence, 0-1
    start_time: float
    end_time: float


@dataclass(frozen=True)
class BidCandidate:
    utterance: Utterance
    player_guess: int  # from turn-order rotation at the time of the utterance
    amount: Optional[int]
    marks: int
    plausibility: float  # bid-rotation legality score, independent of ASR confidence

    @property
    def combined_score(self) -> float:
        return self.utterance.confidence * self.plausibility


class ContinuousTranscriber:
    """Wraps a local Whisper model with VAD-gated buffering. NOT gated on
    "bidding phase active" (that signal can't exist before any event does —
    see the plan's note on why the original gating design was circular)."""

    def __init__(self, model_size: str = "base") -> None:
        raise NotImplementedError("Phase 3 — needs real audio to validate WER against")

    def transcribe_stream(self, audio_chunk: bytes) -> List[Utterance]:
        raise NotImplementedError


class BidCandidateScorer:
    """Scores an utterance against the closed vocabulary AND the current bidding
    rotation's legality (bids must be >=30, strictly increasing, one per player,
    four per hand, starting left of dealer) -- this is what suppresses ordinary
    table talk that happens to contain a number.
    """

    def __init__(self, bidding: BiddingRound) -> None:
        raise NotImplementedError

    def score(self, utterance: Utterance) -> Optional[BidCandidate]:
        raise NotImplementedError("Phase 3 — needs real transcripts to tune the plausibility scoring")


class TrumpCueResolver:
    """Resolves a weak 'low end'/'high end' verbal cue against the CV-observed
    led tile, feeding eye42.engine.trump_inference.TrumpHypothesisTracker rather
    than setting trump directly -- a cue alone (without the tile) proves nothing.
    """

    def resolve(self, cue_text: str, speaker_guess: int) -> Optional[str]:
        """Returns 'low', 'high', or None if the utterance isn't a trump cue."""
        raise NotImplementedError
