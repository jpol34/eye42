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

from eye42.engine.bidding import PLUNGE_MIN_MARKS, BiddingRound

# Trump is an int 0-6 only per confirmed house rules -- no "doubles"/"no-trump"
# vocabulary, so it isn't modeled and shouldn't be added without a house-rule
# change.
#
# Point bids only.
CLOSED_VOCAB_BIDS = list(range(30, 43))
# Marks bids: 1 through PLUNGE_MIN_MARKS (the highest realistic mark amount
# under the confirmed house rules) -- covers plain marks bids as well as the
# spoken amount of a splash/plunge bid. Splash vs. plunge is never
# distinguished from the amount alone (see engine.hand's live-play inference,
# which resolves that from who actually leads the first trick), so no
# separate "splash"/"plunge" vocabulary entries are needed here.
CLOSED_VOCAB_MARKS = list(range(1, PLUNGE_MIN_MARKS + 1))
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
