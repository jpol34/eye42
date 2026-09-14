"""Closed-vocabulary bid/trump-cue parsing from Whisper transcripts.

Validated against two real ~9-minute recorded gameplay sessions
(faster-whisper "base" model, `vad_filter=True`, CPU-only) -- see
RESEARCH.md's "Speech: real-audio Whisper validation" section for the
transcript evidence this design is built against. Marks-bid and pass
language ("two marks", "That's the pass") transcribes reliably; no
point-bid ("thirty", ...) examples turned up in the sampled audio, so that
half of the vocabulary is unvalidated against real speech -- see the same
RESEARCH.md section before trusting it blindly on a real session.

Pipeline shape: `speech.transcribe.transcribe_wav` (real Whisper, batch --
not live streaming; see that module's docstring for why) -> `Utterance` ->
`BidCandidateScorer`/`TrumpCueResolver` (this module, closed-vocabulary
matching scored by bid-rotation plausibility x ASR confidence, NOT raw
vocab-match acceptance, since ordinary table talk containing a number must
not read as a bid) -> turn-order speaker attribution (`player_guess` is
always `BiddingRound.current_bidder`, never true diarization -- a single
shared table mic makes real diarization both hard and unnecessary here),
cross-checked against the CV-observed trick leader (splash/plunge means the
leader is the bidder's PARTNER, which is positive evidence for that bid
type, not a mismatch to flag -- see `engine.hand._maybe_infer_splash_or_plunge`).
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import List, Optional

from eye42.engine.bidding import BiddingError, BiddingRound
from eye42.engine.events import BidMade, Passed

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
CLOSED_VOCAB_PASS_WORDS = ("pass",)
CLOSED_VOCAB_TRUMP_CUES = ("low end", "high end")

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
}
_TENS = {"thirty": 30, "forty": 40}
_PUNCT_RE = re.compile(r"[.,!?]+$")


def _tokenize(text: str) -> List[str]:
    return [_PUNCT_RE.sub("", t) for t in text.lower().replace("-", " ").split()]


def _number_from_token(token: str) -> Optional[int]:
    if token.isdigit():
        return int(token)
    if token in _UNITS:
        return _UNITS[token]
    if token in _TENS:
        return _TENS[token]
    return None


def _small_number_from_token(token: str) -> Optional[int]:
    """A single-digit count only (0-9) -- deliberately narrower than
    ``_number_from_token``, which also matches tens words ("thirty") and
    multi-digit strings ("30") appropriate for a point bid but not a marks
    count. A marks count is always spoken as one small word/digit, never a
    tens+units pair; without this restriction, a nearby tens word or a
    misheard two-digit number (e.g. a stray "30" or "thirty" landing next to
    "marks") would be read as marks=30 -- BiddingRound has no marks upper
    bound, so that would silently outrank every real bid in the hand."""
    if token.isdigit() and len(token) == 1:
        return int(token)
    return _UNITS.get(token)


_MAX_POINT_BID_TOKENS = 5  # a real bid declaration is short and number-focused
# ("Thirty", "I'll go thirty one") -- a longer utterance is more likely
# ordinary conversation that happens to contain a legal bid-shaped number
# than an actual bid, the same false-positive shape RESEARCH.md documents
# for marks bids (see _number_immediately_before below).


def _extract_number(text: str) -> Optional[int]:
    """Finds a point-bid number in ``text`` (digit or spoken tens+units word
    form -- Whisper renders numbers inconsistently as either, "thirty one"
    vs. "31", confirmed by real transcripts, RESEARCH.md), only if the whole
    utterance is short enough to plausibly BE a bid declaration rather than
    ordinary conversation that happens to contain a bid-shaped number
    somewhere in it. For marks bids, use ``_number_immediately_before``
    instead, which scopes the search near the marks word rather than by
    utterance length (there's no equivalent anchor word for a bare point
    bid, and a marks count is always a single small number, never a
    tens+units pair, so it doesn't need this combining logic)."""
    tokens = _tokenize(text)
    if len(tokens) > _MAX_POINT_BID_TOKENS:
        return None
    for i, token in enumerate(tokens):
        if token in _TENS:
            total = _TENS[token]
            if i + 1 < len(tokens) and tokens[i + 1] in _UNITS:
                total += _UNITS[tokens[i + 1]]
            return total
        number = _number_from_token(token)
        if number is not None:
            return number
    return None


def _number_immediately_before(tokens: List[str], index: int, window: int = 2) -> Optional[int]:
    """Looks only at the ``window`` tokens right before ``tokens[index]``
    (a marks-word position) for a single-token number (a marks count is
    always 1-4, never a spoken tens+units pair like a point bid can be) --
    never anywhere else in the utterance. An unrelated number mentioned
    elsewhere in the same segment (e.g. players reciting tile pips: "the six
    six five four ... I gave you the mark") must not be misread as the mark
    count -- only a number directly adjacent to the marks word counts (see
    RESEARCH.md's "Speech: real-audio Whisper validation" for the transcript
    this guards against). Checked closest-token-first, since that's the one
    that actually modifies the marks word."""
    before = tokens[max(0, index - window):index]
    for token in reversed(before):
        number = _small_number_from_token(token)
        if number is not None:
            return number
    return None


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
    is_pass: bool = False

    @property
    def combined_score(self) -> float:
        return self.utterance.confidence * self.plausibility


class BidCandidateScorer:
    """Scores an utterance against the closed vocabulary AND the current bidding
    rotation's legality (bids must be >=30, strictly increasing, one per player,
    four per hand, starting left of dealer) -- this is what suppresses ordinary
    table talk that happens to contain a number.

    ``bidding`` must be the SAME live ``BiddingRound`` instance the caller's
    ``HandState`` owns (e.g. ``hand.bidding``), scored incrementally in step
    with actually-accepted bids -- not a separate or snapshotted round, or
    plausibility gets checked against the wrong rotation state. Legality
    itself is checked via a deep-copied trial call into the real
    ``BiddingRound`` machinery (never by reimplementing its rank/rotation
    rules here), so this stays correct if that machinery's rules ever change.
    """

    def __init__(self, bidding: BiddingRound) -> None:
        self.bidding = bidding

    def score(self, utterance: Utterance) -> Optional[BidCandidate]:
        current_bidder = self.bidding.current_bidder
        if current_bidder is None:
            return None  # bidding already resolved -- nothing to score against

        tokens = _tokenize(utterance.text)
        # A genuine pass declaration puts "pass" at or right at the end of
        # the utterance ("I had to pass", "That's the pass", "To pass
        # also" -- confirmed against real transcripts, RESEARCH.md);
        # requiring it within the last two tokens rather than matching it
        # anywhere rejects an embedded, unrelated use ("I'll pass the
        # salt") that a bare substring match would wrongly accept.
        pass_positions = [i for i, token in enumerate(tokens) if token in CLOSED_VOCAB_PASS_WORDS]
        is_pass = bool(pass_positions) and pass_positions[-1] >= len(tokens) - 2
        marks: int = 0
        amount: Optional[int] = None

        if not is_pass:
            mark_positions = [i for i, token in enumerate(tokens) if token in CLOSED_VOCAB_MARKS_WORDS]
            if mark_positions:
                # Only the tokens immediately before the marks word count --
                # see _number_immediately_before's docstring for the real
                # false positive this guards against.
                number = _number_immediately_before(tokens, mark_positions[0])
                if number is None:
                    return None
                marks = number
            else:
                number = _extract_number(utterance.text)
                if number is None or number not in CLOSED_VOCAB_BIDS:
                    return None
                amount = number

        trial = copy.deepcopy(self.bidding)
        try:
            if is_pass:
                trial.record_pass(Passed(player=current_bidder))
            else:
                trial.record_bid(BidMade(player=current_bidder, amount=amount or 0, marks=marks))
        except BiddingError:
            return None  # not legal in the current rotation -- ordinary table talk, drop it

        return BidCandidate(
            utterance=utterance,
            player_guess=current_bidder,
            amount=amount,
            marks=marks,
            plausibility=1.0,
            is_pass=is_pass,
        )


_MAX_CUE_UTTERANCE_TOKENS = 5  # a real cue is a short, dedicated phrase ("the
# low end", "I think high end") -- the same false-positive shape as a long
# bid-shaped sentence (RESEARCH.md): ordinary conversation ("that's the low
# end of the market") can contain "low end"/"high end" as a plain substring
# without being anywhere near a trump call.


class TrumpCueResolver:
    """Resolves a weak 'low end'/'high end' verbal cue against the CV-observed
    led tile, feeding eye42.engine.trump_inference.TrumpHypothesisTracker rather
    than setting trump directly -- a cue alone (without the tile) proves nothing.
    """

    def resolve(self, cue_text: str, speaker_guess: int) -> Optional[str]:
        """Returns 'low', 'high', or None if the utterance isn't a trump cue."""
        if len(_tokenize(cue_text)) > _MAX_CUE_UTTERANCE_TOKENS:
            return None
        normalized = cue_text.strip().lower()
        for cue in CLOSED_VOCAB_TRUMP_CUES:
            if cue in normalized:
                return "low" if cue.startswith("low") else "high"
        return None
