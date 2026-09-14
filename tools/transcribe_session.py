#!/usr/bin/env python3
"""Batch-transcribes a recorded session's WAV (e.g. one of
``tools/live_view.py``'s ``AudioRecorder`` outputs) and recovers a candidate
bid sequence plus any trump cue, for a human to confirm against the video --
the same post-hoc-confirmation philosophy every other inference in this
project already follows, never an automatic, unconfirmed engine input.

This does not feed events into a live ``HandState`` -- perception's tile-play
events and this tool's speech-derived bid/trump events have no established
shared timeline yet (see ROADMAP.md's "no shared clock/frame convention"
entry), so reconciling them automatically is future work. What this gives
today: run it on a session's audio after the fact, and manually re-enter the
recovered bids/trump into ``tools/live_view.py``'s REPL (or correct them)
instead of re-listening to the whole recording by ear.

Usage:
    python tools/transcribe_session.py <audio.wav> [--dealer SEAT]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eye42.engine.bidding import BiddingRound  # noqa: E402
from eye42.engine.events import BidMade, Passed  # noqa: E402
from eye42.speech.bid_parser import BidCandidateScorer, TrumpCueResolver  # noqa: E402
from eye42.speech.transcribe import transcribe_wav  # noqa: E402


def recover_bids(bidding: BiddingRound, utterances) -> list:
    """Feeds ``utterances`` (chronological) through a scorer wrapping
    ``bidding``, applying each plausible candidate to that same round so
    later utterances are scored against the up-to-date rotation -- an
    implausible/rejected utterance (ordinary table talk, a duplicate ASR
    segment of an already-accepted bid, ...) is silently skipped, same as
    ``HandState.bid``'s own non-strict philosophy."""
    scorer = BidCandidateScorer(bidding)
    accepted = []
    for utterance in utterances:
        if bidding.is_done:
            break
        candidate = scorer.score(utterance)
        if candidate is None:
            continue
        accepted.append(candidate)
        if candidate.is_pass:
            bidding.record_pass(Passed(player=candidate.player_guess))
        else:
            bidding.record_bid(BidMade(player=candidate.player_guess, amount=candidate.amount or 0, marks=candidate.marks))
    return accepted


def recover_trump_cue(utterances) -> list:
    resolver = TrumpCueResolver()
    found = []
    for utterance in utterances:
        cue = resolver.resolve(utterance.text, speaker_guess=-1)
        if cue is not None:
            found.append((utterance, cue))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", help="WAV file to transcribe")
    parser.add_argument("--dealer", type=int, default=0, help="dealer seat 0-3 (default 0)")
    args = parser.parse_args()

    print(f"transcribing {args.audio} ...")
    utterances = transcribe_wav(args.audio)
    print(f"{len(utterances)} utterance(s) transcribed")

    bidding = BiddingRound(dealer=args.dealer)
    candidates = recover_bids(bidding, utterances)

    print("\nrecovered bid sequence:")
    for c in candidates:
        what = "pass" if c.is_pass else (f"{c.marks} marks" if c.marks else str(c.amount))
        print(
            f"  [{c.utterance.start_time:6.1f}s] seat {c.player_guess}: {what}"
            f"  (confidence={c.utterance.confidence:.2f}, text={c.utterance.text!r})"
        )
    if bidding.contract is not None:
        print(f"\ncontract: {bidding.contract}")
    else:
        print("\nbidding not resolved from this audio -- fewer than 4 bid/pass utterances found")

    cues = recover_trump_cue(utterances)
    if cues:
        print("\npossible trump cues:")
        for utterance, cue in cues:
            print(f"  [{utterance.start_time:6.1f}s] {cue!r}  (text={utterance.text!r})")


if __name__ == "__main__":
    main()
