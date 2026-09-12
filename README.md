# eye42

A camera and microphone system that watches a real game of Texas 42 (the domino
trick-taking game) and reconstructs the full game state from it: bids, trump,
every tile played, and the score. Nothing about how the game is played
changes. No tapping tiles, no RFID chips, no wearables, no app anyone at the
table has to touch. A camera looks at the table, a microphone hears the
bidding, and everything else is worked out from that.

## Why this is hard

Detecting which domino is which isn't the interesting problem. The interesting
problem is that a real hand of 42 doesn't always play out in a clean,
easy-to-parse sequence:

- A team can become mathematically unable to make their bid before all 7
  tricks are played, and a normal table just stops the hand there instead of
  finishing it out.
- Players often concede informally, on a hunch the outcome is decided, well
  before the arithmetic actually proves it.
- Trump is frequently never stated out loud. People lead a double, or say
  nothing, and everyone at the table just knows what happened. Software
  watching the video doesn't get that for free.
- People misplay. Someone leads out of turn, or plays off-suit when they could
  have followed, and it either gets caught and corrected on the spot or
  nobody notices until three tricks later when someone's counting their own
  tiles and something doesn't add up.

A tracker that can't handle any of that isn't usable at a real table. The
design goal for this project is that none of it should ever cause the system
to crash or hang. Messy, human hands are the normal case, not an edge case,
and the software has to keep working through them.

## How it's built

The project is built in phases, each one validated against real recorded
footage before moving to the next, cheapest thing that could work first:

1. **Rules engine.** A pure Python, dependency-free engine driven by plain
   events (a bid, a pass, a trump call, a tile played). It knows the full
   rules of 42: standard bidding and trick play, splash and plunge, the
   forced-30 rule on an all-pass, count-tile scoring, and when a hand becomes
   mathematically set. It also carries a weighted trump tracker for the
   common case where trump is never announced, and it does not throw an
   exception when a human does something a computer would call illegal, it
   logs the irregularity and keeps the hand moving. **This part is built and
   tested.**
2. **Vision.** Homography-rectify the table from a fixed camera angle, detect
   and classify each tile, work out who physically played it, and segment
   individual plays and trick sweeps from raw video. Not started, waiting on
   real recorded game footage to design against instead of guessing.
3. **Speech.** Local, on-device transcription (no cloud calls, recordings of
   friends never leave the machine) to pick bids and trump calls out of table
   talk, checked against what's legal at that point in the bidding. Not
   started.
4. **Live dashboard.** A small local web page showing the current bid, trump,
   trick, and score while a game is in progress, plus a live "how likely is
   this bid to make it" probability bar in the style of a poker broadcast's
   equity display. The probability math for this is already built as part of
   the rules engine; the dashboard itself is not.

See the code for the exact module breakdown. The short version:
`engine/` is the rules engine and is where almost all of the work so far has
gone, `perception/` and `speech/` are interface stubs waiting on real
footage, and `tools/crop_rectify.py` is a disposable script for checking that
a camera setup actually resolves domino pips before committing a real game
night to it.

## Status

Phase 1 (the rules engine) is done and tested: 70 tests covering normal play,
splash and plunge, the forced-30 rule, mathematically set hands, redeals,
trump inference when nobody states it, and a range of real-world misplays
(out-of-turn plays, revokes, wrong-seat trump calls, bad deals) that the
engine logs and recovers from instead of crashing on. Everything past that is
either a stub or not started, because it depends on real recorded game
footage that hasn't been captured yet.

## Install

```bash
pip install -e ".[dev]"
```

## Run the tests

```bash
pytest
```

## Try it

```python
from eye42.engine.hand import HandState
from eye42.engine.events import TilePlayed
from eye42.engine.tiles import Tile

hand = HandState(dealer=3)
hand.bid(0, 30)
hand.bid_pass(1)
hand.bid_pass(2)
hand.bid_pass(3)
hand.call_trump(0, trump=6)

hand.play_tile(TilePlayed(player=0, tile=Tile.of(6, 6)))
hand.play_tile(TilePlayed(player=1, tile=Tile.of(1, 0)))
hand.play_tile(TilePlayed(player=2, tile=Tile.of(2, 0)))
hand.play_tile(TilePlayed(player=3, tile=Tile.of(3, 0)))

print(hand.tricks[0].winner)  # 0
```

## License

MIT
