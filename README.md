# eye42

A passive camera + microphone system for reconstructing a full game of **Texas 42** (the domino
trick-taking game) — bidding, trump, every tile played, and score — without changing how the game
is actually played. No RFID, no tapping tiles, no wearables. A camera watches the table, a
microphone picks up table talk, and everything else is inferred.

The hard part isn't detecting dominoes — it's that real hands don't always play out cleanly. A
team can become mathematically "set" before all 7 tricks are played, players often concede a hand
informally well before that's certain, and trump is sometimes never stated out loud at all.
`eye42` is built to recognize and score these situations correctly instead of guessing or hanging.

## Status

Early development — see the project plan for the phased build approach (rules engine first,
offline vision pipeline second, speech layer third, live dashboard last).

## Install (development)

```bash
pip install -e ".[dev]"
```

## License

MIT
