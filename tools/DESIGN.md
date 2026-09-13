# Design notes: `tools/live_view.py` diagnostic page

Scope: `tools/static/live_view.html` only — the live test-session viewer. This
is not a product design system; eye42 has no end-user UI yet (see the root
`ROADMAP.md` for the planned Phase 4 dashboard, which this page is explicitly
not a preview of). This page exists so a human or an AI reviewing a real
recorded test session can see the camera feed, the engine's current game
state, and the recent log tail together, at a glance, in a screenshot.

## Concept

Grounded in the actual subject: a Texas 42 table — felt, ivory-faced tiles
with black pips, brass fittings — rather than a generic dashboard look. Three
panels styled like an instrument panel/scorer's desk: the camera feed as the
visual anchor, a running ledger of game state beside it, and a scrolling
audit log docked along the bottom.

## Palette (dark, default)

- `--bg: #16241c` — felt green
- `--panel: #1e3327` — panel surface, one step lighter
- `--panel-alt: #24392c` — secondary panel surface (log tail)
- `--ink: #f2ead9` — ivory (primary text)
- `--muted: #9db0a1` — dulled ivory-green (secondary text)
- `--accent: #c98a3b` — brass (trump/contract highlights)
- `--alert: #d1573f` — terracotta-red, reserved for open/unresolved
  irregularities — the one color that means "look at this"
- `--rule: #35503f` — dividers

## Palette (light)

- `--bg: #f2ead9` — ivory/parchment
- `--panel: #ffffff`
- `--panel-alt: #ece0c5`
- `--ink: #1c2b21` — felt green (primary text)
- `--muted: #5b6b5e`
- `--accent: #8a5a1f` — brass, darkened for contrast on a light ground
- `--alert: #a63b26`
- `--rule: #cbb98f`

Both defined as CSS custom properties on `:root`; the light set applies under
`prefers-color-scheme: light` (dark is the default, since this is most often
checked next to a physically dim table/camera setup).

## Type

Two system-font roles, no external font requests (this page makes no network
calls beyond its own three endpoints):
- Headings/panel labels: system serif stack (`ui-serif, Georgia, ...`) — a
  ledger/scorecard feel for section titles.
- Everything data-shaped (seat numbers, tile strings, timestamps, log rows):
  system monospace stack (`ui-monospace, SFMono-Regular, ...`) — genuinely
  functional here, not decorative, since the content is columnar/tabular.

Labels are sentence case, not tracked-out capitals.

## Layout

Two-column grid on wide viewports (camera left/large, game-state ledger
right), log tail docked full-width along the bottom; stacks to a single
column under ~700px. No interactivity, no animation beyond a subtle pulse on
the "open questions" panel when a new one appears, since that state is the
one thing on this page that should visibly demand attention.
