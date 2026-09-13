"""Real-world physical dimensions of a domino tile and playing surface, in meters
(the physics engine's native unit) -- see RESEARCH.md's "Domino object physical
properties" section for the manufacturing research this is based on.
"""

from __future__ import annotations

# Standard double-six tile: ~56x28x10mm, a 2:1 length:width ratio that holds across
# cheap-to-professional grades (also matches tile_detect.py's own aspect-ratio filter).
TILE_LENGTH_M = 0.056
TILE_WIDTH_M = 0.028
TILE_THICKNESS_M = 0.010
TILE_HALF_EXTENTS_M = (TILE_LENGTH_M / 2, TILE_WIDTH_M / 2, TILE_THICKNESS_M / 2)

# A typical melamine tile's mass; not independently measured for this project's set,
# only needed for plausible-enough contact dynamics, not precision physics.
TILE_MASS_KG = 0.007

# calibration.json's rectified_size is a homography output size in pixels, not a real-
# world measurement -- this is a plausible placeholder for a domino table's play area
# until a real physical measurement or camera calibration exists (see RESEARCH.md:
# "calibration.json is a homography, not a camera calibration").
TABLE_SIZE_M = (1.3, 1.3)  # comfortably covers trajectory.py's rack zones (+/-0.35m
# plus drop spread), not just the central trick area -- a real table physically extends
# under the players' racks too, unlike the *rectified calibration crop*, which research
# found does NOT (RESEARCH.md: racks/piles fall outside the real rig's calibrated quad).
# This simulation's camera has no such calibration-region restriction, so it shouldn't
# reproduce that limitation. Sized with margin above the rack zone's worst-case reach from
# origin (see test_tile_geometry.py) rather than the tightest value that happens to fit,
# since the exact meters are an unverified placeholder pending a real table measurement.
