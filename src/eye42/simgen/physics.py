"""Headless MuJoCo physics scene: a table plane plus one free rigid body per tile.

MuJoCo, not PyBullet, is the physics backend here: PyBullet has no prebuilt wheel for
this platform/Python version and would require installing a C++ build toolchain to
compile from source, whereas MuJoCo ships one and its box-box collider (the exact shape
needed for many thin, flat tiles in contact) was rewritten and fixed in 3.12+, making it
the more robust choice for this specific scenario, not just the more available one. See
RESEARCH.md's "Standalone 3D simulation" section.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import mujoco
import numpy as np

from eye42.engine.tiles import Tile
from eye42.simgen.tile_geometry import TABLE_SIZE_M, TILE_HALF_EXTENTS_M, TILE_MASS_KG

_TIMESTEP_S = 1.0 / 500.0
_MAX_SETTLE_STEPS = 6000  # 12s of sim time -- generous, but bounded so an unstable
# configuration (e.g. tiles spawned already interpenetrating) can never hang forever.
_SETTLE_LINEAR_VEL_M_S = 0.01
_SETTLE_ANGULAR_VEL_RAD_S = 0.05
_DROP_START_HEIGHT_M = 0.30
_DROP_LAYER_GAP_M = 0.05  # vertical spacing between tiles' random start heights, so
# tiles dropped in the same batch don't spawn already interpenetrating at t=0.


@dataclass(frozen=True)
class TileState:
    """One tile's current rigid-body transform in the simulation."""

    tile: Tile
    position: Tuple[float, float, float]
    orientation_quat: Tuple[float, float, float, float]  # (w, x, y, z), MuJoCo's convention


def _build_model_xml(num_bodies: int) -> str:
    hx, hy, hz = TILE_HALF_EXTENTS_M
    tw, th = TABLE_SIZE_M
    bodies = "\n".join(
        f'''
    <body name="tile_{i}" pos="0 0 {_DROP_START_HEIGHT_M}">
      <freejoint/>
      <geom name="tile_geom_{i}" type="box" size="{hx} {hy} {hz}" mass="{TILE_MASS_KG}"
            rgba="0 0.59 0 1" friction="0.4 0.005 0.0001"/>
    </body>'''
        for i in range(num_bodies)
    )
    return f'''
<mujoco>
  <option timestep="{_TIMESTEP_S}"/>
  <worldbody>
    <geom name="table" type="plane" size="{tw / 2} {th / 2} 0.05" friction="0.4 0.005 0.0001"/>
    <light pos="0 0 1.5" dir="0 0 -1" diffuse="1 1 1"/>
{bodies}
  </worldbody>
</mujoco>
'''


class TileSimulation:
    """A fixed-size scene of rigid tile bodies, addressed by ``Tile`` identity.

    The set of tiles this simulation manages is fixed at construction (MuJoCo's model
    is static once compiled); ``drop_and_settle`` repositions a subset of them to
    randomized start poses and lets physics run until they come to rest.
    """

    def __init__(self, tiles: Sequence[Tile], seed: int) -> None:
        if len(set(tiles)) != len(tiles):
            raise ValueError("TileSimulation requires distinct tiles (each is one rigid body)")
        self._tiles: List[Tile] = list(tiles)
        self._index: Dict[Tile, int] = {tile: i for i, tile in enumerate(self._tiles)}
        self._rng = random.Random(seed)
        self._model = mujoco.MjModel.from_xml_string(_build_model_xml(len(self._tiles)))
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)

    def _qpos_slice(self, index: int) -> slice:
        # Each free joint contributes 7 qpos entries (xyz + wxyz quat), in body order.
        return slice(index * 7, index * 7 + 7)

    def _qvel_slice(self, index: int) -> slice:
        # Each free joint contributes 6 qvel entries (linear + angular), in body order.
        return slice(index * 6, index * 6 + 6)

    def state_of(self, tile: Tile) -> TileState:
        index = self._index[tile]
        qpos = self._data.qpos[self._qpos_slice(index)]
        return TileState(
            tile=tile,
            position=(float(qpos[0]), float(qpos[1]), float(qpos[2])),
            orientation_quat=(float(qpos[3]), float(qpos[4]), float(qpos[5]), float(qpos[6])),
        )

    def states(self) -> List[TileState]:
        return [self.state_of(tile) for tile in self._tiles]

    def _settle(self, max_steps: int = _MAX_SETTLE_STEPS) -> int:
        """Steps the simulation until every body's linear/angular velocity drops below
        threshold, or max_steps is hit (never hangs on an unstable configuration).
        Returns the number of steps actually taken."""
        for step in range(max_steps):
            mujoco.mj_step(self._model, self._data)
            qvel = self._data.qvel
            settled = True
            for i in range(len(self._tiles)):
                lin, ang = qvel[self._qvel_slice(i)][:3], qvel[self._qvel_slice(i)][3:]
                if np.linalg.norm(lin) > _SETTLE_LINEAR_VEL_M_S or np.linalg.norm(ang) > _SETTLE_ANGULAR_VEL_RAD_S:
                    settled = False
                    break
            if settled:
                return step + 1
        return max_steps

    def drop_and_settle(
        self,
        tiles: Optional[Sequence[Tile]] = None,
        drop_center: Tuple[float, float] = (0.0, 0.0),
        spread: float = 0.15,
    ) -> List[TileState]:
        """Repositions ``tiles`` (default: all tiles this simulation manages) to
        randomized start poses above (drop_center +/- spread), then steps physics until
        they settle. Returns the settled state of the tiles that were dropped."""
        drop_tiles = list(tiles) if tiles is not None else self._tiles
        for layer, tile in enumerate(drop_tiles):
            index = self._index[tile]
            x = drop_center[0] + self._rng.uniform(-spread, spread)
            y = drop_center[1] + self._rng.uniform(-spread, spread)
            z = _DROP_START_HEIGHT_M + layer * _DROP_LAYER_GAP_M
            yaw = self._rng.uniform(0, 2 * np.pi)
            quat = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))  # rotation about z only
            qpos = self._data.qpos[self._qpos_slice(index)]
            qpos[0:3] = (x, y, z)
            qpos[3:7] = quat
            self._data.qvel[self._qvel_slice(index)] = 0.0
        mujoco.mj_forward(self._model, self._data)
        self._settle()
        return [self.state_of(tile) for tile in drop_tiles]

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self._model, self._data)

    def apply_push(self, tile: Tile, force_xy: Tuple[float, float], steps: int = 1) -> None:
        """Applies a constant horizontal force to ``tile`` for ``steps`` simulation
        steps, then clears it. Low-level primitive -- prefer ``slide_toward`` for
        director-driven moves, since a blind constant force has no speed limit and can
        easily accelerate a ~7g tile into other tiles hard enough to destabilize the
        solver (this is not hypothetical -- it's what an earlier, unclamped version of
        this push produced a real MuJoCo QACC-instability warning from)."""
        index = self._index[tile]
        self._data.xfrc_applied[self._model.body(f"tile_{index}").id, 0:2] = force_xy
        self.step(steps)
        self._data.xfrc_applied[self._model.body(f"tile_{index}").id, :] = 0.0

    def linear_speed(self, tile: Tile) -> float:
        """Current planar (xy) speed, m/s -- used by slide_toward's speed cap and
        available for tests/diagnostics."""
        index = self._index[tile]
        return float(np.linalg.norm(self._data.qvel[self._qvel_slice(index)][:2]))

    def slide_toward(
        self,
        tile: Tile,
        target_xy: Tuple[float, float],
        max_force_n: float = 0.08,  # comfortably above the ~0.027N static-friction
        # threshold implied by TABLE_FRICTION/TILE_MASS_KG below -- a force at or under
        # that threshold can't reliably start the tile moving at all, which is what an
        # earlier, too-conservative default actually did (it hit max_steps every time,
        # barely moving, rather than sliding and arriving).
        max_speed_m_s: float = 0.3,
        arrival_radius_m: float = 0.01,
        max_steps: int = 4000,
    ) -> int:
        """Closed-loop push: applies a small force toward target_xy, recomputed every
        step, only while the tile's current speed is under max_speed_m_s -- so the tile
        accelerates gently, coasts, and decelerates under table friction near the
        target, rather than a blind constant force overshooting at unrealistic speed.
        Stops once within arrival_radius_m of the target or max_steps is hit (bounded,
        never hangs on a target the tile can't reach, e.g. blocked by another tile).
        Returns the number of steps actually taken. Does not itself wait for full
        settle afterward -- call settle() once done pushing everything for this event."""
        index = self._index[tile]
        body_id = self._model.body(f"tile_{index}").id
        target = np.array(target_xy)
        for step in range(max_steps):
            position = self._data.qpos[self._qpos_slice(index)][:2]
            direction = target - position
            distance = float(np.linalg.norm(direction))
            if distance < arrival_radius_m:
                break
            force = direction / distance * max_force_n if self.linear_speed(tile) < max_speed_m_s else np.zeros(2)
            self._data.xfrc_applied[body_id, 0:2] = force
            mujoco.mj_step(self._model, self._data)
        self._data.xfrc_applied[body_id, :] = 0.0
        return step + 1

    def settle(self, max_steps: int = _MAX_SETTLE_STEPS) -> int:
        """Public wrapper so callers (e.g. director.py, after a push) can wait for
        motion to stop without re-randomizing positions the way drop_and_settle does."""
        return self._settle(max_steps)
