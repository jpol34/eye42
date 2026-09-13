"""Turns physics-simulated tile states into image-space ground truth (projected
footprints, occlusion-correct visibility masks) via a simple pinhole camera model.

This is what actually fixes the bug `synth_data.py`'s own docstring documents (an
occluded tile's stored mask isn't corrected for what a later tile covers) -- here,
correct occlusion follows from real 3D z-order computed from physics, not from
compositing order, regardless of which pixel-rendering backend produces the final RGB
image. Photoreal rendering via BlenderProc/Cycles is a separate, clearly-isolated
function at the bottom of this module, gated behind the optional `sim-render` extra --
see RESEARCH.md's "Standalone 3D simulation" section for its current status.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from eye42.engine.tiles import Tile
from eye42.perception.synth_data import BODY_COLOR, PIP_COLOR, pip_layout_fractions  # reused, not duplicated
from eye42.simgen.physics import TileState
from eye42.simgen.tile_geometry import TABLE_SIZE_M, TILE_HALF_EXTENTS_M

_TOP_TEXTURE_SIZE = (256, 128)  # (w, h) px, matches TILE_LENGTH_M:TILE_WIDTH_M's 2:1 aspect
_DIVIDER_HALF_WIDTH_PX = 3
_PIP_RADIUS_PX = 10


@dataclass(frozen=True)
class Camera:
    """A pinhole camera: intrinsics (a single focal length in pixels + centered
    principal point) and extrinsics (position + look-at target, world meters) --
    a placeholder approximating the real rig's oblique overhead angle, pending a real
    checkerboard calibration (RESEARCH.md: "calibration.json is a homography, not a
    camera calibration"). Self-consistent ground truth for Phase 0; not claimed to
    match the real camera's exact numbers yet."""

    image_size: Tuple[int, int]  # (width, height) px
    focal_px: float
    position_m: Tuple[float, float, float]
    look_at_m: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    up: Tuple[float, float, float] = (0.0, 0.0, 1.0)  # world is Z-up throughout this
    # simulation (MuJoCo gravity, table plane, tile height) -- must match, not the
    # Y-up convention a generic graphics default would suggest, or this basis and
    # render_photoreal's Blender camera (built from the same basis, see below) would
    # silently disagree about "up" for any placement where the two conventions diverge.

    def basis(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(eye, forward, right, true_up), all in world space -- the single source of
        truth both render_ground_truth's projection and render_photoreal's Blender
        camera placement are built from, so the two can never silently disagree about
        which way the camera is oriented (an earlier version had each recompute this
        independently -- Blender's own to_track_quat heuristic vs. this Gram-Schmidt
        basis -- which only coincidentally agreed for one hardcoded camera placement)."""
        return self._basis()

    def _basis(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        eye = np.array(self.position_m, dtype=float)
        forward = np.array(self.look_at_m, dtype=float) - eye
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, np.array(self.up, dtype=float))
        right = right / np.linalg.norm(right)
        true_up = np.cross(right, forward)
        return eye, forward, right, true_up

    def project(self, point_m: Sequence[float]) -> Tuple[float, float]:
        """Projects a 3D world point (meters) to 2D pixel coordinates."""
        eye, forward, right, true_up = self._basis()
        relative = np.array(point_m, dtype=float) - eye
        depth = max(float(np.dot(relative, forward)), 1e-6)  # guard div-by-zero/behind-camera
        x = float(np.dot(relative, right))
        y = float(np.dot(relative, true_up))
        w, h = self.image_size
        return (w / 2 + self.focal_px * x / depth, h / 2 - self.focal_px * y / depth)

    def depth_of(self, point_m: Sequence[float]) -> float:
        eye, forward, _, _ = self._basis()
        return float(np.dot(np.array(point_m, dtype=float) - eye, forward))


def default_camera(image_size: Tuple[int, int] = (640, 640)) -> Camera:
    """A plausible oblique overhead camera, tuned so the table fills most of the frame
    (matching how the real footage looks) -- research found the real rig's near table
    edge projects to about 1.65x the apparent length of the far edge, i.e. markedly
    oblique, not nadir; this approximates that shape without matching its exact numbers."""
    return Camera(image_size=image_size, focal_px=image_size[0] * 1.5, position_m=(0.0, -0.95, 1.05))


def _tile_top_face_corners_m(state: TileState) -> np.ndarray:
    """The 4 corners of a tile's top face, in world space, given its current pose."""
    hx, hy, hz = TILE_HALF_EXTENTS_M
    local_corners = np.array([[-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz]])
    w, x, y, z = state.orientation_quat  # MuJoCo's (w, x, y, z) convention
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return (rotation @ local_corners.T).T + np.array(state.position)


@dataclass(frozen=True)
class TileRenderInfo:
    """One tile's image-space ground truth for one frame."""

    tile: Tile
    polygon_px: Tuple[Tuple[float, float], ...]  # full (unoccluded) projected footprint
    visible_mask: np.ndarray  # bool, (h, w) -- True where this tile is frontmost
    visible_fraction: float  # visible-pixel-count / full-unoccluded-polygon-pixel-count


def render_ground_truth(tile_states: Sequence[TileState], camera: Camera) -> List[TileRenderInfo]:
    """Projects every tile's top face and computes occlusion-correct visibility masks,
    painting back-to-front by camera depth so a nearer tile's mask always wins any pixel
    it shares with a farther one -- a poor-man's z-buffer, but driven by real physics-
    simulated 3D position, not by compositing order."""
    w, h = camera.image_size
    ownership = np.full((h, w), -1, dtype=np.int32)
    full_masks: List[np.ndarray] = []
    polygons: List[Tuple[Tuple[float, float], ...]] = []

    depths = [camera.depth_of(ts.position) for ts in tile_states]
    order_far_to_near = sorted(range(len(tile_states)), key=lambda i: -depths[i])

    for state in tile_states:
        corners_m = _tile_top_face_corners_m(state)
        polygon = tuple(camera.project(c) for c in corners_m)
        polygons.append(polygon)
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 1)
        full_masks.append(mask.astype(bool))

    for i in order_far_to_near:
        ownership[full_masks[i]] = i  # nearer tiles are painted later, overwriting farther ones

    results: List[TileRenderInfo] = []
    for i, state in enumerate(tile_states):
        visible = ownership == i
        full_count = int(full_masks[i].sum())
        fraction = float(visible.sum()) / full_count if full_count > 0 else 0.0
        results.append(
            TileRenderInfo(tile=state.tile, polygon_px=polygons[i], visible_mask=visible, visible_fraction=fraction)
        )
    return results


def render_debug_preview(tile_states: Sequence[TileState], camera: Camera) -> np.ndarray:
    """A flat-shaded top-down preview image (BGR, uint8) for eyeballing a generated
    scene -- NOT the fidelity-bearing renderer this initiative is built around (see
    RESEARCH.md: 3D rendering fidelity matters for these glossy tiles, and this preview
    has none). Real training-data pixels come from BlenderProc/Cycles (below), gated
    behind the `sim-render` extra; this exists so the physics/occlusion pipeline above
    can be visually sanity-checked without it."""
    w, h = camera.image_size
    image = np.full((h, w, 3), 235, dtype=np.uint8)  # off-white table, matches synth_data.py
    infos = render_ground_truth(tile_states, camera)
    depths = {info.tile: camera.depth_of(state.position) for info, state in zip(infos, tile_states)}
    for info in sorted(infos, key=lambda i: -depths[i.tile]):
        pts = np.array(info.polygon_px, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(image, [pts], BODY_COLOR)
    return image


def _tile_top_texture(tile: Tile, size: Tuple[int, int] = _TOP_TEXTURE_SIZE) -> np.ndarray:
    """A tile's top face as a flat BGR uint8 image: body color, a divider line down the
    centerline (perpendicular to the tile's long axis -- the two halves sit side by side
    along x/width here, NOT stacked along y like synth_data.py's draw_tile_crop), and each
    half's pip dots placed via pip_layout_fractions -- the same reusable layout
    synth_data.py's 2D renderer uses, so both stay visually consistent."""
    w, h = size
    image = np.full((h, w, 3), BODY_COLOR, dtype=np.uint8)
    cv2.line(image, (w // 2, 0), (w // 2, h), PIP_COLOR, _DIVIDER_HALF_WIDTH_PX * 2)
    for half_index, count in enumerate((tile.high, tile.low)):
        half_w = w // 2
        x_offset = half_index * half_w
        for u, v in pip_layout_fractions(count):
            cx, cy = x_offset + int(u * half_w), int(v * h)
            cv2.circle(image, (cx, cy), _PIP_RADIUS_PX, PIP_COLOR, -1)
    return image


_SENSOR_NOISE_SIGMA_RANGE = (1.0, 6.0)  # 0-255 scale, unmeasured -- RESEARCH.md's Tier 3
# calls for noise "matched to the real capture chain," which needs a real calibration
# capture this project doesn't have yet (same status as TABLE_SIZE_M's placeholder
# margin). This is a plausible placeholder, not that calibration, applied only so an
# artificially noise-free render isn't itself a synthetic-data tell a model could key on
# (RESEARCH.md: "the idealized clean image problem is a domain-randomization finding, not
# just an aesthetic one"). Randomized per call, not fixed, so a real camera's varying
# noise magnitude is approximated rather than one single arbitrary constant repeated
# identically across every generated frame.


def _add_sensor_noise(image: np.ndarray) -> np.ndarray:
    sigma = np.random.uniform(*_SENSOR_NOISE_SIGMA_RANGE)
    noise = np.random.normal(0.0, sigma, image.shape)
    return np.clip(image.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def render_photoreal(tile_states: Sequence[TileState], camera: Camera, samples: int = 32) -> np.ndarray:
    """Renders tile_states via headless Blender/Cycles (``bpy``) -- the actual
    fidelity-bearing renderer this initiative is built around (RESEARCH.md: ray-traced
    PBR rendering measurably beats flat/2D compositing for glossy, texture-less objects
    like these tiles). Each tile's top face carries a pip/divider texture built by
    _tile_top_texture; measured roughness/gloss calibration against real footage remains
    a later fidelity pass (see ROADMAP.md).

    ``bpy`` is imported lazily so this module (and everything that imports it, like
    ground_truth.py) stays importable without the optional ``sim-render`` extra."""
    import bpy
    import mathutils

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    table_w, table_h = TABLE_SIZE_M
    bpy.ops.mesh.primitive_plane_add(size=1)
    table = bpy.context.object
    table.scale = (table_w / 2, table_h / 2, 1)
    table_mat = bpy.data.materials.new("table")
    table_mat.use_nodes = True
    table_mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.85, 0.85, 0.82, 1.0)
    table.data.materials.append(table_mat)

    tile_mat = bpy.data.materials.new("tile")
    tile_mat.use_nodes = True
    bsdf = tile_mat.node_tree.nodes["Principled BSDF"]
    blue, green, red = BODY_COLOR  # BODY_COLOR is BGR, per this project's cv2 convention
    bsdf.inputs["Base Color"].default_value = (red / 255, green / 255, blue / 255, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.15  # glossy -- real tiles show specular
    # blowout on their corners under normal lighting (confirmed against real footage
    # this session), which a rough/matte material would never reproduce.

    hx, hy, hz = TILE_HALF_EXTENTS_M
    for tile_index, state in enumerate(tile_states):
        bpy.ops.mesh.primitive_cube_add(size=1)
        obj = bpy.context.object
        obj.scale = (hx, hy, hz)
        obj.location = state.position
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = state.orientation_quat
        obj.data.materials.append(tile_mat)  # index 0: sides/bottom

        top_texture_bgr = _tile_top_texture(state.tile)
        tex_h, tex_w = top_texture_bgr.shape[:2]
        rgba = np.ones((tex_h, tex_w, 4), dtype=np.float32)
        rgba[:, :, 0:3] = top_texture_bgr[:, :, ::-1].astype(np.float32) / 255.0  # BGR -> RGB
        rgba = np.flipud(rgba)  # numpy row 0 = image top; Blender image row 0 = bottom
        top_image = bpy.data.images.new(f"tile_top_{tile_index}", width=tex_w, height=tex_h, alpha=True)
        # 'Non-Color' so Cycles reads these values as-is (linear), matching how
        # bsdf.inputs["Base Color"].default_value above is fed raw 0-1 values with no
        # sRGB decode -- otherwise the body (flat default_value) and the top face
        # (an 8-bit image, sRGB by default) would render at visibly different
        # brightness for what's meant to be the identical BODY_COLOR.
        top_image.colorspace_settings.name = "Non-Color"
        top_image.pixels.foreach_set(rgba.ravel())

        top_mat = bpy.data.materials.new(f"tile_top_{tile_index}")
        top_mat.use_nodes = True
        top_bsdf = top_mat.node_tree.nodes["Principled BSDF"]
        top_bsdf.inputs["Roughness"].default_value = 0.15
        tex_node = top_mat.node_tree.nodes.new("ShaderNodeTexImage")
        tex_node.image = top_image
        top_mat.node_tree.links.new(tex_node.outputs["Color"], top_bsdf.inputs["Base Color"])
        obj.data.materials.append(top_mat)  # index 1: top face

        # A freshly-created primitive_cube_add mesh's polygon normals are in mesh-local
        # space, unaffected by obj.location/obj.rotation_quaternion (those are object-
        # level transforms) -- so the polygon whose local normal is +Z is always the top
        # face, regardless of this tile's world-space pose.
        uv_layer = obj.data.uv_layers.active
        for polygon in obj.data.polygons:
            if polygon.normal.z > 0.9:
                polygon.material_index = 1
                # primitive_cube_add's default UVs are a cross-layout unwrap -- each face
                # gets only a QUARTER of the 0..1 image, not its own independent 0..1 quad
                # (confirmed by inspecting a fresh cube's uv_layers directly; an earlier
                # version here assumed a per-face 0..1 quad, which rendered only a corner
                # slice of the pip texture). Remap the top face's own loops directly from
                # local vertex coordinates (-0.5..0.5 on a size=1 cube) so it covers the
                # full 0..1 texture, with local x/y (length/width) mapping to u/v.
                for loop_index in polygon.loop_indices:
                    vertex_index = obj.data.loops[loop_index].vertex_index
                    vx, vy, _ = obj.data.vertices[vertex_index].co
                    uv_layer.data[loop_index].uv = (vx + 0.5, vy + 0.5)

    bpy.ops.object.light_add(type="AREA", location=(0, 0, 1.5))
    bpy.context.object.data.energy = 300

    bpy.ops.object.camera_add(location=camera.position_m)
    cam_obj = bpy.context.object
    # Built from Camera.basis() directly (not Blender's own to_track_quat heuristic) so
    # this can never silently diverge from render_ground_truth's projection -- see
    # Camera.basis()'s docstring. Blender cameras look down their local -Z axis with
    # local +Y up and +X right, which is exactly (right, true_up, -forward) here.
    _, forward, right, true_up = camera.basis()
    rotation_matrix = mathutils.Matrix(
        (
            (right[0], true_up[0], -forward[0]),
            (right[1], true_up[1], -forward[1]),
            (right[2], true_up[2], -forward[2]),
        )
    )
    cam_obj.rotation_mode = "QUATERNION"
    cam_obj.rotation_quaternion = rotation_matrix.to_quaternion()
    scene.camera = cam_obj

    # Match Blender's lens to this Camera's focal_px, in pixels, so the photoreal
    # render's actual field of view agrees with render_ground_truth's pinhole
    # projection -- otherwise YOLO labels computed from the latter wouldn't line up
    # with this image's pixels at all, silently.
    image_w, _ = camera.image_size
    cam_obj.data.sensor_width = 36.0  # mm, Blender's own default
    cam_obj.data.lens = camera.focal_px * cam_obj.data.sensor_width / image_w

    world = bpy.data.worlds.new("world") if not scene.world else scene.world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.4, 0.4, 0.42, 1.0)
    scene.world = world

    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.render.resolution_x, scene.render.resolution_y = camera.image_size
    scene.render.image_settings.file_format = "PNG"

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = f"{tmpdir}/render.png"
        scene.render.filepath = out_path
        bpy.ops.render.render(write_still=True)
        return _add_sensor_noise(cv2.imread(out_path))
