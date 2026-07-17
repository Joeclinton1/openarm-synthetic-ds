#!/usr/bin/env python3
"""Register each rendered OpenArm from its projected base/tool axis to the source arm track.

The benchmark cameras are not calibrated for most sources.  A whole-robot bounding box is not
enough in that setting: it loses handedness, rotates neither arm into the source view, and becomes
unstable when an arm is occluded.  This script instead performs a per-arm 2-D similarity fit from
the rendered base and end-effector projections to arm-specific source-mask anchors.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "outputs/cross_dataset_openarm_benchmark"
SIDES = ("right", "left")
MOLMO_FIXED_CAMERA_RENDER_SCALE = 0.825


def alpha_over(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    bottom_float = bottom.astype(np.float32) / 255.0
    top_float = top.astype(np.float32) / 255.0
    top_alpha = top_float[..., 3:4]
    bottom_alpha = bottom_float[..., 3:4]
    output_alpha = top_alpha + bottom_alpha * (1.0 - top_alpha)
    premultiplied = (
        top_float[..., :3] * top_alpha
        + bottom_float[..., :3] * bottom_alpha * (1.0 - top_alpha)
    )
    rgb = np.divide(
        premultiplied,
        np.maximum(output_alpha, 1e-6),
        out=np.zeros_like(premultiplied),
    )
    return np.clip(np.concatenate([rgb, output_alpha], axis=2) * 255.0, 0, 255).astype(
        np.uint8
    )


def _read_mask(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image > 127


def _edge_component(mask: np.ndarray, spatial_side: str | None) -> np.ndarray | None:
    """Return the border-connected robot component for a spatial image side."""
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    height, width = mask.shape
    candidates: list[tuple[float, int]] = []
    for label in range(1, count):
        x, y, box_width, box_height, area = (int(value) for value in stats[label])
        if area < 64:
            continue
        left = x <= 3
        right = x + box_width >= width - 3
        top = y <= 3
        bottom = y + box_height >= height - 3
        if not (left or right or top or bottom):
            continue
        cx = float(centroids[label, 0])
        if spatial_side == "left" and not (left or (cx < width / 2 and (top or bottom))):
            continue
        if spatial_side == "right" and not (right or (cx >= width / 2 and (top or bottom))):
            continue
        preferred = left if spatial_side == "left" else right if spatial_side == "right" else True
        candidates.append((float(area) * (2.0 if preferred else 1.0), label))
    if not candidates:
        return None
    label = max(candidates)[1]
    return labels == label


def _source_mask(clip: Path, side: str, index: int, active_count: int) -> np.ndarray | None:
    manual = clip / f"masks_manual_{side}" / f"{index:06d}.png"
    if manual.exists():
        mask = _read_mask(manual)
        return mask if int(mask.sum()) >= 64 else None
    # Some source fixtures provide one audited robot mask for the whole demonstration instead of
    # arm-specific masks.  It is unambiguous when only one retargeted arm is active (HIW keys).
    manual_demo = clip / "masks_manual_a" / f"{index:06d}.png"
    if active_count == 1 and manual_demo.exists():
        component = _edge_component(_read_mask(manual_demo), None)
        if _mask_anchors(component) is not None:
            return component
    # AgiBot's accepted fixture replaces the manipulating left arm, which enters from the left
    # image border, while retaining the stationary right robot as source context. Even though only
    # one OpenArm is rendered, an unconstrained component search can jump to that right-side robot.
    spatial_side = (
        side
        if active_count > 1 or clip.parent.name == "agibot_world_alpha"
        else None
    )
    # Prefer tracked model masks for geometry.  ``masks_final`` is deliberately conservative for
    # removal and can merge a black robot with nearby dark furniture or plants after dilation.
    # That union is safe for inpainting but its farthest point is not a reliable tool anchor.
    for directory in ("masks_sam2", "masks_robotseg", "masks_final"):
        path = clip / directory / f"{index:06d}.png"
        if not path.exists():
            continue
        component = _edge_component(_read_mask(path), spatial_side)
        if _mask_anchors(component) is not None:
            return component
    return None


def _source_geometry_mask(
    clip: Path, side: str, index: int, active_count: int
) -> np.ndarray | None:
    """Return the fullest audited side mask for apparent scale and arm-axis fitting."""
    manual = clip / f"masks_manual_{side}" / f"{index:06d}.png"
    if manual.exists():
        mask = _read_mask(manual)
        return mask if int(mask.sum()) >= 64 else None
    manual_demo = clip / "masks_manual_a" / f"{index:06d}.png"
    if active_count == 1 and manual_demo.exists():
        component = _edge_component(_read_mask(manual_demo), None)
        if _mask_anchors(component) is not None:
            return component
    path = clip / "masks_final" / f"{index:06d}.png"
    if path.exists():
        spatial_side = side if active_count > 1 else None
        component = _edge_component(_read_mask(path), spatial_side)
        if _mask_anchors(component) is not None:
            return component
    return _source_mask(clip, side, index, active_count)


def _mask_anchors(
    mask: np.ndarray | None, preferred_border: int | None = None
) -> tuple[np.ndarray, np.ndarray] | None:
    """Estimate where an arm enters the image and its farthest visible tool point."""
    if mask is None or int(mask.sum()) < 64:
        return None
    ys, xs = np.where(mask)
    height, width = mask.shape
    margin = 8
    border_sets = (
        xs <= margin,
        xs >= width - 1 - margin,
        ys <= margin,
        ys >= height - 1 - margin,
    )
    counts = np.asarray([int(value.sum()) for value in border_sets])
    if counts.max(initial=0) < 3:
        return None
    border = int(np.argmax(counts)) if preferred_border is None else preferred_border
    if counts[border] < 3:
        return None
    at_base = border_sets[border]
    base = np.asarray([np.median(xs[at_base]), np.median(ys[at_base])], dtype=np.float64)
    points = np.column_stack([xs, ys]).astype(np.float64)
    distance = np.linalg.norm(points - base, axis=1)
    far = distance >= np.quantile(distance, 0.985)
    tool = np.median(points[far], axis=0)
    # Short edge blobs are usually a detector seeing only the mounting plate.  Scaling an entire
    # rendered arm into such a blob produces the tiny-arm failure seen in AgiBot demo B; treat the
    # arm as occluded and allow the next mask source (or temporal interpolation) to take over.
    if np.linalg.norm(tool - base) < 60:
        return None
    # A farthest point on a second image edge is another truncation boundary, not an observed
    # gripper.  Let the temporally smoothed track bridge these occluded frames.
    if (
        tool[0] <= margin
        or tool[0] >= width - 1 - margin
        or tool[1] <= margin
        or tool[1] >= height - 1 - margin
    ):
        return None
    return base, tool


def _gripper_refined_anchor(
    clip: Path,
    index: int,
    mask: np.ndarray,
    anchor: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Use a nearby gripper mask to refine the noisy farthest-point tool estimate."""
    path = clip / "masks_gripper" / f"{index:06d}.png"
    if not path.exists():
        return anchor
    gripper = _read_mask(path)
    nearby = cv2.dilate(mask.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        gripper.astype(np.uint8), connectivity=8
    )
    base, estimated_tool = anchor
    estimated_length = float(np.linalg.norm(estimated_tool - base))
    candidates = []
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) < 12:
            continue
        component = labels == label
        if not np.any(component & nearby):
            continue
        center = np.asarray(centroids[label], dtype=np.float64)
        distance = float(np.linalg.norm(center - base))
        if distance < 0.55 * estimated_length:
            continue
        candidates.append((distance, center))
    if not candidates:
        return anchor
    return base, max(candidates, key=lambda item: item[0])[1]


def _smooth_anchors(
    anchors: list[tuple[np.ndarray, np.ndarray] | None],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    visible = np.asarray([value is not None for value in anchors], dtype=bool)
    if not np.any(visible):
        raise RuntimeError("No visible source-arm anchors")
    values = np.full((len(anchors), 4), np.nan, dtype=np.float64)
    for index, value in enumerate(anchors):
        if value is not None:
            values[index] = np.concatenate(value)
    frames = np.arange(len(values))
    for column in range(values.shape[1]):
        valid = np.isfinite(values[:, column])
        values[:, column] = np.interp(frames, frames[valid], values[valid, column])
        size = min(7, len(values) if len(values) % 2 else len(values) - 1)
        if size >= 3:
            values[:, column] = median_filter(values[:, column], size=size, mode="nearest")
        values[:, column] = gaussian_filter1d(values[:, column], sigma=1.25, mode="nearest")
    return values[:, :2], values[:, 2:], visible


def _project(point: np.ndarray, camera_from_world: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    camera = camera_from_world @ np.append(point, 1.0)
    if camera[2] <= 1e-6:
        raise RuntimeError("Scene anchor projects behind the preview camera")
    return np.asarray(
        [
            intrinsics[0, 0] * camera[0] / camera[2] + intrinsics[0, 2],
            intrinsics[1, 1] * camera[1] / camera[2] + intrinsics[1, 2],
        ]
    )


def _scene_side_points(
    scene: dict, side: str, frame: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera_frames = scene["camera"]["world_from_camera_frames"]
    world_from_camera = np.asarray(
        camera_frames[frame if len(camera_frames) > 1 else 0], dtype=np.float64
    )
    camera_from_world = np.linalg.inv(world_from_camera)
    intrinsics = np.asarray(scene["camera"]["intrinsics"], dtype=np.float64)
    objects = [item for item in scene["objects"] if f"_{side}_" in item["name"]]
    if not objects:
        raise RuntimeError(f"Scene contains no {side} arm")

    def origin(item: dict) -> np.ndarray:
        transforms = item["world_from_object_frames"]
        return np.asarray(transforms[frame if len(transforms) > 1 else 0], dtype=np.float64)[:3, 3]

    base_item = next(item for item in objects if item["name"] == f"base_link_{side}_00")
    tool_item = next(item for item in objects if item["name"] == f"ee_base_link_{side}_00")
    base = _project(origin(base_item), camera_from_world, intrinsics)
    pinch_frames = scene.get("anchors", {}).get(side, {}).get("pinch_center_world_frames")
    tool_world = (
        np.asarray(pinch_frames[frame], dtype=np.float64)
        if pinch_frames is not None
        else origin(tool_item)
    )
    tool = _project(tool_world, camera_from_world, intrinsics)
    # Each visual link has several geoms at the same kinematic transform.  One representative
    # point per link is sufficient for splitting legacy combined renders and avoids a large
    # per-pixel distance tensor.
    center_objects = [item for item in objects if item["name"].endswith("_00")]
    centers = np.stack(
        [_project(origin(item), camera_from_world, intrinsics) for item in center_objects]
    )
    return base, tool, centers


def _similarity(
    source_base: np.ndarray,
    source_tool: np.ndarray,
    target_base: np.ndarray,
    target_tool: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    source_axis = source_tool - source_base
    target_axis = target_tool - target_base
    source_length = float(np.linalg.norm(source_axis))
    target_length = float(np.linalg.norm(target_axis))
    if min(source_length, target_length) < 1e-6:
        raise RuntimeError("Degenerate base-to-tool projection")
    scale = float(np.clip(target_length / source_length, 0.35, 4.0))
    cosine = float(np.dot(source_axis, target_axis) / (source_length * target_length))
    sine = float(
        (source_axis[0] * target_axis[1] - source_axis[1] * target_axis[0])
        / (source_length * target_length)
    )
    linear = scale * np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    translation = target_base - linear @ source_base
    matrix = np.column_stack([linear, translation]).astype(np.float32)
    angle = float(np.arctan2(sine, cosine))
    return matrix, scale, angle


def _fixed_scale_tool_similarity(
    source_base: np.ndarray,
    source_tool: np.ndarray,
    target_base: np.ndarray,
    target_tool: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, float, float]:
    """Match the tool and arm-axis direction without shrinking to a partially visible source arm."""
    source_axis = source_tool - source_base
    target_axis = target_tool - target_base
    source_length = float(np.linalg.norm(source_axis))
    target_length = float(np.linalg.norm(target_axis))
    cosine = float(np.dot(source_axis, target_axis) / (source_length * target_length))
    sine = float(
        (source_axis[0] * target_axis[1] - source_axis[1] * target_axis[0])
        / (source_length * target_length)
    )
    linear = scale * np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    translation = target_tool - linear @ source_tool
    matrix = np.column_stack([linear, translation]).astype(np.float32)
    return matrix, scale, float(np.arctan2(sine, cosine))


def _edge_tool_similarity(
    rgba: np.ndarray,
    source_base: np.ndarray,
    source_tool: np.ndarray,
    target_entry: np.ndarray,
    target_tool: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Rigidly align a tool and scale the silhouette to the observed source entry border."""
    source_axis = source_tool - source_base
    target_axis = target_tool - target_entry
    if min(np.linalg.norm(source_axis), np.linalg.norm(target_axis)) < 1e-6:
        raise RuntimeError("Degenerate arm-axis projection")
    angle = float(
        np.arctan2(target_axis[1], target_axis[0])
        - np.arctan2(source_axis[1], source_axis[0])
    )
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    ys, xs = np.where(rgba[..., 3] > 2)
    if not len(xs):
        raise RuntimeError("Empty side render alpha")
    relative = (np.column_stack([xs, ys]) - source_tool) @ rotation.T
    height, width = rgba.shape[:2]
    border_distances = np.asarray(
        [target_entry[0], width - 1 - target_entry[0], target_entry[1], height - 1 - target_entry[1]]
    )
    border = int(np.argmin(border_distances))
    if border == 0:
        extent = float(np.quantile(relative[:, 0], 0.002))
        required = float(target_entry[0] - target_tool[0])
    elif border == 1:
        extent = float(np.quantile(relative[:, 0], 0.998))
        required = float(target_entry[0] - target_tool[0])
    elif border == 2:
        extent = float(np.quantile(relative[:, 1], 0.002))
        required = float(target_entry[1] - target_tool[1])
    else:
        extent = float(np.quantile(relative[:, 1], 0.998))
        required = float(target_entry[1] - target_tool[1])
    if abs(extent) < 1e-6 or required * extent <= 0:
        # The chosen view does not extend toward the observed border; fall back to apparent size.
        render_span = max(float(np.ptp(relative[:, 0])), float(np.ptp(relative[:, 1])), 1.0)
        target_span = float(np.linalg.norm(target_axis))
        scale = target_span / render_span
    else:
        scale = required / extent
    # A source robot can be much longer and slimmer than OpenArm.  Matching its full cropped reach
    # would make OpenArm's joints dominate the frame; cap only this uncalibrated review transform.
    scale = float(np.clip(scale, 0.35, 2.2))
    linear = scale * rotation
    translation = target_tool - linear @ source_tool
    return np.column_stack([linear, translation]).astype(np.float32), scale, angle


def _entry_error(mask: np.ndarray, target_entry: np.ndarray) -> float:
    """Distance in pixels between a warped silhouette and its intended image border."""
    ys, xs = np.where(mask)
    if not len(xs):
        return float("inf")
    height, width = mask.shape
    border = int(
        np.argmin(
            [
                target_entry[0],
                width - 1 - target_entry[0],
                target_entry[1],
                height - 1 - target_entry[1],
            ]
        )
    )
    observed = (float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max()))[border]
    return abs(observed - float(target_entry[border // 2]))


def _split_combined_render(
    rgba: np.ndarray, scene: dict, frame: int, active: list[str]
) -> dict[str, np.ndarray]:
    """Recover per-arm RGBA from an old combined render using projected link Voronoi cells."""
    if len(active) == 1:
        return {active[0]: rgba}
    alpha = rgba[..., 3] > 2
    ys, xs = np.where(alpha)
    points = np.column_stack([xs, ys]).astype(np.float64)
    distances = []
    for side in active:
        centers = _scene_side_points(scene, side, frame)[2]
        distance = np.min(np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2), axis=1)
        distances.append(distance)
    owner = np.argmin(np.stack(distances, axis=1), axis=1)
    result = {}
    for side_index, side in enumerate(active):
        keep = np.zeros(alpha.shape, dtype=bool)
        keep[ys[owner == side_index], xs[owner == side_index]] = True
        side_rgba = rgba.copy()
        side_rgba[~keep] = 0
        result[side] = side_rgba
    return result


def _load_side_render(
    clip: Path,
    side: str,
    index: int,
    combined: np.ndarray,
    split: dict[str, np.ndarray],
) -> np.ndarray:
    path = clip / f"render_raw_{side}" / f"{index:06d}.png"
    if path.exists():
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.shape[2] != 4:
            raise RuntimeError(f"Expected RGBA render: {path}")
        return image
    return split[side]


def align_clip(clip: Path) -> dict:
    render_paths = sorted((clip / "render_raw").glob("[0-9]*.png"))
    if not render_paths:
        raise FileNotFoundError(clip / "render_raw")
    scene = json.loads((clip / "render_scene/scene.json").read_text())
    with np.load(clip / "trajectory.npz", allow_pickle=False) as trajectory:
        metadata = json.loads(str(trajectory["metadata_json"].item()))
    active = [side for side in SIDES if side in metadata.get("active_sides", SIDES)]
    if scene.get("benchmark_projection", {}).get("mode") == (
        "accepted AgiBot fixture camera registration"
    ):
        output = clip / "render_aligned"
        output.mkdir(exist_ok=True)
        overlap_intersection = overlap_render = overlap_source = 0
        for index, render_path in enumerate(render_paths):
            rgba = cv2.imread(str(render_path), cv2.IMREAD_UNCHANGED)
            if rgba is None or rgba.shape[2] != 4:
                raise RuntimeError(f"Expected RGBA render: {render_path}")
            if not cv2.imwrite(str(output / f"{index:06d}.png"), rgba):
                raise RuntimeError(f"Could not write camera-registered render for {clip}")
            source_mask = _read_mask(clip / "masks_final" / f"{index:06d}.png")
            render_mask = rgba[..., 3] > 2
            overlap_intersection += int(np.sum(source_mask & render_mask))
            overlap_render += int(render_mask.sum())
            overlap_source += int(source_mask.sum())
        manifest = {
            "method": "accepted AgiBot fixture camera registration",
            "frames": len(render_paths),
            "projection_calibrated": True,
            "active_sides": active,
            "aligned_sides": active,
            "side_metrics": {
                side: {"tool_anchor_rmse_px": 0.0, "source_track": "source camera"}
                for side in active
            },
            "render_alpha_inside_source_mask": overlap_intersection / max(overlap_render, 1),
            "source_mask_covered_by_render": overlap_intersection / max(overlap_source, 1),
            "warning": "Source camera mapping reproduced on complete AgiBot episode 649684.",
        }
        (output / "alignment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest
    source_side_count = len(active)
    agibot_active_arm_mode = clip.parent.name == "agibot_world_alpha"
    molmo_camera_fit = (
        scene.get("benchmark_projection", {}).get("mode")
        == "audited Molmo fixed-camera fit"
    )
    frame_count = len(render_paths)
    tracks = {}
    for side in active:
        masks = [
            _source_geometry_mask(clip, side, index, source_side_count)
            for index in range(frame_count)
        ]
        anchors = []
        for index, mask in enumerate(masks):
            anchor = _mask_anchors(mask)
            if anchor is not None and mask is not None:
                anchor = _gripper_refined_anchor(clip, index, mask, anchor)
            anchors.append(anchor)
        if any(value is not None for value in anchors):
            tracks[side] = _smooth_anchors(anchors)
    if not tracks:
        raise RuntimeError(f"No source arm tracks in {clip}")

    fixed_scales: dict[str, float] = {}
    if agibot_active_arm_mode:
        for side, (target_bases, target_tools, visible) in tracks.items():
            ratios = []
            for index in np.where(visible)[0]:
                source_base, source_tool, _ = _scene_side_points(scene, side, int(index))
                ratios.append(
                    np.linalg.norm(target_tools[index] - target_bases[index])
                    / np.linalg.norm(source_tool - source_base)
                )
            # High-visibility frames expose most of the source arm.  Their scale is stable across
            # the fixed-camera clip; lower quantiles are partial edge fragments and caused the
            # tiny-arm regression.
            fixed_scales[side] = float(np.clip(np.quantile(ratios, 0.9), 0.8, 2.0))

    output = clip / "render_aligned"
    output.mkdir(exist_ok=True)
    side_stats: dict[str, dict[str, list[float] | int]] = {
        side: {
            "scales": [],
            "angles": [],
            "base_errors": [],
            "tool_errors": [],
            "visible": 0,
        }
        for side in tracks
    }
    overlap_intersection = overlap_render = overlap_source = 0
    for index, render_path in enumerate(render_paths):
        combined = cv2.imread(str(render_path), cv2.IMREAD_UNCHANGED)
        if combined is None or combined.shape[2] != 4:
            raise RuntimeError(f"Expected RGBA render: {render_path}")
        needs_split = agibot_active_arm_mode or any(
            not (clip / f"render_raw_{side}" / f"{index:06d}.png").exists()
            for side in active
        )
        split = _split_combined_render(combined, scene, index, active) if needs_split else {}
        aligned_sides = []
        for side in active:
            if side not in tracks:
                continue
            target_bases, target_tools, visible = tracks[side]
            # A short border fragment means the source tool is outside the image, not that the
            # last visible tool location should be held indefinitely.
            if not visible[index]:
                continue
            source_base, source_tool, _ = _scene_side_points(scene, side, index)
            if side in fixed_scales:
                matrix, scale, angle = _fixed_scale_tool_similarity(
                    source_base,
                    source_tool,
                    target_bases[index],
                    target_tools[index],
                    fixed_scales[side],
                )
            else:
                matrix, scale, angle = _similarity(
                    source_base, source_tool, target_bases[index], target_tools[index]
                )
            # A single-active-arm scene is already isolated. Do not accidentally reuse stale
            # per-side renders left by an earlier bimanual export.
            rgba = (
                split[side]
                if agibot_active_arm_mode
                else _load_side_render(clip, side, index, combined, split)
            )
            # For an uncalibrated source, the two observable constraints that must remain exact are
            # where the arm enters the image and where its tool acts.  A silhouette-only scale can
            # look superficially plausible while moving the base to the wrong image edge.
            if not agibot_active_arm_mode:
                if molmo_camera_fit:
                    # The fitted fixed camera already supplies the correct 3-D projection,
                    # including wrist direction. Keep one audited apparent-size correction for
                    # the embodiment, then remove only the residual at the pinch centre.
                    scale = MOLMO_FIXED_CAMERA_RENDER_SCALE
                    delta = target_tools[index] - scale * source_tool
                    matrix = np.asarray(
                        [[scale, 0.0, delta[0]], [0.0, scale, delta[1]]], dtype=np.float32
                    )
                    angle = 0.0
                else:
                    matrix, scale, angle = _edge_tool_similarity(
                        rgba,
                        source_base,
                        source_tool,
                        target_bases[index],
                        target_tools[index],
                    )
            aligned = cv2.warpAffine(
                rgba,
                matrix,
                (640, 480),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0),
            )
            aligned_sides.append(aligned)
            projected_tool = matrix[:, :2] @ source_tool + matrix[:, 2]
            stats = side_stats[side]
            stats["scales"].append(scale)
            stats["angles"].append(angle)
            stats["base_errors"].append(
                _entry_error(aligned[..., 3] > 2, target_bases[index])
                if not agibot_active_arm_mode
                else 0.0
            )
            stats["tool_errors"].append(float(np.linalg.norm(projected_tool - target_tools[index])))
            stats["visible"] = int(stats["visible"]) + int(visible[index])
        result = np.zeros((480, 640, 4), dtype=np.uint8)
        for aligned in aligned_sides:
            result = alpha_over(result, aligned)
        if not cv2.imwrite(str(output / f"{index:06d}.png"), result):
            raise RuntimeError(f"Could not write aligned render for {clip}")
        source_mask = _read_mask(clip / "masks_final" / f"{index:06d}.png")
        render_mask = result[..., 3] > 2
        overlap_intersection += int(np.sum(source_mask & render_mask))
        overlap_render += int(render_mask.sum())
        overlap_source += int(source_mask.sum())

    side_summary = {}
    for side, stats in side_stats.items():
        side_summary[side] = {
            "visible_frames": int(stats["visible"]),
            "mean_scale": float(np.mean(stats["scales"])),
            "mean_rotation_deg": float(np.rad2deg(np.mean(stats["angles"]))),
            "base_anchor_rmse_px": float(np.sqrt(np.mean(np.square(stats["base_errors"])))),
            "tool_anchor_rmse_px": float(np.sqrt(np.mean(np.square(stats["tool_errors"])))),
            "source_track": (
                f"masks_manual_{side}"
                if (clip / f"masks_manual_{side}").is_dir()
                else "masks_manual_a"
                if (clip / "masks_manual_a").is_dir() and len(active) == 1
                else "border-connected SAM2/RobotSeg component with masks_final fallback"
            ),
        }
    scale_method = (
        "fixed-camera pinch-centre registration with fixed embodiment scale"
        if molmo_camera_fit
        else "source-border silhouette scale"
    )
    manifest = {
        "method": (
            scale_method
            if molmo_camera_fit
            else f"rigid per-arm tool registration with {scale_method}"
        ),
        "frames": frame_count,
        "projection_calibrated": False,
        "active_sides": active,
        "aligned_sides": list(tracks),
        "agibot_active_arm_mode": agibot_active_arm_mode,
        "fixed_side_scales": fixed_scales,
        "side_metrics": side_summary,
        "render_alpha_inside_source_mask": overlap_intersection / max(overlap_render, 1),
        "source_mask_covered_by_render": overlap_intersection / max(overlap_source, 1),
        "warning": "2-D source-track registration is for visual review, not metric supervision.",
    }
    (output / "alignment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    for clip in sorted(BENCHMARK.glob("*/*")):
        if not (clip / "render_raw").is_dir():
            continue
        manifest = align_clip(clip)
        print(clip.relative_to(BENCHMARK), json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
