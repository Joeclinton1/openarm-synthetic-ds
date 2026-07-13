from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from itertools import zip_longest
from pathlib import Path

import cv2
import numpy as np


def _video_info(path: str | Path) -> dict[str, float | int]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    values = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
    if not values.get("streams"):
        raise FileNotFoundError(f"No video stream: {path}")
    stream = values["streams"][0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(Fraction(stream["avg_frame_rate"])),
        "frames": int(stream["nb_frames"]) if stream.get("nb_frames", "N/A") != "N/A" else -1,
    }


def _iter_bgr_frames(path: str | Path, width: int, height: int):
    """Software-FFmpeg decoder used for AV1 sources unsupported by OpenCV builds."""
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None:
        raise RuntimeError("Could not open FFmpeg output pipe")
    frame_bytes = width * height * 3
    try:
        while True:
            buffer = bytearray()
            while len(buffer) < frame_bytes:
                block = process.stdout.read(frame_bytes - len(buffer))
                if not block:
                    break
                buffer.extend(block)
            if not buffer:
                break
            if len(buffer) != frame_bytes:
                raise RuntimeError("FFmpeg returned a partial video frame")
            yield np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 3).copy()
    finally:
        if process.poll() is None:
            process.terminate()
        process.communicate()


def _binary_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    if mask.shape != shape:
        raise ValueError(f"Mask {path.name} has shape {mask.shape}, expected {shape}")
    return mask > 0


def _convex_object_mask(mask: np.ndarray) -> np.ndarray:
    """Fill fragmented/transparent object tracks without retaining tiny remote speckles."""
    binary = mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return mask
    largest = int(np.max(stats[1:, cv2.CC_STAT_AREA]))
    keep = np.zeros_like(binary)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= max(8, round(0.015 * largest)):
            keep[labels == label] = 1
    points = cv2.findNonZero(keep)
    if points is None:
        return mask
    hull = cv2.convexHull(points)
    filled = np.zeros_like(binary)
    cv2.fillConvexPoly(filled, hull, 1)
    return filled.astype(bool)


def _warp_previous_to_current(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    previous_mask: np.ndarray,
    flow_scale: float = 0.25,
) -> np.ndarray:
    """Warp a previous-frame mask into the current frame with dense backward flow."""
    flow = _backward_flow(previous_gray, current_gray, flow_scale)
    return _warp_with_flow(previous_mask.astype(np.uint8), flow, cv2.INTER_NEAREST).astype(bool)


def _backward_flow(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    scale: float = 0.25,
) -> np.ndarray:
    """Compute current-to-previous flow cheaply, then scale it to full resolution."""
    if not 0 < scale <= 1:
        raise ValueError("flow scale must be in (0, 1]")
    if scale < 1:
        size = (
            max(16, round(current_gray.shape[1] * scale)),
            max(16, round(current_gray.shape[0] * scale)),
        )
        current_work = cv2.resize(current_gray, size, interpolation=cv2.INTER_AREA)
        previous_work = cv2.resize(previous_gray, size, interpolation=cv2.INTER_AREA)
    else:
        current_work, previous_work = current_gray, previous_gray
    flow = cv2.calcOpticalFlowFarneback(
        current_work,
        previous_work,
        None,
        0.5,
        3,
        21,
        3,
        7,
        1.5,
        0,
    )
    if scale < 1:
        flow = cv2.resize(
            flow, (current_gray.shape[1], current_gray.shape[0]), interpolation=cv2.INTER_LINEAR
        )
        flow /= scale
    return flow


def _warp_with_flow(value: np.ndarray, flow: np.ndarray, interpolation: int) -> np.ndarray:
    height, width = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    return cv2.remap(
        value,
        grid_x + flow[..., 0],
        grid_y + flow[..., 1],
        interpolation,
        borderMode=cv2.BORDER_REFLECT,
    )


def refine_robot_masks(
    video_path: str | Path,
    mask_dir: str | Path,
    output_dir: str | Path,
    *,
    protected_mask_dir: str | Path | None = None,
    dilation_radius: int = 7,
    closing_radius: int = 3,
    protect_margin: int = 2,
    use_optical_flow: bool = True,
    flow_scale: float = 0.25,
    protect_convex_hull: bool = True,
) -> Path:
    """Make removal masks contact-safe and temporally complete.

    SAM masks are closed, motion-compensated from the preceding and following frames, and
    dilated to cover antialiased robot boundaries. Optional manipulated-object masks are
    subtracted after a small expansion, ensuring that held objects survive both inpainting and
    compositing. Processing is streaming and bounded to three decoded frames.
    """
    if min(dilation_radius, closing_radius, protect_margin) < 0:
        raise ValueError("Mask radii cannot be negative")
    video_path = Path(video_path).resolve()
    mask_dir = Path(mask_dir).resolve()
    protected = Path(protected_mask_dir).resolve() if protected_mask_dir else None
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    info = _video_info(video_path)
    width, height = int(info["width"]), int(info["height"])
    fps = float(info["fps"])
    shape = (height, width)
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * closing_radius + 1, 2 * closing_radius + 1)
    )
    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * dilation_radius + 1, 2 * dilation_radius + 1)
    )
    protect_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * protect_margin + 1, 2 * protect_margin + 1)
    )
    areas: list[float] = []
    protected_areas: list[float] = []
    propagated_previous: list[float] = []
    propagated_next: list[float] = []

    def load(index: int, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raw = _binary_mask(mask_dir / f"{index:06d}.png", shape).astype(np.uint8)
        if closing_radius:
            raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, close_kernel)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), raw

    frame_iterator = iter(_iter_bgr_frames(video_path, width, height))
    first_frame = next(frame_iterator, None)
    if first_frame is None:
        raise RuntimeError("Video decoder returned zero frames")
    index = 0
    previous: tuple[np.ndarray, np.ndarray] | None = None
    current = load(0, first_frame)
    next_frame = next(frame_iterator, None)
    following = load(1, next_frame) if next_frame is not None else None
    while current is not None:
        gray, raw = current
        current_u8 = raw.copy()
        if use_optical_flow and previous is not None:
            previous_warp = _warp_previous_to_current(
                previous[0], gray, previous[1], flow_scale
            ).astype(np.uint8)
            propagated_previous.append(float(np.mean((previous_warp > 0) & (raw == 0))))
            current_u8 |= previous_warp
        if use_optical_flow and following is not None:
            next_warp = _warp_previous_to_current(
                following[0], gray, following[1], flow_scale
            ).astype(np.uint8)
            propagated_next.append(float(np.mean((next_warp > 0) & (raw == 0))))
            current_u8 |= next_warp
        if dilation_radius:
            current_u8 = cv2.dilate(current_u8, dilate_kernel)
        if protected is not None:
            keep_mask = _binary_mask(protected / f"{index:06d}.png", shape)
            if protect_convex_hull:
                keep_mask = _convex_object_mask(keep_mask)
            keep = keep_mask.astype(np.uint8)
            if protect_margin:
                keep = cv2.dilate(keep, protect_kernel)
            current_u8[keep > 0] = 0
            protected_areas.append(float(np.mean(keep > 0)))
        areas.append(float(np.mean(current_u8 > 0)))
        if not cv2.imwrite(str(output_dir / f"{index:06d}.png"), current_u8 * 255):
            raise RuntimeError(f"Could not write refined mask {index:06d}")
        index += 1
        previous, current = current, following
        if current is None:
            following = None
        else:
            next_frame = next(frame_iterator, None)
            following = load(index + 1, next_frame) if next_frame is not None else None
    if (mask_dir / f"{index:06d}.png").exists():
        raise ValueError("Mask directory contains more frames than the video")
    manifest = {
        "method": "closed+dilated+bidirectional-motion-compensated robot masks",
        "source_video": str(video_path),
        "source_masks": str(mask_dir),
        "protected_masks": str(protected) if protected else None,
        "frames": index,
        "fps": fps,
        "resolution": [width, height],
        "dilation_radius": dilation_radius,
        "closing_radius": closing_radius,
        "protect_margin": protect_margin,
        "optical_flow": use_optical_flow,
        "optical_flow_direction": "previous+next" if use_optical_flow else "disabled",
        "flow_scale": flow_scale,
        "protect_convex_hull": protect_convex_hull,
        "mean_removal_fraction": float(np.mean(areas)),
        "max_removal_fraction": float(np.max(areas)),
        "mean_previous_propagated_fraction": (
            float(np.mean(propagated_previous)) if propagated_previous else 0.0
        ),
        "mean_next_propagated_fraction": (
            float(np.mean(propagated_next)) if propagated_next else 0.0
        ),
        "mean_protected_fraction": (float(np.mean(protected_areas)) if protected_areas else 0.0),
    }
    path = output_dir / "refinement_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def validate_robot_masks(
    video_path: str | Path,
    mask_dir: str | Path,
    *,
    source_mask_dir: str | Path | None = None,
    protected_mask_dir: str | Path | None = None,
    maximum_empty_fraction: float = 0.0,
    maximum_area_jump_p95: float = 0.08,
    minimum_temporal_iou_p05: float = 0.35,
    minimum_source_recall: float = 0.98,
    maximum_protected_overlap: float = 0.01,
    flow_scale: float = 0.25,
) -> dict:
    """Audit completeness, temporal consistency, and contact-object preservation of masks."""
    info = _video_info(video_path)
    width, height = int(info["width"]), int(info["height"])
    shape = (height, width)
    root = Path(mask_dir)
    source_root = Path(source_mask_dir) if source_mask_dir else None
    protected_root = Path(protected_mask_dir) if protected_mask_dir else None
    areas: list[float] = []
    temporal_ious: list[float] = []
    source_recall: list[float] = []
    protected_overlap: list[float] = []
    previous_gray: np.ndarray | None = None
    previous_mask: np.ndarray | None = None
    index = 0
    for frame in _iter_bgr_frames(video_path, width, height):
        mask = _binary_mask(root / f"{index:06d}.png", shape)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        areas.append(float(np.mean(mask)))
        keep = None
        if protected_root is not None:
            keep = _binary_mask(protected_root / f"{index:06d}.png", shape)
            if np.any(keep):
                protected_overlap.append(
                    float(np.count_nonzero(mask & keep) / np.count_nonzero(keep))
                )
        if source_root is not None:
            source = _binary_mask(source_root / f"{index:06d}.png", shape)
            if keep is not None:
                source &= ~keep
            if np.any(source):
                source_recall.append(
                    float(np.count_nonzero(mask & source) / np.count_nonzero(source))
                )
        if previous_gray is not None and previous_mask is not None:
            warped = _warp_previous_to_current(previous_gray, gray, previous_mask, flow_scale)
            union = np.count_nonzero(mask | warped)
            temporal_ious.append(float(np.count_nonzero(mask & warped) / union) if union else 1.0)
        previous_gray, previous_mask = gray, mask
        index += 1
    errors: list[str] = []
    if index == 0:
        errors.append("no frames decoded")
    if (root / f"{index:06d}.png").exists():
        errors.append("mask directory contains more frames than the video")
    area_array = np.asarray(areas)
    area_jumps = np.abs(np.diff(area_array))
    metrics = {
        "empty_fraction": float(np.mean(area_array == 0)) if len(area_array) else 1.0,
        "mean_area_fraction": float(np.mean(area_array)) if len(area_array) else 0.0,
        "area_jump_p95": float(np.quantile(area_jumps, 0.95)) if len(area_jumps) else 0.0,
        "temporal_iou_p05": (float(np.quantile(temporal_ious, 0.05)) if temporal_ious else 1.0),
        "mean_source_recall": float(np.mean(source_recall)) if source_recall else 1.0,
        "maximum_protected_overlap": (
            float(np.max(protected_overlap)) if protected_overlap else 0.0
        ),
    }
    limits = {
        "maximum_empty_fraction": maximum_empty_fraction,
        "maximum_area_jump_p95": maximum_area_jump_p95,
        "minimum_temporal_iou_p05": minimum_temporal_iou_p05,
        "minimum_source_recall": minimum_source_recall,
        "maximum_protected_overlap": maximum_protected_overlap,
    }
    comparisons = (
        ("empty_fraction", metrics["empty_fraction"] <= maximum_empty_fraction),
        ("area_jump_p95", metrics["area_jump_p95"] <= maximum_area_jump_p95),
        ("temporal_iou_p05", metrics["temporal_iou_p05"] >= minimum_temporal_iou_p05),
        ("mean_source_recall", metrics["mean_source_recall"] >= minimum_source_recall),
        (
            "maximum_protected_overlap",
            metrics["maximum_protected_overlap"] <= maximum_protected_overlap,
        ),
    )
    for name, accepted in comparisons:
        if not accepted:
            errors.append(f"{name} failed its acceptance threshold")
    return {
        "ok": not errors,
        "frames": index,
        **metrics,
        "limits": limits,
        "source_masks": str(source_root.resolve()) if source_root else None,
        "protected_masks": str(protected_root.resolve()) if protected_root else None,
        "flow_scale": flow_scale,
        "errors": errors,
    }


def stabilize_masks(
    masks: list[np.ndarray], dilation: int = 9, temporal_radius: int = 2
) -> list[np.ndarray]:
    kernel = np.ones((dilation, dilation), np.uint8)
    binary = [cv2.dilate((mask > 0).astype(np.uint8) * 255, kernel) for mask in masks]
    result: list[np.ndarray] = []
    for index in range(len(binary)):
        window = binary[max(0, index - temporal_radius) : index + temporal_radius + 1]
        result.append(np.max(window, axis=0))
    return result


def inpaint_video(
    video_path: str | Path,
    mask_dir: str | Path,
    output: str | Path,
    method: str = "telea",
) -> Path:
    """Bounded-memory deterministic baseline; masks may come from any segmenter."""
    if method not in {"telea", "ns"}:
        raise ValueError("method must be 'telea' or 'ns'")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {path}")
    dilation = 9
    temporal_radius = 2
    kernel = np.ones((dilation, dilation), np.uint8)
    frames: dict[int, np.ndarray] = {}
    masks: dict[int, np.ndarray] = {}
    index = 0
    next_output = 0
    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS

    def emit(frame_index: int, final_index: int) -> None:
        low = max(0, frame_index - temporal_radius)
        high = min(final_index, frame_index + temporal_radius)
        stable = np.maximum.reduce([masks[i] for i in range(low, high + 1)])
        writer.write(cv2.inpaint(frames.pop(frame_index), stable, 5, flag))

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            mask = cv2.imread(str(Path(mask_dir) / f"{index:06d}.png"), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Missing mask frame {index:06d}.png")
            if mask.shape != (height, width):
                raise ValueError(f"Mask {index:06d} resolution does not match the video")
            frames[index] = frame
            masks[index] = cv2.dilate((mask > 0).astype(np.uint8) * 255, kernel)
            while next_output + temporal_radius <= index:
                emit(next_output, index)
                next_output += 1
                stale = next_output - temporal_radius - 1
                masks.pop(stale, None)
            index += 1
        while next_output < index:
            emit(next_output, index - 1)
            next_output += 1
            stale = next_output - temporal_radius - 1
            masks.pop(stale, None)
    finally:
        capture.release()
        writer.release()
    if index == 0:
        path.unlink(missing_ok=True)
        raise RuntimeError("Video decoder returned zero frames")
    return path


def fuse_robot_gripper_masks(
    robot_mask_dir: str | Path,
    gripper_mask_dir: str | Path,
    output_dir: str | Path,
    *,
    proximity_radius: int = 24,
    minimum_component_area: int = 12,
) -> Path:
    """Add gripper-specific detections without accepting unrelated prompt drift.

    A gripper component is retained only when it lies near the primary robot mask. This catches
    small fingers omitted by a robot-category model while rejecting remote objects that a
    gripper prompt occasionally mistakes for an end effector.
    """
    if proximity_radius < 0 or minimum_component_area < 1:
        raise ValueError("Mask fusion radii and areas must be positive")
    robot_root = Path(robot_mask_dir).resolve()
    gripper_root = Path(gripper_mask_dir).resolve()
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    robot_paths = sorted(robot_root.glob("*.png"))
    gripper_paths = sorted(gripper_root.glob("*.png"))
    if not robot_paths or len(robot_paths) != len(gripper_paths):
        raise ValueError("Robot and gripper mask directories must contain equal PNG sequences")
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * proximity_radius + 1, 2 * proximity_radius + 1)
    )
    added_fractions: list[float] = []
    gripper_recall_before: list[float] = []
    rejected_components = 0
    retained_components = 0
    for index, (robot_path, gripper_path) in enumerate(
        zip(robot_paths, gripper_paths, strict=True)
    ):
        robot = cv2.imread(str(robot_path), cv2.IMREAD_GRAYSCALE)
        gripper = cv2.imread(str(gripper_path), cv2.IMREAD_GRAYSCALE)
        if robot is None or gripper is None or robot.shape != gripper.shape:
            raise ValueError(f"Invalid fusion masks at frame {index:06d}")
        primary = robot > 0
        nearby = cv2.dilate(primary.astype(np.uint8), kernel).astype(bool)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (gripper > 0).astype(np.uint8), connectivity=8
        )
        accepted = np.zeros_like(primary)
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            component = labels == label
            if area >= minimum_component_area and np.any(component & nearby):
                accepted |= component
                retained_components += 1
            else:
                rejected_components += 1
        fused = primary | accepted
        added_fractions.append(float(np.mean(fused & ~primary)))
        if np.any(accepted):
            gripper_recall_before.append(
                float(np.count_nonzero(primary & accepted) / np.count_nonzero(accepted))
            )
        if not cv2.imwrite(str(output_root / f"{index:06d}.png"), fused.astype(np.uint8) * 255):
            raise RuntimeError(f"Could not write fused mask {index:06d}")
    manifest = {
        "method": "primary robot mask plus proximity-gated gripper components",
        "robot_masks": str(robot_root),
        "gripper_masks": str(gripper_root),
        "frames": len(robot_paths),
        "proximity_radius": proximity_radius,
        "minimum_component_area": minimum_component_area,
        "retained_gripper_components": retained_components,
        "rejected_gripper_components": rejected_components,
        "mean_added_fraction": float(np.mean(added_fractions)),
        "maximum_added_fraction": float(np.max(added_fractions)),
        "mean_gripper_recall_before_fusion": (
            float(np.mean(gripper_recall_before)) if gripper_recall_before else 1.0
        ),
        "mean_gripper_recall_after_fusion": 1.0,
    }
    path = output_root / "fusion_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def inpaint_static_camera(
    video_path: str | Path,
    mask_dir: str | Path,
    output: str | Path,
    *,
    protected_mask_dir: str | Path | None = None,
    fallback_video: str | Path | None = None,
    fallback_disagreement_threshold: float = 0.08,
    sample_stride: int = 5,
    minimum_clean_observations: int = 3,
    fallback_radius: int = 7,
    feather_radius: int = 1,
) -> Path:
    """Remove masked pixels using a robust clean plate from a fixed camera.

    Only frames where a pixel is outside both the robot and optional manipulated-object masks
    contribute to that pixel's temporal median. Pixels never revealed in the video use a single
    spatial fallback on the clean plate. A neural inpainted video can optionally supply only
    those never-revealed pixels; its temporal median prevents neural flicker from leaking into
    the static plate. Every unmasked output pixel is copied from its source frame before encoding.
    """
    if sample_stride < 1 or minimum_clean_observations < 1:
        raise ValueError("sample_stride and minimum_clean_observations must be positive")
    if fallback_radius < 1 or feather_radius < 0:
        raise ValueError("fallback_radius must be positive and feather_radius non-negative")
    if not 0 <= fallback_disagreement_threshold <= 1:
        raise ValueError("fallback_disagreement_threshold must be in [0, 1]")
    started = time.perf_counter()
    info = _video_info(video_path)
    width, height = int(info["width"]), int(info["height"])
    shape = (height, width)
    mask_root = Path(mask_dir).resolve()
    protected_root = Path(protected_mask_dir).resolve() if protected_mask_dir else None
    sampled_frames: list[np.ndarray] = []
    sampled_valid: list[np.ndarray] = []
    sampled_indices: list[int] = []
    for index, frame in enumerate(_iter_bgr_frames(video_path, width, height)):
        if index % sample_stride:
            continue
        valid = ~_binary_mask(mask_root / f"{index:06d}.png", shape)
        if protected_root is not None:
            valid &= ~_binary_mask(protected_root / f"{index:06d}.png", shape)
        sampled_frames.append(frame)
        sampled_valid.append(valid)
        sampled_indices.append(index)
    if not sampled_frames:
        raise RuntimeError("Video decoder returned zero sampled frames")
    frames = np.stack(sampled_frames)
    valid = np.stack(sampled_valid)
    clean_count = np.sum(valid, axis=0)
    plate = np.zeros((height, width, 3), dtype=np.uint8)
    tile_rows = 32
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for top in range(0, height, tile_rows):
            bottom = min(height, top + tile_rows)
            values = frames[:, top:bottom].astype(np.float32)
            values = np.where(valid[:, top:bottom, :, None], values, np.nan)
            median = np.nanmedian(values, axis=0)
            plate[top:bottom] = np.nan_to_num(median, nan=0.0).astype(np.uint8)
    insufficient = clean_count < minimum_clean_observations
    ever_mask = np.any(~valid, axis=0)
    fallback_used = insufficient.copy()
    if fallback_video is not None:
        fallback_info = _video_info(fallback_video)
        if (int(fallback_info["width"]), int(fallback_info["height"])) != (width, height):
            raise ValueError("Fallback video resolution does not match the source")
        fallback_samples = [
            frame
            for index, frame in enumerate(_iter_bgr_frames(fallback_video, width, height))
            if index % sample_stride == 0
        ]
        if len(fallback_samples) != len(sampled_frames):
            raise ValueError("Fallback video frame count does not match the source")
        fallback_plate = np.median(np.stack(fallback_samples), axis=0).astype(np.uint8)
        disagreement = (
            np.mean(np.abs(plate.astype(np.float32) - fallback_plate.astype(np.float32)), axis=2)
            / 255.0
        )
        fallback_used |= ever_mask & (disagreement > fallback_disagreement_threshold)
        plate[fallback_used] = fallback_plate[fallback_used]
    elif np.any(insufficient):
        plate = cv2.inpaint(
            plate,
            insufficient.astype(np.uint8) * 255,
            fallback_radius,
            cv2.INPAINT_TELEA,
        )

    # Cross-validation is restricted to the trajectory neighbourhood where the plate is used.
    validation_region = cv2.dilate(
        ever_mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
    ).astype(bool)
    validation_mae: list[float] = []
    for frame, is_valid in zip(frames, valid, strict=True):
        region = is_valid & validation_region & ~insufficient
        if np.any(region):
            validation_mae.append(
                float(
                    np.mean(np.abs(frame.astype(np.float32) - plate.astype(np.float32))[region])
                    / 255.0
                )
            )

    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), float(info["fps"]), (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create static-camera output: {output}")
    distance_scale = float(max(feather_radius, 1))
    frame_count = 0
    try:
        for index, frame in enumerate(_iter_bgr_frames(video_path, width, height)):
            mask = _binary_mask(mask_root / f"{index:06d}.png", shape)
            if feather_radius:
                distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
                alpha = np.clip(distance / distance_scale, 0.0, 1.0)[..., None]
            else:
                alpha = mask.astype(np.float32)[..., None]
            result = frame.astype(np.float32) * (1 - alpha) + plate.astype(np.float32) * alpha
            writer.write(np.clip(result, 0, 255).astype(np.uint8))
            frame_count += 1
    finally:
        writer.release()
    if frame_count == 0:
        output.unlink(missing_ok=True)
        raise RuntimeError("Video decoder returned zero frames")
    manifest = {
        "method": "static-camera mask-aware temporal median clean plate",
        "source_video": str(Path(video_path).resolve()),
        "mask_directory": str(mask_root),
        "protected_masks_excluded_from_plate": str(protected_root) if protected_root else None,
        "fallback_video": str(Path(fallback_video).resolve()) if fallback_video else None,
        "fallback_disagreement_threshold": fallback_disagreement_threshold,
        "fallback_used_fraction": float(np.mean(fallback_used)),
        "output": str(output),
        "frames": frame_count,
        "fps": float(info["fps"]),
        "resolution": [width, height],
        "sample_stride": sample_stride,
        "sampled_frames": len(sampled_indices),
        "minimum_clean_observations": minimum_clean_observations,
        "fully_observed_fraction": float(np.mean(~insufficient)),
        "never_observed_fraction": float(np.mean(clean_count == 0)),
        "fallback_radius": fallback_radius,
        "feather_radius": feather_radius,
        "plate_cross_validation_mae_mean": (
            float(np.mean(validation_mae)) if validation_mae else 0.0
        ),
        "plate_cross_validation_mae_p95": (
            float(np.quantile(validation_mae, 0.95)) if validation_mae else 0.0
        ),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    manifest_path = output.with_suffix(".static.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return output


def restore_protected_video(
    source_video: str | Path,
    clean_video: str | Path,
    protected_mask_dir: str | Path,
    output: str | Path,
    *,
    exclude_mask_dir: str | Path | None = None,
    exclude_margin: int = 2,
    feather_radius: int = 2,
    minimum_mask_area_ratio: float = 0.25,
) -> Path:
    """Restore visible manipulated-object pixels after full robot/object removal.

    Inpainting the complete robot/object contact region avoids old-gripper halos. Only pixels
    explicitly tracked as the manipulated object are then copied from the source; occluded object
    pixels remain reconstructed and can be covered by the replacement gripper. A short inward
    feather prevents a hard segmentation seam without expanding into old robot pixels.
    """
    if min(feather_radius, exclude_margin) < 0:
        raise ValueError("feather_radius and exclude_margin cannot be negative")
    if not 0 <= minimum_mask_area_ratio <= 1:
        raise ValueError("minimum_mask_area_ratio must be in [0, 1]")
    source_info = _video_info(source_video)
    clean_info = _video_info(clean_video)
    width, height = int(source_info["width"]), int(source_info["height"])
    if (int(clean_info["width"]), int(clean_info["height"])) != (width, height):
        raise ValueError("Source and clean video resolutions differ")
    fps = float(source_info["fps"])
    mask_root = Path(protected_mask_dir)
    exclude_root = Path(exclude_mask_dir) if exclude_mask_dir else None
    mask_paths = sorted(mask_root.glob("*.png"))
    if not mask_paths:
        raise ValueError("Protected mask directory contains no PNG frames")
    mask_areas = []
    for path in mask_paths:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != (height, width):
            raise ValueError(f"Invalid protected mask: {path}")
        mask_areas.append(float(np.mean(mask > 0)))
    nonempty_areas = [area for area in mask_areas if area > 0]
    median_mask_area = float(np.median(nonempty_areas)) if nonempty_areas else 0.0
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create protected-object output: {output}")
    sentinel = object()
    index = 0
    restored_fraction: list[float] = []
    skipped_low_confidence = 0
    try:
        for source, clean in zip_longest(
            _iter_bgr_frames(source_video, width, height),
            _iter_bgr_frames(clean_video, width, height),
            fillvalue=sentinel,
        ):
            if source is sentinel or clean is sentinel:
                raise ValueError("Source and clean video frame counts differ")
            assert isinstance(source, np.ndarray) and isinstance(clean, np.ndarray)
            mask = _binary_mask(mask_root / f"{index:06d}.png", (height, width))
            if (
                median_mask_area > 0
                and mask_areas[index] / median_mask_area < minimum_mask_area_ratio
            ):
                mask[:] = False
                skipped_low_confidence += 1
            if exclude_root is not None:
                exclude = _binary_mask(exclude_root / f"{index:06d}.png", (height, width)).astype(
                    np.uint8
                )
                if exclude_margin:
                    kernel = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (2 * exclude_margin + 1, 2 * exclude_margin + 1),
                    )
                    exclude = cv2.dilate(exclude, kernel)
                mask &= ~(exclude > 0)
            if feather_radius and np.any(mask):
                distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
                alpha = np.clip(distance / float(feather_radius), 0.0, 1.0)[..., None]
            else:
                alpha = mask.astype(np.float32)[..., None]
            restored = source.astype(np.float32) * alpha + clean.astype(np.float32) * (1 - alpha)
            writer.write(np.clip(restored, 0, 255).astype(np.uint8))
            restored_fraction.append(float(np.mean(alpha)))
            index += 1
    finally:
        writer.release()
    if index == 0:
        output.unlink(missing_ok=True)
        raise RuntimeError("Video decoder returned zero frames")
    if (mask_root / f"{index:06d}.png").exists():
        output.unlink(missing_ok=True)
        raise ValueError("Protected mask directory contains more frames than the videos")
    manifest = {
        "method": "full removal followed by inward-feathered protected-object restoration",
        "source_video": str(Path(source_video).resolve()),
        "clean_video": str(Path(clean_video).resolve()),
        "protected_masks": str(mask_root.resolve()),
        "excluded_masks": str(exclude_root.resolve()) if exclude_root else None,
        "exclude_margin": exclude_margin,
        "output": str(output),
        "frames": index,
        "fps": fps,
        "resolution": [width, height],
        "feather_radius": feather_radius,
        "minimum_mask_area_ratio": minimum_mask_area_ratio,
        "median_nonempty_mask_area_fraction": median_mask_area,
        "skipped_low_confidence_frames": skipped_low_confidence,
        "mean_restored_fraction": float(np.mean(restored_fraction)),
    }
    output.with_suffix(".restore.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output


def inpaint_propainter(
    video_path: str | Path,
    mask_dir: str | Path,
    output: str | Path,
    repository: str | Path,
    *,
    device: int = 0,
    subvideo_length: int = 80,
    fp16: bool = True,
    episode_chunk_frames: int = 250,
    overlap_frames: int = 20,
    workers: int = 1,
    device_count: int = 1,
) -> Path:
    """Run the official ProPainter inference entry point and validate its output.

    ProPainter is kept as a pinned external checkout because it is an application repository,
    not an importable library. Dataset episodes should be processed separately; its
    ``subvideo_length`` option bounds GPU work while retaining episode-level temporal context.
    """
    video_path = Path(video_path).resolve()
    mask_dir = Path(mask_dir).resolve()
    output = Path(output).resolve()
    repository = Path(repository).resolve()
    script = repository / "inference_propainter.py"
    if not script.is_file():
        raise FileNotFoundError(f"Not a ProPainter checkout: {repository}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    input_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    masks = sorted(mask_dir.glob("*.png"))
    if len(masks) != input_frames:
        raise ValueError(f"Expected {input_frames} masks, found {len(masks)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if episode_chunk_frames <= overlap_frames * 2:
        raise ValueError("episode_chunk_frames must be greater than twice overlap_frames")
    if workers < 1 or device_count < 1:
        raise ValueError("workers and device_count must be positive")
    with tempfile.TemporaryDirectory(prefix="openarm-propainter-") as temporary:
        temporary_path = Path(temporary)

        def run_clip(
            clip_video: Path, clip_masks: Path, result_root: Path, assigned_device: int
        ) -> Path:
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(assigned_device)
            environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            environment.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
            command = [
                sys.executable,
                str(script),
                "--video",
                str(clip_video),
                "--mask",
                str(clip_masks),
                "--output",
                str(result_root),
                "--subvideo_length",
                str(subvideo_length),
                "--save_fps",
                str(round(fps)),
            ]
            if fp16:
                command.append("--fp16")
            subprocess.run(command, cwd=repository, env=environment, check=True)
            result = result_root / clip_video.stem / "inpaint_out.mp4"
            if not result.is_file():
                raise RuntimeError("ProPainter completed without producing inpaint_out.mp4")
            return result

        if input_frames <= episode_chunk_frames:
            clean_masks = temporary_path / "masks"
            clean_masks.mkdir()
            for index, source in enumerate(masks):
                shutil.copy2(source, clean_masks / f"{index:06d}.png")
            result = run_clip(video_path, clean_masks, temporary_path / "results", device)
            shutil.copy2(result, output)
            chunk_count = 1
        else:
            jobs = []
            for chunk_index, core_start in enumerate(range(0, input_frames, episode_chunk_frames)):
                core_end = min(input_frames, core_start + episode_chunk_frames)
                padded_start = max(0, core_start - overlap_frames)
                padded_end = min(input_frames, core_end + overlap_frames)
                chunk_root = temporary_path / f"chunk_{chunk_index:04d}"
                chunk_root.mkdir()
                clip_video = chunk_root / f"clip_{chunk_index:04d}.mp4"
                subprocess.run(
                    [
                        "ffmpeg",
                        "-loglevel",
                        "error",
                        "-y",
                        "-i",
                        str(video_path),
                        "-vf",
                        (
                            f"trim=start_frame={padded_start}:end_frame={padded_end},"
                            "setpts=PTS-STARTPTS"
                        ),
                        "-an",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "fast",
                        "-crf",
                        "12",
                        "-pix_fmt",
                        "yuv420p",
                        str(clip_video),
                    ],
                    check=True,
                )
                clip_masks = chunk_root / "masks"
                clip_masks.mkdir()
                for local, source_index in enumerate(range(padded_start, padded_end)):
                    shutil.copy2(masks[source_index], clip_masks / f"{local:06d}.png")
                jobs.append(
                    (
                        chunk_index,
                        core_start,
                        core_end,
                        padded_start,
                        padded_end,
                        clip_video,
                        clip_masks,
                        chunk_root / "results",
                        device + (chunk_index % device_count),
                    )
                )
            with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
                futures = {
                    job[0]: pool.submit(run_clip, job[5], job[6], job[7], job[8]) for job in jobs
                }
                results = {index: future.result() for index, future in futures.items()}
            stitched = temporary_path / "stitched.mp4"
            writer = cv2.VideoWriter(
                str(stitched), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
            if not writer.isOpened():
                raise RuntimeError("Could not create the stitched ProPainter output")
            written = 0
            try:
                for job in jobs:
                    (
                        chunk_index,
                        core_start,
                        core_end,
                        padded_start,
                        padded_end,
                        _clip_video,
                        _clip_masks,
                        _result_root,
                        _assigned_device,
                    ) = job
                    result = results[chunk_index]
                    capture_result = cv2.VideoCapture(str(result))
                    local_start = core_start - padded_start
                    local_end = core_end - padded_start
                    local = 0
                    while True:
                        ok, frame = capture_result.read()
                        if not ok:
                            break
                        if local_start <= local < local_end:
                            writer.write(frame)
                            written += 1
                        local += 1
                    capture_result.release()
                    if local != padded_end - padded_start:
                        raise RuntimeError(
                            f"ProPainter chunk {chunk_index} frame mismatch: "
                            f"{local} != {padded_end - padded_start}"
                        )
            finally:
                writer.release()
            chunk_count = len(jobs)
            if written != input_frames:
                raise RuntimeError(f"Stitched ProPainter frames {written} != {input_frames}")
            shutil.copy2(stitched, output)
    capture = cv2.VideoCapture(str(output))
    output_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if output_frames != input_frames:
        output.unlink(missing_ok=True)
        raise RuntimeError(f"ProPainter frame mismatch: {output_frames} != {input_frames}")
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    manifest = {
        "method": "ProPainter official inference",
        "repository": "https://github.com/sczhou/ProPainter",
        "revision": revision,
        "source_video": str(video_path),
        "mask_directory": str(mask_dir),
        "frames": input_frames,
        "fps": fps,
        "device": device,
        "workers": workers,
        "device_count": device_count,
        "subvideo_length": subvideo_length,
        "episode_chunk_frames": episode_chunk_frames,
        "overlap_frames": overlap_frames,
        "chunks": chunk_count,
        "fp16": fp16,
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output


def validate_inpainting(
    source_video: str | Path,
    inpainted_video: str | Path,
    removal_mask_dir: str | Path,
    *,
    source_robot_mask_dir: str | Path | None = None,
    protected_mask_dir: str | Path | None = None,
    maximum_outside_mae: float = 0.06,
    maximum_boundary_p95: float = 0.22,
    maximum_temporal_mae: float = 0.16,
    maximum_protected_mae: float = 0.08,
    maximum_inside_copy_fraction: float = 0.18,
    maximum_inside_copy_p95: float = 0.30,
    copy_delta_threshold: float = 0.04,
    flow_scale: float = 0.25,
) -> dict:
    """Audit leakage, seams, flicker, held objects, and copied source-robot residuals.

    Errors are normalized to [0, 1]. The temporal metric uses optical-flow compensation so
    camera/object motion is not mistaken for inpainting flicker.
    """
    source_info = _video_info(source_video)
    result_info = _video_info(inpainted_video)
    width, height = int(source_info["width"]), int(source_info["height"])
    shape = (height, width)
    if int(result_info["width"]) != width or int(result_info["height"]) != height:
        raise ValueError("Source and inpainted video resolutions differ")
    mask_root = Path(removal_mask_dir)
    source_robot_root = Path(source_robot_mask_dir) if source_robot_mask_dir else mask_root
    protected_root = Path(protected_mask_dir) if protected_mask_dir else None
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    outside_mae: list[float] = []
    boundary_error: list[float] = []
    temporal_error: list[float] = []
    protected_error: list[float] = []
    inside_copy: list[float] = []
    inside_copy_frames: list[int] = []
    previous_gray: np.ndarray | None = None
    previous_output: np.ndarray | None = None
    previous_mask: np.ndarray | None = None
    index = 0
    errors: list[str] = []
    sentinel = object()
    for source_frame, output_frame in zip_longest(
        _iter_bgr_frames(source_video, width, height),
        _iter_bgr_frames(inpainted_video, width, height),
        fillvalue=sentinel,
    ):
        if source_frame is sentinel or output_frame is sentinel:
            errors.append("source and inpainted frame counts differ")
            break
        assert isinstance(source_frame, np.ndarray) and isinstance(output_frame, np.ndarray)
        mask = _binary_mask(mask_root / f"{index:06d}.png", shape)
        keep = (
            _binary_mask(protected_root / f"{index:06d}.png", shape)
            if protected_root is not None
            else None
        )
        delta = (
            np.mean(
                np.abs(output_frame.astype(np.float32) - source_frame.astype(np.float32)),
                axis=2,
            )
            / 255.0
        )
        outside = ~cv2.dilate(mask.astype(np.uint8), ring_kernel).astype(bool)
        if np.any(outside):
            outside_mae.append(float(np.mean(delta[outside])))
        inner = mask & ~cv2.erode(mask.astype(np.uint8), ring_kernel).astype(bool)
        source_robot = _binary_mask(source_robot_root / f"{index:06d}.png", shape)
        if keep is not None:
            source_robot &= ~keep
        core = cv2.erode(source_robot.astype(np.uint8), ring_kernel).astype(bool)
        if np.any(core):
            inside_copy.append(float(np.mean(delta[core] < copy_delta_threshold)))
            inside_copy_frames.append(index)
        outer = cv2.dilate(mask.astype(np.uint8), ring_kernel).astype(bool) & ~mask
        if np.any(inner) and np.any(outer):
            inner_mean = cv2.blur(output_frame.astype(np.float32), (3, 3))
            local = np.mean(np.abs(output_frame.astype(np.float32) - inner_mean), axis=2) / 255.0
            boundary_error.append(float(abs(np.mean(local[inner]) - np.mean(local[outer]))))
        gray = cv2.cvtColor(source_frame, cv2.COLOR_BGR2GRAY)
        if previous_gray is not None and previous_output is not None and previous_mask is not None:
            flow = _backward_flow(previous_gray, gray, flow_scale)
            warped = _warp_with_flow(previous_output, flow, cv2.INTER_LINEAR)
            region = mask | _warp_with_flow(
                previous_mask.astype(np.uint8), flow, cv2.INTER_NEAREST
            ).astype(bool)
            if np.any(region):
                temporal_error.append(
                    float(
                        np.mean(
                            np.abs(output_frame.astype(np.float32) - warped.astype(np.float32))[
                                region
                            ]
                        )
                        / 255.0
                    )
                )
        if keep is not None:
            if np.any(keep):
                protected_error.append(float(np.mean(delta[keep])))
        previous_gray, previous_output, previous_mask = gray, output_frame, mask
        index += 1
    if index == 0:
        errors.append("no frames decoded")
    metrics = {
        "outside_mae": float(np.mean(outside_mae)) if outside_mae else 0.0,
        "boundary_p95": float(np.quantile(boundary_error, 0.95)) if boundary_error else 0.0,
        "temporal_mae": float(np.mean(temporal_error)) if temporal_error else 0.0,
        "protected_mae": float(np.mean(protected_error)) if protected_error else 0.0,
        "inside_copy_fraction": float(np.mean(inside_copy)) if inside_copy else 0.0,
        "inside_copy_p95": float(np.quantile(inside_copy, 0.95)) if inside_copy else 0.0,
    }
    limits = {
        "outside_mae": maximum_outside_mae,
        "boundary_p95": maximum_boundary_p95,
        "temporal_mae": maximum_temporal_mae,
        "protected_mae": maximum_protected_mae,
        "inside_copy_fraction": maximum_inside_copy_fraction,
        "inside_copy_p95": maximum_inside_copy_p95,
    }
    for name, limit in limits.items():
        if metrics[name] > limit and (name != "protected_mae" or protected_root is not None):
            errors.append(f"{name} {metrics[name]:.4f} exceeds {limit:.4f}")
    return {
        "ok": not errors,
        "frames": index,
        **metrics,
        "limits": limits,
        "protected_masks": str(protected_root.resolve()) if protected_root else None,
        "source_robot_masks": str(source_robot_root.resolve()),
        "flow_scale": flow_scale,
        "copy_delta_threshold": copy_delta_threshold,
        "worst_inside_copy_frames": [
            {"frame": int(frame), "copy_fraction": float(value)}
            for value, frame in sorted(
                zip(inside_copy, inside_copy_frames, strict=True), reverse=True
            )[:10]
        ],
        "errors": errors,
    }


def composite_rgba(
    background: np.ndarray,
    foreground_rgba: np.ndarray,
    source_depth_m: np.ndarray | None = None,
    render_depth_m: np.ndarray | None = None,
    depth_tolerance_m: float = 0.01,
) -> np.ndarray:
    if background.shape[:2] != foreground_rgba.shape[:2]:
        raise ValueError("Background and foreground must have equal resolution")
    if (source_depth_m is None) != (render_depth_m is None):
        raise ValueError("Source and rendered depth must be supplied together")
    alpha = foreground_rgba[..., 3:4].astype(np.float32) / 255
    if source_depth_m is not None and render_depth_m is not None:
        if (
            source_depth_m.shape != background.shape[:2]
            or render_depth_m.shape != background.shape[:2]
        ):
            raise ValueError("Depth maps must match the image resolution")
        known_source = np.isfinite(source_depth_m) & (source_depth_m > 0)
        robot_behind_scene = known_source & (render_depth_m > source_depth_m + depth_tolerance_m)
        alpha = alpha.copy()
        alpha[robot_behind_scene] = 0
    return np.clip(foreground_rgba[..., :3] * alpha + background * (1 - alpha), 0, 255).astype(
        np.uint8
    )


def _load_depth_frame(root: Path, index: int) -> np.ndarray:
    stem = f"{index:06d}"
    npy = root / f"{stem}.npy"
    npz = root / f"{stem}.npz"
    if npy.is_file():
        return np.load(npy).astype(np.float32, copy=False)
    if npz.is_file():
        with np.load(npz) as payload:
            key = "depth_m" if "depth_m" in payload else payload.files[0]
            return payload[key].astype(np.float32, copy=False)
    raise FileNotFoundError(f"Missing depth frame {stem}.npy/.npz in {root}")


def composite_video(
    background_video: str | Path,
    rgba_dir: str | Path,
    output: str | Path,
    protected_mask_dir: str | Path | None = None,
    source_depth_dir: str | Path | None = None,
    render_depth_dir: str | Path | None = None,
    depth_tolerance_m: float = 0.01,
) -> Path:
    """Composite renderer RGBA frames while preserving optional foreground-object masks."""
    capture = cv2.VideoCapture(str(background_video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    index = 0
    try:
        while True:
            ok, background = capture.read()
            if not ok:
                break
            foreground = cv2.imread(str(Path(rgba_dir) / f"{index:06d}.png"), cv2.IMREAD_UNCHANGED)
            if foreground is None or foreground.shape[-1] != 4:
                raise FileNotFoundError(f"Missing RGBA render frame {index:06d}.png")
            source_depth = render_depth = None
            if source_depth_dir is not None or render_depth_dir is not None:
                if source_depth_dir is None or render_depth_dir is None:
                    raise ValueError("Source and rendered depth directories are both required")
                source_depth = _load_depth_frame(Path(source_depth_dir), index)
                render_depth = _load_depth_frame(Path(render_depth_dir), index)
            if protected_mask_dir is not None:
                protected = cv2.imread(
                    str(Path(protected_mask_dir) / f"{index:06d}.png"),
                    cv2.IMREAD_GRAYSCALE,
                )
                if protected is None:
                    raise FileNotFoundError(f"Missing protected mask {index:06d}.png")
                foreground[protected > 0, 3] = 0
            writer.write(
                composite_rgba(
                    background,
                    foreground,
                    source_depth,
                    render_depth,
                    depth_tolerance_m,
                )
            )
            index += 1
    finally:
        capture.release()
        writer.release()
    if index == 0:
        Path(output).unlink(missing_ok=True)
        raise RuntimeError("Video decoder returned zero frames")
    return Path(output)


def validate_composite_video(
    background_video: str | Path,
    composited_video: str | Path,
    rgba_dir: str | Path,
    protected_mask_dir: str | Path | None = None,
    *,
    maximum_outside_mae: float = 0.03,
    maximum_protected_mae: float = 0.04,
    minimum_robot_change: float = 0.015,
) -> dict:
    """Check that compositing changes the robot region and preserves everything else."""
    info = _video_info(background_video)
    result_info = _video_info(composited_video)
    if (info["width"], info["height"]) != (result_info["width"], result_info["height"]):
        raise ValueError("Background and composite resolutions differ")
    width, height = int(info["width"]), int(info["height"])
    errors: list[str] = []
    outside_error: list[float] = []
    protected_error: list[float] = []
    robot_change: list[float] = []
    protected_root = Path(protected_mask_dir) if protected_mask_dir else None
    count = 0
    for index, (background, result) in enumerate(
        zip_longest(
            _iter_bgr_frames(background_video, width, height),
            _iter_bgr_frames(composited_video, width, height),
        )
    ):
        if background is None or result is None:
            errors.append("background and composite frame counts differ")
            break
        rgba = cv2.imread(str(Path(rgba_dir) / f"{index:06d}.png"), cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.shape != (height, width, 4):
            errors.append(f"missing or invalid RGBA frame {index}")
            break
        robot = rgba[..., 3] > 0
        protected = (
            _binary_mask(protected_root / f"{index:06d}.png", (height, width))
            if protected_root
            else np.zeros((height, width), dtype=bool)
        )
        delta = np.abs(background.astype(np.float32) - result.astype(np.float32)) / 255
        outside = ~cv2.dilate(robot.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        if np.any(outside):
            outside_error.append(float(np.mean(delta[outside])))
        if np.any(protected):
            protected_error.append(float(np.mean(delta[protected])))
        visible_robot = robot & ~protected
        if np.any(visible_robot):
            robot_change.append(float(np.mean(delta[visible_robot])))
        count += 1
    metrics = {
        "outside_mae": float(np.mean(outside_error)) if outside_error else 0.0,
        "protected_mae": float(np.mean(protected_error)) if protected_error else 0.0,
        "robot_region_change": float(np.mean(robot_change)) if robot_change else 0.0,
    }
    if metrics["outside_mae"] > maximum_outside_mae:
        errors.append("composite changed background outside the robot")
    if protected_root and metrics["protected_mae"] > maximum_protected_mae:
        errors.append("composite changed protected-object pixels")
    if metrics["robot_region_change"] < minimum_robot_change:
        errors.append("composite did not visibly insert the replacement robot")
    return {
        "ok": not errors,
        "frames": count,
        **metrics,
        "limits": {
            "outside_mae": maximum_outside_mae,
            "protected_mae": maximum_protected_mae,
            "minimum_robot_region_change": minimum_robot_change,
        },
        "errors": errors,
    }


def validate_rgba_render(
    rgba_dir: str | Path,
    expected_frames: int | None = None,
    minimum_coverage: float = 0.002,
    minimum_visible_fraction: float = 0.95,
) -> dict:
    """Reject camera/render combinations where the target robot is mostly off-screen."""
    paths = sorted(Path(rgba_dir).glob("*.png"))
    errors: list[str] = []
    if expected_frames is not None and len(paths) != expected_frames:
        errors.append(f"frame count {len(paths)} does not match {expected_frames}")
    coverage = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 3 or image.shape[2] != 4:
            errors.append(f"invalid RGBA frame: {path.name}")
            continue
        coverage.append(float(np.mean(image[..., 3] > 0)))
    visible_fraction = float(np.mean(np.asarray(coverage) >= minimum_coverage)) if coverage else 0.0
    if visible_fraction < minimum_visible_fraction:
        errors.append(
            f"robot visible in only {visible_fraction:.1%} of frames; "
            f"required {minimum_visible_fraction:.1%}"
        )
    return {
        "ok": not errors,
        "frames": len(paths),
        "mean_alpha_coverage": float(np.mean(coverage)) if coverage else 0.0,
        "visible_fraction": visible_fraction,
        "minimum_coverage": minimum_coverage,
        "minimum_visible_fraction": minimum_visible_fraction,
        "errors": errors,
    }


def validate_depth_render(
    rgba_dir: str | Path,
    depth_dir: str | Path,
    minimum_p05_alpha_coverage: float = 0.85,
    maximum_depth_m: float = 20.0,
) -> dict:
    """Validate that metric depth is finite on the rendered robot and nowhere else."""
    rgba_paths = sorted(Path(rgba_dir).glob("*.png"))
    errors: list[str] = []
    coverage: list[float] = []
    leakage: list[float] = []
    depths: list[np.ndarray] = []
    for index, path in enumerate(rgba_paths):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 3 or image.shape[2] != 4:
            errors.append(f"invalid RGBA frame: {path.name}")
            continue
        try:
            depth = _load_depth_frame(Path(depth_dir), index)
        except FileNotFoundError as error:
            errors.append(str(error))
            continue
        if depth.shape != image.shape[:2]:
            errors.append(f"depth resolution differs at frame {index}")
            continue
        alpha = image[..., 3] > 0
        valid = np.isfinite(depth) & (depth > 0)
        alpha_area = max(1, int(np.count_nonzero(alpha)))
        coverage.append(float(np.count_nonzero(valid & alpha) / alpha_area))
        leakage.append(float(np.count_nonzero(valid & ~alpha) / max(1, np.count_nonzero(valid))))
        if np.any(valid):
            depths.append(depth[valid])
    p05_coverage = float(np.quantile(coverage, 0.05)) if coverage else 0.0
    maximum_leakage = float(np.max(leakage)) if leakage else 0.0
    minimum_depth = float(min(values.min() for values in depths)) if depths else 0.0
    maximum_depth = float(max(values.max() for values in depths)) if depths else 0.0
    if not rgba_paths:
        errors.append("no RGBA frames")
    if p05_coverage < minimum_p05_alpha_coverage:
        errors.append(
            f"p05 alpha depth coverage {p05_coverage:.4f} is below {minimum_p05_alpha_coverage:.4f}"
        )
    if maximum_leakage > 0:
        errors.append(f"depth leaks outside alpha by {maximum_leakage:.4f}")
    if maximum_depth > maximum_depth_m:
        errors.append(f"maximum depth {maximum_depth:.3f}m exceeds {maximum_depth_m:.3f}m")
    return {
        "ok": not errors,
        "frames": len(coverage),
        "p05_alpha_depth_coverage": p05_coverage,
        "mean_alpha_depth_coverage": float(np.mean(coverage)) if coverage else 0.0,
        "maximum_depth_leakage": maximum_leakage,
        "minimum_depth_m": minimum_depth,
        "maximum_depth_m": maximum_depth,
        "required_p05_alpha_depth_coverage": minimum_p05_alpha_coverage,
        "errors": errors,
    }


def validate_render_alignment(
    reference_rgba_dir: str | Path,
    candidate_rgba_dir: str | Path,
    minimum_mean_iou: float = 0.9,
) -> dict:
    """Compare alpha silhouettes from two render engines using the same camera and joints."""
    reference = sorted(Path(reference_rgba_dir).glob("*.png"))
    candidate = sorted(Path(candidate_rgba_dir).glob("*.png"))
    errors: list[str] = []
    if len(reference) != len(candidate):
        errors.append(f"frame counts differ: {len(reference)} != {len(candidate)}")
    ious = []
    for first, second in zip(reference, candidate, strict=False):
        first_image = cv2.imread(str(first), cv2.IMREAD_UNCHANGED)
        second_image = cv2.imread(str(second), cv2.IMREAD_UNCHANGED)
        if (
            first_image is None
            or second_image is None
            or first_image.shape != second_image.shape
            or first_image.ndim != 3
            or first_image.shape[2] != 4
        ):
            errors.append(f"invalid or mismatched RGBA frames: {first.name}, {second.name}")
            continue
        first_alpha = first_image[..., 3] > 0
        second_alpha = second_image[..., 3] > 0
        union = np.count_nonzero(first_alpha | second_alpha)
        ious.append(float(np.count_nonzero(first_alpha & second_alpha) / union) if union else 1.0)
    mean_iou = float(np.mean(ious)) if ious else 0.0
    if mean_iou < minimum_mean_iou:
        errors.append(f"mean silhouette IoU {mean_iou:.4f} is below {minimum_mean_iou:.4f}")
    return {
        "ok": not errors,
        "frames": len(ious),
        "mean_silhouette_iou": mean_iou,
        "minimum_silhouette_iou": float(np.min(ious)) if ious else 0.0,
        "p05_silhouette_iou": float(np.quantile(ious, 0.05)) if ious else 0.0,
        "required_mean_iou": minimum_mean_iou,
        "errors": errors,
    }


def validate_embodiment_alignment(
    source_robot_masks: str | Path,
    rendered_rgba_dir: str | Path,
    minimum_mean_containment: float = 0.6,
    minimum_p05_containment: float = 0.35,
) -> dict:
    """Require the replacement silhouette to remain in the source robot's image region."""
    masks = sorted(Path(source_robot_masks).glob("*.png"))
    renders = sorted(Path(rendered_rgba_dir).glob("*.png"))
    errors: list[str] = []
    if len(masks) != len(renders):
        errors.append(f"frame counts differ: {len(masks)} != {len(renders)}")
    containment = []
    ious = []
    for mask_path, render_path in zip(masks, renders, strict=False):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        render = cv2.imread(str(render_path), cv2.IMREAD_UNCHANGED)
        if (
            mask is None
            or render is None
            or render.ndim != 3
            or render.shape[2] != 4
            or mask.shape != render.shape[:2]
        ):
            errors.append(f"invalid or mismatched frames: {mask_path.name}, {render_path.name}")
            continue
        source = mask > 0
        replacement = render[..., 3] > 0
        intersection = np.count_nonzero(source & replacement)
        replacement_area = np.count_nonzero(replacement)
        union = np.count_nonzero(source | replacement)
        containment.append(float(intersection / replacement_area) if replacement_area else 0.0)
        ious.append(float(intersection / union) if union else 1.0)
    mean_containment = float(np.mean(containment)) if containment else 0.0
    p05_containment = float(np.quantile(containment, 0.05)) if containment else 0.0
    if mean_containment < minimum_mean_containment:
        errors.append(
            f"mean replacement containment {mean_containment:.4f} is below "
            f"{minimum_mean_containment:.4f}"
        )
    if p05_containment < minimum_p05_containment:
        errors.append(
            f"p05 replacement containment {p05_containment:.4f} is below "
            f"{minimum_p05_containment:.4f}"
        )
    return {
        "ok": not errors,
        "frames": len(containment),
        "mean_replacement_containment": mean_containment,
        "p05_replacement_containment": p05_containment,
        "mean_mask_iou": float(np.mean(ious)) if ious else 0.0,
        "required_mean_containment": minimum_mean_containment,
        "required_p05_containment": minimum_p05_containment,
        "errors": errors,
    }


def distort_rgba_frames(
    rgba_dir: str | Path,
    camera_json: str | Path,
    output_dir: str | Path,
) -> Path:
    """Map ideal pinhole RGBA renders into a source camera's plumb-bob image coordinates."""
    camera_payload = json.loads(Path(camera_json).read_text())
    # Blender scene manifests embed the complete source-camera payload. Accept
    # either that reproducible manifest or a standalone camera JSON.
    camera = camera_payload.get("camera", camera_payload)
    intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64)
    distortion = np.asarray(camera.get("distortion", []), dtype=np.float64)
    if intrinsics.shape != (3, 3) or distortion.shape not in ((0,), (4,), (5,), (8,)):
        raise ValueError("Expected a 3x3 intrinsic matrix and OpenCV distortion coefficients")
    paths = sorted(Path(rgba_dir).glob("*.png"))
    if not paths:
        raise ValueError("No RGBA frames to distort")
    first = cv2.imread(str(paths[0]), cv2.IMREAD_UNCHANGED)
    if first is None or first.ndim != 3 or first.shape[2] != 4:
        raise ValueError("Input frames must be RGBA PNGs")
    height, width = first.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    distorted_pixels = np.stack([grid_x, grid_y], axis=-1).astype(np.float32).reshape(-1, 1, 2)
    if distortion.size:
        undistorted_pixels = cv2.undistortPoints(
            distorted_pixels, intrinsics, distortion, P=intrinsics
        ).reshape(height, width, 2)
    else:
        undistorted_pixels = distorted_pixels.reshape(height, width, 2)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.shape != first.shape:
            raise ValueError(f"Invalid or mismatched RGBA frame: {path}")
        alpha = image[..., 3:4].astype(np.float32) / 255.0
        premultiplied = image.astype(np.float32)
        premultiplied[..., :3] *= alpha
        mapped = cv2.remap(
            premultiplied,
            undistorted_pixels[..., 0],
            undistorted_pixels[..., 1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        mapped_alpha = mapped[..., 3:4] / 255.0
        result = mapped
        result[..., :3] = np.divide(
            mapped[..., :3],
            mapped_alpha,
            out=np.zeros_like(mapped[..., :3]),
            where=mapped_alpha > 1e-6,
        )
        result = np.clip(result, 0, 255).astype(np.uint8)
        cv2.imwrite(str(destination / path.name), result)
    manifest = {
        "method": "OpenCV inverse plumb-bob mapping",
        "camera": str(Path(camera_json).resolve()),
        "input": str(Path(rgba_dir).resolve()),
        "frames": len(paths),
        "intrinsics": intrinsics.tolist(),
        "distortion": distortion.tolist(),
    }
    (destination / "distortion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return destination


def distort_depth_frames(
    depth_dir: str | Path,
    camera_json: str | Path,
    output_dir: str | Path,
) -> Path:
    """Apply the same plumb-bob mapping to metric depth, preserving invalid pixels."""
    camera = json.loads(Path(camera_json).read_text())
    intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64)
    distortion = np.asarray(camera.get("distortion", []), dtype=np.float64)
    inputs = sorted(Path(depth_dir).glob("*.npz"))
    if not inputs:
        inputs = sorted(Path(depth_dir).glob("*.npy"))
    if not inputs:
        raise ValueError("No depth frames to distort")
    first = _load_depth_frame(Path(depth_dir), 0)
    height, width = first.shape
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    distorted_pixels = np.stack([grid_x, grid_y], axis=-1).astype(np.float32).reshape(-1, 1, 2)
    if distortion.size:
        undistorted = cv2.undistortPoints(
            distorted_pixels, intrinsics, distortion, P=intrinsics
        ).reshape(height, width, 2)
    else:
        undistorted = distorted_pixels.reshape(height, width, 2)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for index in range(len(inputs)):
        depth = _load_depth_frame(Path(depth_dir), index)
        if depth.shape != first.shape:
            raise ValueError(f"Depth resolution differs at frame {index}")
        valid = np.isfinite(depth) & (depth > 0)
        numerator = cv2.remap(
            np.where(valid, depth, 0).astype(np.float32),
            undistorted[..., 0],
            undistorted[..., 1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        weight = cv2.remap(
            valid.astype(np.float32),
            undistorted[..., 0],
            undistorted[..., 1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        result = np.full(first.shape, np.inf, dtype=np.float32)
        np.divide(numerator, weight, out=result, where=weight > 1e-3)
        np.savez_compressed(destination / f"{index:06d}.npz", depth_m=result)
    manifest = {
        "schema": "openarm-distorted-depth-v1",
        "method": "validity-normalized OpenCV inverse plumb-bob mapping",
        "camera": str(Path(camera_json).resolve()),
        "input": str(Path(depth_dir).resolve()),
        "frames": len(inputs),
        "intrinsics": intrinsics.tolist(),
        "distortion": distortion.tolist(),
    }
    (destination / "distortion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return destination


def harmonize_rgba_frames(
    background_video: str | Path,
    rgba_dir: str | Path,
    output_dir: str | Path,
    *,
    strength: float = 0.65,
    context_radius: int = 18,
    temporal_smoothing: float = 0.9,
) -> Path:
    """Match robot tone to its local scene context without changing geometry or alpha.

    This is the deterministic, bounded-memory appearance baseline. It uses each frame's
    surrounding pixels as in-context illumination evidence and temporally smooths the robust
    LAB transfer. It deliberately cannot invent geometry or alter the segmentation boundary.
    """
    if not 0 <= strength <= 1 or not 0 <= temporal_smoothing < 1:
        raise ValueError("strength must be in [0,1] and temporal_smoothing in [0,1)")
    if context_radius < 1:
        raise ValueError("context_radius must be positive")
    info = _video_info(background_video)
    width, height = int(info["width"]), int(info["height"])
    paths = sorted(Path(rgba_dir).glob("*.png"))
    if not paths:
        raise ValueError("No RGBA frames to harmonize")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * context_radius + 1, 2 * context_radius + 1)
    )
    smooth_center: np.ndarray | None = None
    smooth_scale: np.ndarray | None = None
    processed = 0
    for index, background in enumerate(_iter_bgr_frames(background_video, width, height)):
        if index >= len(paths):
            break
        foreground = cv2.imread(str(paths[index]), cv2.IMREAD_UNCHANGED)
        if foreground is None or foreground.shape != (height, width, 4):
            raise ValueError(f"Invalid RGBA frame: {paths[index]}")
        alpha = foreground[..., 3]
        mask = alpha >= 8
        ring = (cv2.dilate(mask.astype(np.uint8), kernel) > 0) & ~mask
        result = foreground.copy()
        if np.count_nonzero(mask) >= 32 and np.count_nonzero(ring) >= 32:
            robot_lab = cv2.cvtColor(foreground[..., :3], cv2.COLOR_BGR2LAB).astype(np.float32)
            scene_lab = cv2.cvtColor(background, cv2.COLOR_BGR2LAB).astype(np.float32)
            source_values = robot_lab[mask]
            target_values = scene_lab[ring]
            source_center = np.median(source_values, axis=0)
            target_center = np.median(target_values, axis=0)
            source_spread = np.percentile(source_values, 75, axis=0) - np.percentile(
                source_values, 25, axis=0
            )
            target_spread = np.percentile(target_values, 75, axis=0) - np.percentile(
                target_values, 25, axis=0
            )
            scale = np.clip(target_spread / np.maximum(source_spread, 4), 0.75, 1.25)
            # Local surfaces are illumination evidence, not a request to recolor the robot as
            # the table. Cap LAB shifts to retain OpenArm identity and material readability.
            shift = np.clip(target_center - source_center, [-24, -10, -10], [24, 10, 10])
            target_center = source_center + shift
            if smooth_center is None:
                smooth_center, smooth_scale = target_center, scale
            else:
                smooth_center = (
                    temporal_smoothing * smooth_center + (1 - temporal_smoothing) * target_center
                )
                smooth_scale = temporal_smoothing * smooth_scale + (1 - temporal_smoothing) * scale
            mapped = (robot_lab - source_center) * smooth_scale + smooth_center
            mapped = np.clip(mapped, 0, 255).astype(np.uint8)
            mapped_bgr = cv2.cvtColor(mapped, cv2.COLOR_LAB2BGR)
            blend = strength * (alpha.astype(np.float32) / 255.0)[..., None]
            styled = foreground[..., :3] * (1 - blend) + mapped_bgr * blend
            result[..., :3] = np.clip(styled, 0, 255).astype(np.uint8)
        if not cv2.imwrite(
            str(destination / f"{index:06d}.png"),
            result,
            [cv2.IMWRITE_PNG_COMPRESSION, 3],
        ):
            raise RuntimeError(f"Could not write harmonized frame {index}")
        processed += 1
    if processed != len(paths):
        raise RuntimeError(f"Background supplied {processed} frames for {len(paths)} RGBA frames")
    manifest = {
        "schema": "openarm-photometric-harmonization-v1",
        "method": "temporally smoothed robust local LAB transfer",
        "background": str(Path(background_video).resolve()),
        "input_rgba": str(Path(rgba_dir).resolve()),
        "frames": processed,
        "strength": strength,
        "context_radius": context_radius,
        "temporal_smoothing": temporal_smoothing,
        "geometry_modified": False,
        "alpha_modified": False,
        "generative": False,
    }
    (destination / "harmonization_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return destination


def validate_harmonized_rgba(
    reference_rgba_dir: str | Path,
    candidate_rgba_dir: str | Path,
    maximum_alpha_mismatch_fraction: float = 0.0,
) -> dict:
    """Reject an appearance pass if it changes the accepted robot silhouette."""
    reference = sorted(Path(reference_rgba_dir).glob("*.png"))
    candidate = sorted(Path(candidate_rgba_dir).glob("*.png"))
    errors: list[str] = []
    if len(reference) != len(candidate):
        errors.append(f"frame counts differ: {len(reference)} != {len(candidate)}")
    mismatches = []
    color_change = []
    for first, second in zip(reference, candidate, strict=False):
        a = cv2.imread(str(first), cv2.IMREAD_UNCHANGED)
        b = cv2.imread(str(second), cv2.IMREAD_UNCHANGED)
        if a is None or b is None or a.shape != b.shape or a.shape[-1] != 4:
            errors.append(f"invalid or mismatched RGBA frames: {first.name}, {second.name}")
            continue
        mask = a[..., 3] > 0
        mismatches.append(float(np.mean(a[..., 3] != b[..., 3])))
        if np.any(mask):
            color_change.append(float(np.mean(np.abs(a[..., :3].astype(float) - b[..., :3])[mask])))
    maximum_mismatch = float(np.max(mismatches)) if mismatches else 1.0
    if maximum_mismatch > maximum_alpha_mismatch_fraction:
        errors.append(
            f"maximum alpha mismatch {maximum_mismatch:.8f} exceeds "
            f"{maximum_alpha_mismatch_fraction:.8f}"
        )
    return {
        "ok": not errors,
        "frames": len(mismatches),
        "maximum_alpha_mismatch_fraction": maximum_mismatch,
        "mean_robot_color_change_255": float(np.mean(color_change)) if color_change else 0.0,
        "errors": errors,
    }


def apply_mask_constrained_style(
    reference_video: str | Path,
    candidate_video: str | Path,
    reference_rgba_dir: str | Path,
    output: str | Path,
    *,
    start_frame: int = 0,
    protected_mask_dir: str | Path | None = None,
    strength: float = 0.75,
    maximum_channel_delta: int = 72,
) -> dict:
    """Restrict a generative candidate to photometric changes inside accepted robot alpha.

    The deterministic composite remains authoritative everywhere outside the renderer alpha and
    on protected object pixels. The channel-delta bound prevents a video editor from turning a
    correctly rendered link into an unrelated texture. This converts an unconstrained video
    candidate into a geometry-preserving appearance candidate by construction.
    """
    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")
    if not 0 <= strength <= 1:
        raise ValueError("strength must be in [0,1]")
    if not 0 <= maximum_channel_delta <= 255:
        raise ValueError("maximum_channel_delta must be in [0,255]")
    reference_info = _video_info(reference_video)
    candidate_info = _video_info(candidate_video)
    if (reference_info["width"], reference_info["height"]) != (
        candidate_info["width"],
        candidate_info["height"],
    ):
        raise ValueError("Reference and candidate resolutions differ")
    width, height = int(reference_info["width"]), int(reference_info["height"])
    candidate_count = int(candidate_info["frames"])
    if candidate_count < 1:
        raise ValueError("Candidate video must report a positive frame count")
    rgba_root = Path(reference_rgba_dir)
    protected_root = Path(protected_mask_dir) if protected_mask_dir else None
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(reference_info["fps"]),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {destination}")
    reference_frames = _iter_bgr_frames(reference_video, width, height)
    for _ in range(start_frame):
        if next(reference_frames, None) is None:
            writer.release()
            raise ValueError("start_frame is beyond the reference video")
    changes: list[float] = []
    clipped: list[float] = []
    active_fractions: list[float] = []
    processed = 0
    try:
        for local_index, (reference, candidate) in enumerate(
            zip(reference_frames, _iter_bgr_frames(candidate_video, width, height), strict=False)
        ):
            if local_index >= candidate_count:
                break
            global_index = start_frame + local_index
            rgba = cv2.imread(str(rgba_root / f"{global_index:06d}.png"), cv2.IMREAD_UNCHANGED)
            if rgba is None or rgba.shape != (height, width, 4):
                raise ValueError(f"Missing or invalid RGBA authority frame {global_index:06d}")
            weight = rgba[..., 3].astype(np.float32) / 255
            if protected_root is not None:
                protected = _binary_mask(
                    protected_root / f"{global_index:06d}.png", (height, width)
                )
                weight[protected] = 0
            active = weight > 0
            delta = candidate.astype(np.int16) - reference.astype(np.int16)
            limited = np.clip(delta, -maximum_channel_delta, maximum_channel_delta)
            blend = (weight * strength)[..., None]
            result = np.clip(reference + limited * blend, 0, 255).astype(np.uint8)
            writer.write(result)
            if np.any(active):
                changes.append(float(np.mean(np.abs(result.astype(float) - reference)[active])))
                clipped.append(float(np.mean(np.abs(delta[active]) > maximum_channel_delta)))
            active_fractions.append(float(np.mean(active)))
            processed += 1
    finally:
        writer.release()
    if processed != candidate_count:
        raise RuntimeError(f"Processed {processed} frames for a {candidate_count}-frame candidate")
    manifest = {
        "schema": "openarm-mask-constrained-style-v1",
        "reference_video": str(Path(reference_video).resolve()),
        "candidate_video": str(Path(candidate_video).resolve()),
        "rgba_geometry_authority": str(rgba_root.resolve()),
        "protected_masks": str(protected_root.resolve()) if protected_root else None,
        "output_video": str(destination.resolve()),
        "start_frame": start_frame,
        "frames": processed,
        "strength": strength,
        "maximum_channel_delta": maximum_channel_delta,
        "geometry_modified": False,
        "background_modified_before_encoding": False,
        "protected_pixels_modified_before_encoding": False,
        "mean_robot_color_change_255": float(np.mean(changes)) if changes else 0.0,
        "mean_candidate_channel_clip_fraction": float(np.mean(clipped)) if clipped else 0.0,
        "mean_active_frame_fraction": float(np.mean(active_fractions)),
        "release_accepted": False,
        "acceptance_requirement": (
            "validate output plus independent robot segmentation; otherwise use reference"
        ),
    }
    manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def apply_mask_constrained_style_batch(
    batch_manifest: str | Path,
    output: str | Path,
    *,
    strength: float = 0.75,
    maximum_channel_delta: int = 72,
) -> dict:
    """Center-stitch and hard-constrain every candidate in a VACE batch manifest."""
    manifest_path = Path(batch_manifest)
    batch = json.loads(manifest_path.read_text())
    if batch.get("schema") != "openarm-vace-style-batch-v1":
        raise ValueError("Expected an openarm-vace-style-batch-v1 manifest")
    if not 0 <= strength <= 1 or not 0 <= maximum_channel_delta <= 255:
        raise ValueError("Invalid style strength or channel-delta limit")
    reference_path = Path(batch["input_video"])
    rgba_root = Path(batch["rgba_geometry_authority"])
    protected_root = Path(batch["protected_masks"]) if batch.get("protected_masks") else None
    info = _video_info(reference_path)
    width, height = int(info["width"]), int(info["height"])
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(info["fps"]),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {destination}")
    references = _iter_bgr_frames(reference_path, width, height)
    changes: list[float] = []
    clipped: list[float] = []
    global_index = 0
    try:
        for job in sorted(batch["jobs"], key=lambda item: item["keep_start"]):
            if int(job["keep_start"]) != global_index:
                raise ValueError("Batch keep ranges are not contiguous")
            capture = cv2.VideoCapture(str(job["candidate"]))
            if not capture.isOpened():
                raise FileNotFoundError(job["candidate"])
            local_start = int(job["keep_start"]) - int(job["start"])
            keep_count = int(job["keep_end"]) - int(job["keep_start"])
            capture.set(cv2.CAP_PROP_POS_FRAMES, local_start)
            for _ in range(keep_count):
                ok, candidate = capture.read()
                reference = next(references, None)
                if not ok or reference is None or candidate.shape != (height, width, 3):
                    capture.release()
                    raise RuntimeError(f"Could not read retained frame {global_index}")
                rgba = cv2.imread(str(rgba_root / f"{global_index:06d}.png"), cv2.IMREAD_UNCHANGED)
                if rgba is None or rgba.shape != (height, width, 4):
                    raise ValueError(f"Invalid geometry authority frame {global_index:06d}")
                weight = rgba[..., 3].astype(np.float32) / 255
                if protected_root is not None:
                    protected = _binary_mask(
                        protected_root / f"{global_index:06d}.png", (height, width)
                    )
                    weight[protected] = 0
                active = weight > 0
                delta = candidate.astype(np.int16) - reference.astype(np.int16)
                limited = np.clip(delta, -maximum_channel_delta, maximum_channel_delta)
                result = np.clip(
                    reference + limited * (weight * strength)[..., None], 0, 255
                ).astype(np.uint8)
                writer.write(result)
                if np.any(active):
                    changes.append(float(np.mean(np.abs(result.astype(float) - reference)[active])))
                    clipped.append(float(np.mean(np.abs(delta[active]) > maximum_channel_delta)))
                global_index += 1
            capture.release()
    finally:
        writer.release()
    if global_index != int(batch["total_frames"]):
        raise RuntimeError(f"Merged {global_index} of {batch['total_frames']} expected frames")
    result_manifest = {
        "schema": "openarm-mask-constrained-style-v1",
        "batch_manifest": str(manifest_path.resolve()),
        "reference_video": str(reference_path.resolve()),
        "output_video": str(destination.resolve()),
        "rgba_geometry_authority": str(rgba_root.resolve()),
        "protected_masks": str(protected_root.resolve()) if protected_root else None,
        "frames": global_index,
        "strength": strength,
        "maximum_channel_delta": maximum_channel_delta,
        "mean_robot_color_change_255": float(np.mean(changes)) if changes else 0.0,
        "mean_candidate_channel_clip_fraction": float(np.mean(clipped)) if clipped else 0.0,
        "geometry_modified": False,
        "background_modified_before_encoding": False,
        "protected_pixels_modified_before_encoding": False,
        "release_accepted": False,
        "acceptance_requirement": "full independent segmentation and temporal validation",
    }
    destination.with_suffix(destination.suffix + ".manifest.json").write_text(
        json.dumps(result_manifest, indent=2) + "\n"
    )
    return result_manifest


def _load_visual_mask(root: Path, index: int, shape: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(root / f"{index:06d}.png"), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(root / f"{index:06d}.png")
    if image.shape[:2] != shape:
        raise ValueError(f"Mask resolution differs at frame {index}")
    return image[..., 3] > 0 if image.ndim == 3 and image.shape[2] == 4 else image > 0


def validate_style_refinement(
    reference_video: str | Path,
    candidate_video: str | Path,
    reference_robot_masks: str | Path,
    candidate_robot_masks: str | Path,
    protected_mask_dir: str | Path | None = None,
    *,
    minimum_mean_robot_iou: float = 0.9,
    maximum_background_mae: float = 0.04,
    maximum_protected_mae: float = 0.04,
    maximum_background_flow_error_px: float = 0.75,
    maximum_robot_style_temporal_error: float = 0.08,
    maximum_p99_robot_style_temporal_error: float = 0.12,
) -> dict:
    """Gate an optional generative refinement against the deterministic composite.

    Candidate masks must be produced independently from the refined output. Passing the same
    control mask back as a prediction would not test whether the generator changed geometry.
    """
    reference_info = _video_info(reference_video)
    candidate_info = _video_info(candidate_video)
    errors: list[str] = []
    if (reference_info["width"], reference_info["height"]) != (
        candidate_info["width"],
        candidate_info["height"],
    ):
        raise ValueError("Reference and candidate resolutions differ")
    width, height = int(reference_info["width"]), int(reference_info["height"])
    mask_root = Path(reference_robot_masks)
    candidate_root = Path(candidate_robot_masks)
    protected_root = Path(protected_mask_dir) if protected_mask_dir else None
    ious: list[float] = []
    background_mae: list[float] = []
    protected_mae: list[float] = []
    flow_error: list[float] = []
    robot_style_temporal_error: list[float] = []
    previous_reference_gray = previous_candidate_gray = previous_background = None
    previous_style_residual = previous_robot_mask = None
    count = 0
    pairs = zip_longest(
        _iter_bgr_frames(reference_video, width, height),
        _iter_bgr_frames(candidate_video, width, height),
    )
    for index, (reference, candidate) in enumerate(pairs):
        if reference is None or candidate is None:
            errors.append("reference and candidate frame counts differ")
            break
        reference_mask = _load_visual_mask(mask_root, index, (height, width))
        candidate_mask = _load_visual_mask(candidate_root, index, (height, width))
        union = reference_mask | candidate_mask
        union_area = np.count_nonzero(union)
        ious.append(
            float(np.count_nonzero(reference_mask & candidate_mask) / union_area)
            if union_area
            else 1.0
        )
        protected = (
            _binary_mask(protected_root / f"{index:06d}.png", (height, width))
            if protected_root
            else np.zeros((height, width), dtype=bool)
        )
        guard = cv2.dilate(union.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        background = ~guard & ~protected
        delta = np.abs(reference.astype(np.float32) - candidate.astype(np.float32)) / 255
        style_residual = candidate.astype(np.float32) - reference.astype(np.float32)
        if np.any(background):
            background_mae.append(float(np.mean(delta[background])))
        if np.any(protected):
            protected_mae.append(float(np.mean(delta[protected])))
        robot_mask = reference_mask & candidate_mask
        if previous_style_residual is not None:
            stable_robot = robot_mask & previous_robot_mask
            if np.any(stable_robot):
                robot_style_temporal_error.append(
                    float(
                        np.mean(np.abs(style_residual - previous_style_residual)[stable_robot])
                        / 255
                    )
                )
        reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        candidate_gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
        if previous_reference_gray is not None and np.any(background & previous_background):
            scale = 0.25
            size = (max(8, round(width * scale)), max(8, round(height * scale)))
            ref_flow = cv2.calcOpticalFlowFarneback(
                cv2.resize(previous_reference_gray, size),
                cv2.resize(reference_gray, size),
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            )
            cand_flow = cv2.calcOpticalFlowFarneback(
                cv2.resize(previous_candidate_gray, size),
                cv2.resize(candidate_gray, size),
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            )
            stable = (
                cv2.resize(
                    (background & previous_background).astype(np.uint8),
                    size,
                    interpolation=cv2.INTER_NEAREST,
                )
                > 0
            )
            if np.any(stable):
                # Convert quarter-resolution flow differences back to full-resolution pixels.
                flow_error.append(
                    float(np.mean(np.linalg.norm(ref_flow - cand_flow, axis=2)[stable]) / scale)
                )
        previous_reference_gray = reference_gray
        previous_candidate_gray = candidate_gray
        previous_background = background
        previous_style_residual = style_residual
        previous_robot_mask = robot_mask
        count += 1
    metrics = {
        "mean_robot_mask_iou": float(np.mean(ious)) if ious else 0.0,
        "p05_robot_mask_iou": float(np.quantile(ious, 0.05)) if ious else 0.0,
        "background_mae": float(np.mean(background_mae)) if background_mae else 0.0,
        "protected_object_mae": float(np.mean(protected_mae)) if protected_mae else 0.0,
        "background_flow_error_px": float(np.mean(flow_error)) if flow_error else 0.0,
        "robot_style_temporal_error": (
            float(np.mean(robot_style_temporal_error)) if robot_style_temporal_error else 0.0
        ),
        "p99_robot_style_temporal_error": (
            float(np.quantile(robot_style_temporal_error, 0.99))
            if robot_style_temporal_error
            else 0.0
        ),
    }
    limits = {
        "mean_robot_mask_iou": minimum_mean_robot_iou,
        "background_mae": maximum_background_mae,
        "protected_object_mae": maximum_protected_mae,
        "background_flow_error_px": maximum_background_flow_error_px,
        "robot_style_temporal_error": maximum_robot_style_temporal_error,
        "p99_robot_style_temporal_error": maximum_p99_robot_style_temporal_error,
    }
    if metrics["mean_robot_mask_iou"] < minimum_mean_robot_iou:
        errors.append("refinement changed the robot silhouette")
    for metric in (
        "background_mae",
        "protected_object_mae",
        "background_flow_error_px",
        "robot_style_temporal_error",
        "p99_robot_style_temporal_error",
    ):
        if metrics[metric] > limits[metric]:
            errors.append(f"{metric} {metrics[metric]:.4f} exceeds {limits[metric]:.4f}")
    return {"ok": not errors, "frames": count, **metrics, "limits": limits, "errors": errors}


def record_style_validation(
    manifest_path: str | Path,
    candidate_video: str | Path,
    report: dict,
    report_output: str | Path | None = None,
) -> dict:
    """Atomically bind an independent validation report to a constrained-style manifest."""
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "openarm-mask-constrained-style-v1":
        raise ValueError("Acceptance can only update a mask-constrained style manifest")
    recorded_output = Path(manifest["output_video"]).resolve()
    if recorded_output != Path(candidate_video).resolve():
        raise ValueError("Candidate video does not match the constrained-style manifest")
    manifest["release_accepted"] = bool(report.get("ok", False))
    manifest["independent_validation"] = report
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)
    if report_output is not None:
        Path(report_output).write_text(json.dumps(report, indent=2) + "\n")
    return manifest


def write_render_manifest(
    output: str | Path,
    episode_path: str | Path,
    camera_intrinsics: list[list[float]],
    world_from_camera: list[list[float]],
    engine: str = "UnrealRoboticsLab",
) -> Path:
    manifest = {
        "engine": engine,
        "episode": str(Path(episode_path).resolve()),
        "camera": {
            "intrinsics": camera_intrinsics,
            "world_from_camera": world_from_camera,
            "convention": "OpenCV camera: +x right, +y down, +z forward",
        },
        "passes": ["rgba", "depth_m", "robot_segmentation", "shadow_catcher"],
        "requirements": [
            "Use the official OpenArm 2.0 meshes and solved joints",
            "Match source exposure, rolling shutter, distortion, and motion blur",
            "Return linear depth for source-scene occlusion",
        ],
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def write_cosmos_transfer_manifest(
    output: str | Path,
    composited_video: str | Path,
    depth_video: str | Path,
    segmentation_video: str | Path,
    prompt: str,
) -> Path:
    payload = {
        "model_family": "Cosmos-Transfer2.5",
        "input_video": str(composited_video),
        "controls": {"depth": str(depth_video), "segmentation": str(segmentation_video)},
        "prompt": prompt,
        "negative_prompt": "different robot geometry, extra limbs, moving objects, camera motion",
        "validation": {
            "require_robot_mask_iou": 0.9,
            "require_background_flow_consistency": True,
            "reject_if_object_tracks_change": True,
        },
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
