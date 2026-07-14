#!/usr/bin/env python3
"""Register uncalibrated OpenArm preview renders to audited source robot masks."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "outputs/cross_dataset_openarm_benchmark"


def bbox(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def smooth_boxes(boxes: list[tuple[float, float, float, float] | None]) -> np.ndarray:
    valid = [item for item in boxes if item is not None]
    if not valid:
        raise RuntimeError("No non-empty source masks")
    fallback = np.median(np.asarray(valid), axis=0)
    values = np.asarray([item if item is not None else fallback for item in boxes], dtype=np.float32)
    for column in range(4):
        values[:, column] = median_filter(values[:, column], size=9, mode="nearest")
        values[:, column] = gaussian_filter1d(values[:, column], sigma=2.0, mode="nearest")
    return values


def affine_for_boxes(
    target: tuple[float, float, float, float],
    source: np.ndarray,
    max_scale: float | None = None,
) -> tuple[np.ndarray, list[float]]:
    tx1, ty1, tx2, ty2 = target
    sx1, sy1, sx2, sy2 = source
    target_width, target_height = max(tx2 - tx1, 1.0), max(ty2 - ty1, 1.0)
    source_width, source_height = max(sx2 - sx1, 1.0), max(sy2 - sy1, 1.0)
    scale_x = 0.94 * source_width / target_width
    scale_y = 0.94 * source_height / target_height
    ratio = scale_x / max(scale_y, 1e-6)
    if ratio > 1.6:
        scale_x = 1.6 * scale_y
    elif ratio < 1 / 1.6:
        scale_y = 1.6 * scale_x
    if max_scale is not None and max(scale_x, scale_y) > max_scale:
        reduction = max_scale / max(scale_x, scale_y)
        scale_x *= reduction
        scale_y *= reduction
    target_cx, target_cy = (tx1 + tx2) / 2.0, (ty1 + ty2) / 2.0
    source_cx, source_cy = (sx1 + sx2) / 2.0, (sy1 + sy2) / 2.0
    matrix = np.asarray(
        [
            [scale_x, 0.0, source_cx - scale_x * target_cx],
            [0.0, scale_y, source_cy - scale_y * target_cy],
        ],
        dtype=np.float32,
    )
    return matrix, [float(scale_x), float(scale_y)]


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
    return np.clip(np.concatenate([rgb, output_alpha], axis=2) * 255.0, 0, 255).astype(np.uint8)


def main() -> None:
    for clip in sorted(BENCHMARK.glob("*/*")):
        render_paths = sorted((clip / "render_raw").glob("[0-9]*.png"))
        if not render_paths:
            continue
        source_boxes = []
        for index in range(len(render_paths)):
            mask = cv2.imread(str(clip / "masks_final" / f"{index:06d}.png"), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(clip / "masks_final" / f"{index:06d}.png")
            source_boxes.append(bbox(mask > 127))
        registered_boxes = smooth_boxes(source_boxes)
        output = clip / "render_aligned"
        output.mkdir(exist_ok=True)
        scales = []
        translations = []
        for index, render_path in enumerate(render_paths):
            rgba = cv2.imread(str(render_path), cv2.IMREAD_UNCHANGED)
            if rgba is None or rgba.shape[2] != 4:
                raise RuntimeError(f"Expected RGBA render: {render_path}")
            target = bbox(rgba[..., 3] > 2)
            if target is None:
                raise RuntimeError(f"Empty render alpha: {render_path}")
            matrix, scale = affine_for_boxes(target, registered_boxes[index])
            aligned = cv2.warpAffine(
                rgba,
                matrix,
                (640, 480),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0),
            )
            if not cv2.imwrite(str(output / f"{index:06d}.png"), aligned):
                raise RuntimeError(f"Could not write aligned render for {clip}")
            scales.append(scale)
            translations.append([float(matrix[0, 2]), float(matrix[1, 2])])
        side_registered = False
        if (clip / "render_raw_left").exists() and (clip / "render_raw_right").exists():
            side_outputs: dict[str, list[np.ndarray]] = {}
            for side in ("left", "right"):
                side_paths = sorted((clip / f"render_raw_{side}").glob("[0-9]*.png"))
                side_boxes = []
                visible = []
                for index in range(len(side_paths)):
                    mask = cv2.imread(
                        str(clip / "masks_final" / f"{index:06d}.png"),
                        cv2.IMREAD_GRAYSCALE,
                    ) > 127
                    if side == "left":
                        mask[:, 320:] = False
                    else:
                        mask[:, :320] = False
                    visible.append(int(mask.sum()) >= 32)
                    side_boxes.append(bbox(mask))
                smoothed = smooth_boxes(side_boxes)
                aligned_side = []
                for index, path in enumerate(side_paths):
                    rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                    target = bbox(rgba[..., 3] > 2)
                    if target is None or not visible[index]:
                        aligned_side.append(np.zeros((480, 640, 4), dtype=np.uint8))
                        continue
                    matrix, _ = affine_for_boxes(target, smoothed[index], max_scale=2.0)
                    aligned_side.append(
                        cv2.warpAffine(
                            rgba,
                            matrix,
                            (640, 480),
                            flags=cv2.INTER_LANCZOS4,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0, 0, 0, 0),
                        )
                    )
                side_outputs[side] = aligned_side
            for index, (left, right) in enumerate(
                zip(side_outputs["left"], side_outputs["right"], strict=True)
            ):
                combined = alpha_over(left, right)
                if not cv2.imwrite(str(output / f"{index:06d}.png"), combined):
                    raise RuntimeError(f"Could not write side-registered render for {clip}")
            side_registered = True
        manifest = {
            "method": "smoothed mask-bounding-box affine registration",
            "frames": len(render_paths),
            "projection_calibrated": False,
            "mean_scale_xy": np.mean(scales, axis=0).tolist(),
            "mean_translation_xy": np.mean(translations, axis=0).tolist(),
            "warning": "Suitable for a visual baseline, not metric image-space supervision.",
            "independent_side_registration": side_registered,
        }
        (output / "alignment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(clip.relative_to(BENCHMARK), json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
