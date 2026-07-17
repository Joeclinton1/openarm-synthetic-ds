#!/usr/bin/env python3
"""Fuse and clean benchmark robot masks after visual model inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = ROOT / "outputs/cross_dataset_openarm_benchmark"


def read(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image > 127


def edge_components(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    result = np.zeros_like(mask)
    height, width = mask.shape
    for label in range(1, count):
        x, y, box_width, box_height, area = stats[label]
        touches_edge = x <= 2 or x + box_width >= width - 2 or y + box_height >= height - 2
        if touches_edge and area >= 32:
            result |= labels == label
    return result


def non_bottom_edge_components(mask: np.ndarray) -> np.ndarray:
    """Keep dark robot housings entering from the top/sides, excluding the dark tabletop."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    result = np.zeros_like(mask)
    height, width = mask.shape
    for label in range(1, count):
        x, y, box_width, box_height, area = stats[label]
        touches_entry_edge = x <= 2 or x + box_width >= width - 2 or y <= 2
        touches_bottom = y + box_height >= height - 2
        if touches_entry_edge and not touches_bottom and area >= 32:
            result |= labels == label
    return result


def horizontal_edge_components(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    result = np.zeros_like(mask)
    width = mask.shape[1]
    for label in range(1, count):
        x, _, box_width, _, area = stats[label]
        if (x <= 2 or x + box_width >= width - 2) and area >= 80:
            result |= labels == label
    return result


def nearby_gripper(robot: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    nearby = cv2.dilate(
        robot.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (49, 49)),
    ).astype(bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        gripper.astype(np.uint8), connectivity=8
    )
    accepted = np.zeros_like(robot)
    for label in range(1, count):
        component = labels == label
        if stats[label, cv2.CC_STAT_AREA] >= 12 and np.any(component & nearby):
            accepted |= component
    return accepted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument(
        "--accepted-agibot-fixture",
        type=Path,
        help="Optional full-episode accepted masks used for the AgiBot demo-B slice",
    )
    parser.add_argument(
        "--accepted-agibot-start-frame",
        type=int,
        default=805,
        help="First full-episode mask corresponding to frame zero of AgiBot demo B",
    )
    args = parser.parse_args()
    benchmark = args.benchmark.resolve()
    accepted_fixture = (
        args.accepted_agibot_fixture.resolve() if args.accepted_agibot_fixture else None
    )
    if accepted_fixture is not None and not accepted_fixture.is_dir():
        raise NotADirectoryError(accepted_fixture)
    for clip in sorted(benchmark.glob("*/*")):
        if not (clip / "01_source.mp4").exists():
            continue
        dataset = clip.parent.name
        frame_paths = sorted((clip / "masks_robotseg").glob("[0-9]*.png"))
        if not frame_paths:
            raise FileNotFoundError(f"No RobotSeg masks found in {clip / 'masks_robotseg'}")
        output = clip / "masks_final"
        output.mkdir(exist_ok=True)
        capture = cv2.VideoCapture(str(clip / "01_source.mp4"))
        areas = []
        added = []
        source_description = "RobotSeg"
        for index, robot_path in enumerate(frame_paths):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode frame {index} from {clip}")
            robotseg = read(robot_path)
            gripper = read(clip / "masks_gripper" / f"{index:06d}.png")
            if dataset == "agibot_world_alpha":
                if "demo_b" in clip.name and accepted_fixture is not None:
                    accepted_path = (
                        accepted_fixture
                        / f"{args.accepted_agibot_start_frame + index:06d}.png"
                    )
                    combined = read(accepted_path)
                    gripper_addition = np.zeros_like(combined)
                    source_description = (
                        "accepted full-fixture masks, starting at frame "
                        f"{args.accepted_agibot_start_frame}"
                    )
                else:
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    dark = (hsv[..., 2] < 105) & (hsv[..., 1] < 170)
                    dark = cv2.morphologyEx(
                        dark.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
                    ).astype(bool)
                    combined = edge_components(robotseg) | horizontal_edge_components(dark)
                    gripper_addition = nearby_gripper(combined, gripper)
                    source_description = "edge-filtered RobotSeg plus dark-arm appearance prior"
            elif dataset == "robomind_agilex_3rgb":
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                dark = (hsv[..., 2] < 100) & (hsv[..., 1] < 190)
                dark = cv2.morphologyEx(
                    dark.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
                ).astype(bool)
                combined = robotseg | non_bottom_edge_components(dark)
                source_description = (
                    "RobotSeg plus non-bottom-edge dark AgileX appearance prior"
                )
                gripper_addition = nearby_gripper(combined, gripper)
            else:
                combined = robotseg
                gripper_addition = nearby_gripper(combined, gripper)
            combined |= gripper_addition
            combined = cv2.morphologyEx(
                combined.astype(np.uint8),
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            )
            combined = cv2.dilate(
                combined,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
            ).astype(bool)
            if not cv2.imwrite(str(output / f"{index:06d}.png"), combined.astype(np.uint8) * 255):
                raise RuntimeError(f"Could not write mask {index} for {clip}")
            areas.append(float(combined.mean()))
            added.append(float(gripper_addition.mean()))
        capture.release()
        manifest = {
            "method": "dataset-audited RobotSeg/SAM2 fusion with proximity-gated gripper recall",
            "dataset": dataset,
            "frames": len(frame_paths),
            "mean_mask_fraction": float(np.mean(areas)),
            "maximum_mask_fraction": float(np.max(areas)),
            "mean_gripper_added_fraction": float(np.mean(added)),
            "morphology": {"closing_diameter_px": 7, "dilation_diameter_px": 11},
            "agibot_edge_component_filter": dataset == "agibot_world_alpha",
            "mask_source": source_description,
            "accepted_agibot_fixture": (
                str(accepted_fixture) if accepted_fixture is not None else None
            ),
        }
        (output / "mask_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(clip.relative_to(benchmark), json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
