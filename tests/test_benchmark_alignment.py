import importlib.util
from pathlib import Path

import cv2
import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/align_benchmark_renders.py"
SPEC = importlib.util.spec_from_file_location("align_benchmark_renders", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
alignment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(alignment)


def test_similarity_maps_projected_base_and_tool_with_rotation() -> None:
    source_base = np.array([300.0, 200.0])
    source_tool = np.array([400.0, 200.0])
    target_base = np.array([10.0, 470.0])
    target_tool = np.array([10.0, 270.0])
    matrix, scale, angle = alignment._similarity(
        source_base, source_tool, target_base, target_tool
    )
    np.testing.assert_allclose(matrix[:, :2] @ source_base + matrix[:, 2], target_base)
    np.testing.assert_allclose(matrix[:, :2] @ source_tool + matrix[:, 2], target_tool)
    assert scale == 2.0
    np.testing.assert_allclose(angle, -np.pi / 2)


def test_edge_component_keeps_arm_on_requested_spatial_side() -> None:
    mask = np.zeros((480, 640), dtype=np.uint8)
    cv2.line(mask, (0, 250), (180, 220), 30, 1)
    cv2.line(mask, (639, 170), (500, 280), 30, 1)
    left = alignment._edge_component(mask > 0, "left")
    right = alignment._edge_component(mask > 0, "right")
    assert left is not None and right is not None
    assert np.median(np.where(left)[1]) < 320
    assert np.median(np.where(right)[1]) > 320


def test_mask_anchors_reject_short_mount_blob_and_follow_long_arm() -> None:
    short = np.zeros((480, 640), dtype=np.uint8)
    cv2.rectangle(short, (0, 200), (35, 250), 1, -1)
    assert alignment._mask_anchors(short > 0) is None

    arm = np.zeros_like(short)
    cv2.line(arm, (0, 240), (220, 320), 32, 1)
    anchors = alignment._mask_anchors(arm > 0)
    assert anchors is not None
    base, tool = anchors
    assert base[0] < 10
    assert tool[0] > 190
    assert np.linalg.norm(tool - base) > 190


def test_edge_tool_similarity_preserves_shape_and_reaches_source_border() -> None:
    rgba = np.zeros((120, 200, 4), dtype=np.uint8)
    cv2.line(rgba, (60, 60), (140, 60), (255, 255, 255, 255), 20)
    source_base = np.array([60.0, 60.0])
    source_tool = np.array([140.0, 60.0])
    target_entry = np.array([0.0, 90.0])
    target_tool = np.array([160.0, 90.0])
    matrix, scale, angle = alignment._edge_tool_similarity(
        rgba, source_base, source_tool, target_entry, target_tool
    )
    aligned = cv2.warpAffine(rgba, matrix, (200, 120))
    np.testing.assert_allclose(matrix[:, :2] @ source_tool + matrix[:, 2], target_tool)
    assert alignment._entry_error(aligned[..., 3] > 2, target_entry) <= 2
    assert 1.7 < scale < 2.1
    np.testing.assert_allclose(angle, 0.0)


def test_mask_anchors_can_use_known_mounting_border() -> None:
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.line(mask, (40, 199), (150, 50), 24, 1)
    cv2.rectangle(mask, (0, 100), (30, 190), 1, -1)
    anchors = alignment._mask_anchors(mask > 0, preferred_border=3)
    assert anchors is not None
    base, tool = anchors
    assert base[1] >= 190
    assert tool[1] < 80
