import json

import cv2
import numpy as np
import pytest

from openarm_retarget.ai_masks import (
    _read_video_chunks,
    carry_object_box,
    carry_robot_boxes,
    combine_object_masks,
    select_prompt_boxes,
    select_robot_boxes,
)
from openarm_retarget.media import (
    apply_mask_constrained_style,
    composite_rgba,
    distort_rgba_frames,
    distort_depth_frames,
    fuse_robot_gripper_masks,
    harmonize_rgba_frames,
    inpaint_static_camera,
    inpaint_video,
    refine_robot_masks,
    record_style_validation,
    restore_protected_video,
    stabilize_masks,
    validate_inpainting,
    validate_harmonized_rgba,
    validate_depth_render,
    validate_composite_video,
    validate_robot_masks,
    validate_style_refinement,
    validate_rgba_render,
    validate_render_alignment,
    validate_embodiment_alignment,
)


def test_rgba_composite() -> None:
    background = np.full((1, 1, 3), 10, dtype=np.uint8)
    foreground = np.array([[[110, 210, 10, 128]]], dtype=np.uint8)
    result = composite_rgba(background, foreground)
    np.testing.assert_allclose(result[0, 0], [60, 110, 10], atol=1)


def test_composite_validation_checks_inserted_region(tmp_path) -> None:
    background = tmp_path / "background.mp4"
    result = tmp_path / "result.mp4"
    rgba = tmp_path / "rgba"
    rgba.mkdir()
    bg_writer = cv2.VideoWriter(str(background), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 24))
    out_writer = cv2.VideoWriter(str(result), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 24))
    for index in range(2):
        frame = np.full((24, 32, 3), 40, dtype=np.uint8)
        changed = frame.copy()
        changed[8:16, 12:20] = 180
        bg_writer.write(frame)
        out_writer.write(changed)
        layer = np.zeros((24, 32, 4), dtype=np.uint8)
        layer[8:16, 12:20, 3] = 255
        cv2.imwrite(str(rgba / f"{index:06d}.png"), layer)
    bg_writer.release()
    out_writer.release()
    report = validate_composite_video(background, result, rgba)
    assert report["ok"]
    assert report["robot_region_change"] > 0.1


def test_depth_composite_preserves_nearer_scene_surface() -> None:
    background = np.full((1, 2, 3), 10, dtype=np.uint8)
    foreground = np.full((1, 2, 4), [110, 210, 10, 255], dtype=np.uint8)
    source_depth = np.array([[0.5, 2.0]], dtype=np.float32)
    robot_depth = np.array([[1.0, 1.0]], dtype=np.float32)
    result = composite_rgba(background, foreground, source_depth, robot_depth)
    np.testing.assert_array_equal(result[0, 0], background[0, 0])
    np.testing.assert_array_equal(result[0, 1], foreground[0, 1, :3])


def test_temporal_mask_stabilization() -> None:
    masks = [np.zeros((5, 5), dtype=np.uint8) for _ in range(3)]
    masks[1][2, 2] = 255
    stable = stabilize_masks(masks, dilation=1, temporal_radius=1)
    assert all(mask[2, 2] == 255 for mask in stable)


def test_select_bimanual_robot_boxes_rejects_whole_frame() -> None:
    detections = [
        {"score": 0.9, "box": {"xmin": 0, "ymin": 0, "xmax": 640, "ymax": 480}},
        {"score": 0.8, "box": {"xmin": 0, "ymin": 260, "xmax": 250, "ymax": 480}},
        {"score": 0.7, "box": {"xmin": 370, "ymin": 255, "xmax": 640, "ymax": 480}},
        {"score": 0.85, "box": {"xmin": 0, "ymin": 250, "xmax": 640, "ymax": 480}},
    ]
    assert select_robot_boxes(detections, 640, 480, expansion_fraction=0) == [
        [0.0, 260.0, 250.0, 480.0],
        [370.0, 255.0, 640.0, 480.0],
    ]


def test_select_robot_boxes_fails_without_plausible_arm() -> None:
    with pytest.raises(RuntimeError, match="No plausible"):
        select_robot_boxes([], 640, 480, expected_arms=1)


def test_combine_object_masks_preserves_both_objects() -> None:
    masks = np.zeros((1, 2, 1, 5, 6), dtype=bool)
    masks[0, 0, 0, 1, 1] = True
    masks[0, 1, 0, 3, 4] = True
    result = combine_object_masks(masks)
    assert result.shape == (5, 6)
    assert result[1, 1] and result[3, 4]


def test_select_edge_robot_housings_as_additional_objects() -> None:
    detections = [
        {"score": 0.8, "box": {"xmin": 0, "ymin": 260, "xmax": 250, "ymax": 480}},
        {"score": 0.7, "box": {"xmin": 370, "ymin": 255, "xmax": 640, "ymax": 480}},
        {"score": 0.6, "box": {"xmin": 0, "ymin": 230, "xmax": 45, "ymax": 380}},
        {"score": 0.5, "box": {"xmin": 560, "ymin": 225, "xmax": 640, "ymax": 375}},
    ]
    boxes = select_robot_boxes(
        detections,
        640,
        480,
        expansion_fraction=0,
        include_edge_components=True,
    )
    assert len(boxes) == 4
    assert boxes[2:] == [[0.0, 230.0, 45.0, 380.0], [560.0, 225.0, 640.0, 375.0]]


def test_partial_bimanual_detection_keeps_visible_side() -> None:
    detection = [
        {
            "score": 0.8,
            "box": {"xmin": 0, "ymin": 100, "xmax": 200, "ymax": 400},
        }
    ]
    boxes = select_robot_boxes(detection, 640, 480, expansion_fraction=0, allow_partial=True)
    assert boxes == [[0.0, 100.0, 200.0, 400.0]]


def test_select_prompt_boxes_rejects_scene_box_and_keeps_best_object() -> None:
    detections = [
        {"score": 0.99, "box": {"xmin": 0, "ymin": 0, "xmax": 640, "ymax": 480}},
        {"score": 0.8, "box": {"xmin": 100, "ymin": 120, "xmax": 180, "ymax": 230}},
        {"score": 0.6, "box": {"xmin": 300, "ymin": 200, "xmax": 360, "ymax": 270}},
    ]
    assert select_prompt_boxes(detections, 640, 480, expansion_fraction=0, max_objects=1) == [
        [100.0, 120.0, 180.0, 230.0]
    ]


def test_carry_object_box_retains_initial_extent_for_fragmented_track() -> None:
    mask = np.zeros((100, 200), dtype=bool)
    mask[45:50, 95:100] = True
    box = carry_object_box(mask, 200, 100, minimum_size=(60.0, 40.0), expansion_fraction=0.1)
    assert box is not None
    assert np.isclose(box[2] - box[0], 72.0)
    assert np.isclose(box[3] - box[1], 48.0)
    assert box[0] < 95 < box[2]


def test_carry_robot_boxes_rejects_free_scene_objects() -> None:
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[20:80, :30] = 1
    mask[10:75, 175:] = 1
    mask[40:70, 80:110] = 1

    boxes = carry_robot_boxes(mask, 200, 100, expansion_fraction=0)

    assert boxes is not None
    assert sorted(boxes) == [[0.0, 20.0, 30.0, 80.0], [175.0, 10.0, 200.0, 75.0]]


def test_inpaint_video_streams_and_preserves_frame_count(tmp_path) -> None:
    source = tmp_path / "source.mp4"
    masks = tmp_path / "masks"
    masks.mkdir()
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
    for index in range(7):
        frame = np.full((24, 32, 3), 40 + index, dtype=np.uint8)
        frame[8:16, 12:20] = 255
        writer.write(frame)
        mask = np.zeros((24, 32), dtype=np.uint8)
        mask[8:16, 12:20] = 255
        cv2.imwrite(str(masks / f"{index:06d}.png"), mask)
    writer.release()
    output = inpaint_video(source, masks, tmp_path / "inpainted.mp4")
    capture = cv2.VideoCapture(str(output))
    count = 0
    while capture.read()[0]:
        count += 1
    capture.release()
    assert count == 7


def test_gripper_mask_fusion_keeps_only_components_near_robot(tmp_path) -> None:
    robot = tmp_path / "robot"
    gripper = tmp_path / "gripper"
    fused = tmp_path / "fused"
    robot.mkdir()
    gripper.mkdir()
    primary = np.zeros((40, 60), dtype=np.uint8)
    primary[12:28, 5:20] = 255
    auxiliary = np.zeros_like(primary)
    auxiliary[15:25, 22:30] = 255
    auxiliary[4:12, 48:56] = 255
    cv2.imwrite(str(robot / "000000.png"), primary)
    cv2.imwrite(str(gripper / "000000.png"), auxiliary)
    fuse_robot_gripper_masks(robot, gripper, fused, proximity_radius=4)
    result = cv2.imread(str(fused / "000000.png"), cv2.IMREAD_GRAYSCALE)
    assert np.all(result[15:25, 22:30] == 255)
    assert not np.any(result[4:12, 48:56])


def test_static_camera_clean_plate_recovers_temporally_visible_background(tmp_path) -> None:
    source = tmp_path / "source.mp4"
    masks = tmp_path / "masks"
    masks.mkdir()
    width, height = 64, 48
    x = np.arange(width, dtype=np.uint8)[None, :]
    y = np.arange(height, dtype=np.uint8)[:, None]
    background = np.stack(
        [
            np.broadcast_to(40 + x, (height, width)),
            np.broadcast_to(60 + y, (height, width)),
            np.full((height, width), 90, dtype=np.uint8),
        ],
        axis=2,
    )
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height))
    for index in range(8):
        frame = background.copy()
        left = 4 + index * 6
        frame[16:30, left : left + 12] = (10, 10, 220)
        writer.write(frame)
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[16:30, left : left + 12] = 255
        cv2.imwrite(str(masks / f"{index:06d}.png"), mask)
    writer.release()
    output = inpaint_static_camera(
        source,
        masks,
        tmp_path / "clean.mp4",
        sample_stride=1,
        minimum_clean_observations=2,
        feather_radius=0,
    )
    capture = cv2.VideoCapture(str(output))
    ok, recovered = capture.read()
    capture.release()
    assert ok
    region = recovered[16:30, 4:16].astype(np.float32)
    expected = background[16:30, 4:16].astype(np.float32)
    assert np.mean(np.abs(region - expected)) < 8


def test_static_camera_uses_neural_fallback_only_for_never_seen_pixels(tmp_path) -> None:
    source = tmp_path / "source.mp4"
    fallback = tmp_path / "fallback.mp4"
    masks = tmp_path / "masks"
    masks.mkdir()
    source_writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 24))
    fallback_writer = cv2.VideoWriter(str(fallback), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 24))
    for index in range(4):
        frame = np.full((24, 32, 3), 40, dtype=np.uint8)
        frame[8:16, 12:20] = 220
        source_writer.write(frame)
        fallback_writer.write(np.full_like(frame, 130))
        mask = np.zeros((24, 32), dtype=np.uint8)
        mask[8:16, 12:20] = 255
        cv2.imwrite(str(masks / f"{index:06d}.png"), mask)
    source_writer.release()
    fallback_writer.release()

    output = inpaint_static_camera(
        source,
        masks,
        tmp_path / "clean.mp4",
        fallback_video=fallback,
        sample_stride=1,
        feather_radius=0,
    )
    capture = cv2.VideoCapture(str(output))
    ok, recovered = capture.read()
    capture.release()
    assert ok
    assert 115 < float(np.mean(recovered[9:15, 13:19])) < 140
    assert 30 < float(np.mean(recovered[:6, :6])) < 55


def test_render_validation_rejects_offscreen_robot(tmp_path) -> None:
    empty = np.zeros((20, 30, 4), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "000000.png"), empty)
    report = validate_rgba_render(tmp_path, expected_frames=1)
    assert not report["ok"]
    assert report["visible_fraction"] == 0


def test_depth_render_validation_accepts_metric_robot_depth(tmp_path) -> None:
    rgba = tmp_path / "rgba"
    depth = tmp_path / "depth"
    rgba.mkdir()
    depth.mkdir()
    image = np.zeros((10, 12, 4), dtype=np.uint8)
    image[2:8, 3:9, 3] = 255
    values = np.full((10, 12), np.inf, dtype=np.float32)
    values[2:8, 3:9] = 0.7
    cv2.imwrite(str(rgba / "000000.png"), image)
    np.savez_compressed(depth / "000000.npz", depth_m=values)
    report = validate_depth_render(rgba, depth)
    assert report["ok"]
    assert report["mean_alpha_depth_coverage"] == 1.0


def test_render_alignment_measures_alpha_iou(tmp_path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    first = np.zeros((10, 10, 4), dtype=np.uint8)
    second = first.copy()
    first[2:6, 2:6, 3] = 255
    second[2:6, 3:7, 3] = 255
    cv2.imwrite(str(reference / "000000.png"), first)
    cv2.imwrite(str(candidate / "000000.png"), second)
    report = validate_render_alignment(reference, candidate, minimum_mean_iou=0.5)
    assert report["ok"]
    assert report["mean_silhouette_iou"] == pytest.approx(0.6)


def test_embodiment_alignment_uses_replacement_containment(tmp_path) -> None:
    masks = tmp_path / "masks"
    renders = tmp_path / "renders"
    masks.mkdir()
    renders.mkdir()
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[1:9, 1:9] = 255
    render = np.zeros((10, 10, 4), dtype=np.uint8)
    render[3:7, 3:7, 3] = 255
    cv2.imwrite(str(masks / "000000.png"), mask)
    cv2.imwrite(str(renders / "000000.png"), render)
    report = validate_embodiment_alignment(masks, renders)
    assert report["ok"]
    assert report["mean_replacement_containment"] == 1.0


def test_zero_distortion_preserves_rgba_frame(tmp_path) -> None:
    source = tmp_path / "rgba"
    source.mkdir()
    image = np.zeros((8, 10, 4), dtype=np.uint8)
    image[2:6, 3:7] = [20, 40, 60, 255]
    cv2.imwrite(str(source / "000000.png"), image)
    camera = tmp_path / "camera.json"
    camera.write_text(
        json.dumps(
            {
                "intrinsics": [[100, 0, 5], [0, 100, 4], [0, 0, 1]],
                "distortion": [0, 0, 0, 0, 0],
            }
        )
    )
    output = distort_rgba_frames(source, camera, tmp_path / "distorted")
    np.testing.assert_array_equal(cv2.imread(str(output / "000000.png"), -1), image)


def test_zero_distortion_preserves_metric_depth(tmp_path) -> None:
    source = tmp_path / "depth"
    source.mkdir()
    values = np.full((8, 10), np.inf, dtype=np.float32)
    values[2:6, 3:7] = 0.75
    np.savez_compressed(source / "000000.npz", depth_m=values)
    camera = tmp_path / "camera.json"
    camera.write_text(
        json.dumps(
            {
                "intrinsics": [[100, 0, 5], [0, 100, 4], [0, 0, 1]],
                "distortion": [0, 0, 0, 0, 0],
            }
        )
    )
    output = distort_depth_frames(source, camera, tmp_path / "distorted_depth")
    with np.load(output / "000000.npz") as payload:
        result = payload["depth_m"]
    np.testing.assert_array_equal(np.isfinite(result), np.isfinite(values))
    np.testing.assert_allclose(result[np.isfinite(result)], 0.75)


def test_harmonization_preserves_alpha_exactly(tmp_path) -> None:
    video = tmp_path / "background.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 24))
    writer.write(np.full((24, 32, 3), [40, 90, 130], dtype=np.uint8))
    writer.release()
    rgba = tmp_path / "rgba"
    rgba.mkdir()
    image = np.zeros((24, 32, 4), dtype=np.uint8)
    image[7:17, 10:22] = [200, 200, 200, 255]
    cv2.imwrite(str(rgba / "000000.png"), image)
    output = harmonize_rgba_frames(video, rgba, tmp_path / "harmonized")
    result = cv2.imread(str(output / "000000.png"), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(result[..., 3], image[..., 3])
    assert np.mean(result[7:17, 10:22, :3]) != np.mean(image[7:17, 10:22, :3])
    assert validate_harmonized_rgba(rgba, output)["ok"]


def test_style_refinement_gate_accepts_identical_video(tmp_path) -> None:
    video = tmp_path / "reference.mp4"
    masks = tmp_path / "masks"
    masks.mkdir()
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 24))
    for index in range(2):
        frame = np.full((24, 32, 3), 60 + index, dtype=np.uint8)
        writer.write(frame)
        mask = np.zeros((24, 32), dtype=np.uint8)
        mask[8:16, 12:20] = 255
        cv2.imwrite(str(masks / f"{index:06d}.png"), mask)
    writer.release()
    report = validate_style_refinement(video, video, masks, masks)
    assert report["ok"]
    assert report["mean_robot_mask_iou"] == 1
    assert report["background_mae"] == 0


def test_mask_constrained_style_cannot_change_background_or_protected_pixels(tmp_path) -> None:
    reference = tmp_path / "reference.mp4"
    candidate = tmp_path / "candidate.mp4"
    rgba = tmp_path / "rgba"
    protected = tmp_path / "protected"
    rgba.mkdir()
    protected.mkdir()
    reference_writer = cv2.VideoWriter(
        str(reference), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 24)
    )
    candidate_writer = cv2.VideoWriter(
        str(candidate), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 24)
    )
    for index in range(2):
        reference_writer.write(np.full((24, 32, 3), 40, dtype=np.uint8))
        candidate_writer.write(np.full((24, 32, 3), 180, dtype=np.uint8))
        layer = np.zeros((24, 32, 4), dtype=np.uint8)
        layer[6:18, 8:24, 3] = 255
        cv2.imwrite(str(rgba / f"{index:06d}.png"), layer)
        keep = np.zeros((24, 32), dtype=np.uint8)
        keep[10:14, 14:18] = 255
        cv2.imwrite(str(protected / f"{index:06d}.png"), keep)
    reference_writer.release()
    candidate_writer.release()
    output = tmp_path / "safe.mp4"
    report = apply_mask_constrained_style(
        reference,
        candidate,
        rgba,
        output,
        protected_mask_dir=protected,
        strength=1,
        maximum_channel_delta=50,
    )
    assert report["geometry_modified"] is False
    assert report["mean_robot_color_change_255"] > 40
    reference_capture = cv2.VideoCapture(str(reference))
    reference_ok, reference_frame = reference_capture.read()
    reference_capture.release()
    assert reference_ok
    capture = cv2.VideoCapture(str(output))
    ok, result = capture.read()
    capture.release()
    assert ok
    outside = np.ones((24, 32), dtype=bool)
    outside[6:18, 8:24] = False
    assert np.mean(np.abs(result.astype(float) - reference_frame)[outside]) < 3
    assert np.mean(np.abs(result.astype(float) - reference_frame)[10:14, 14:18]) < 4
    assert np.mean(np.abs(result.astype(float) - reference_frame)[7:10, 9:14]) > 35
    manifest = record_style_validation(
        output.with_suffix(output.suffix + ".manifest.json"),
        output,
        {"ok": True, "mean_robot_mask_iou": 1.0},
        tmp_path / "validation.json",
    )
    assert manifest["release_accepted"] is True
    assert json.loads((tmp_path / "validation.json").read_text())["ok"] is True


def test_ffmpeg_chunk_decoder_reads_every_frame(tmp_path) -> None:
    source = tmp_path / "decode.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
    for value in range(5):
        writer.write(np.full((24, 32, 3), value * 20, dtype=np.uint8))
    writer.release()
    chunks = list(_read_video_chunks(source, 32, 24, chunk_frames=2, max_frames=None))
    assert [len(chunk) for chunk in chunks] == [2, 2, 1]


def test_refine_masks_closes_dilates_and_protects_object(tmp_path) -> None:
    video = tmp_path / "source.mp4"
    raw = tmp_path / "raw"
    protected = tmp_path / "protected"
    refined = tmp_path / "refined"
    raw.mkdir()
    protected.mkdir()
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
    for index in range(3):
        frame = np.full((24, 32, 3), 40, dtype=np.uint8)
        frame[8:16, 10 + index : 18 + index] = 180
        writer.write(frame)
        mask = np.zeros((24, 32), dtype=np.uint8)
        mask[8:16, 10 + index : 18 + index] = 255
        mask[11, 13 + index] = 0
        keep = np.zeros_like(mask)
        keep[10:14, 14 + index : 17 + index] = 255
        cv2.imwrite(str(raw / f"{index:06d}.png"), mask)
        cv2.imwrite(str(protected / f"{index:06d}.png"), keep)
    writer.release()
    manifest = refine_robot_masks(
        video,
        raw,
        refined,
        protected_mask_dir=protected,
        dilation_radius=2,
        closing_radius=1,
        protect_margin=0,
        use_optical_flow=False,
    )
    assert manifest.is_file()
    result = cv2.imread(str(refined / "000001.png"), cv2.IMREAD_GRAYSCALE)
    assert result[7, 12] == 255
    assert np.all(result[10:14, 15:18] == 0)


def test_validate_inpainting_accepts_clean_static_reconstruction(tmp_path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    masks = tmp_path / "masks"
    masks.mkdir()
    source_writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
    output_writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
    for index in range(4):
        frame = np.full((24, 32, 3), 90, dtype=np.uint8)
        source_writer.write(frame)
        reconstructed = frame.copy()
        reconstructed[8:16, 12:20] = 180
        output_writer.write(reconstructed)
        mask = np.zeros((24, 32), dtype=np.uint8)
        mask[8:16, 12:20] = 255
        cv2.imwrite(str(masks / f"{index:06d}.png"), mask)
    source_writer.release()
    output_writer.release()
    report = validate_inpainting(source, output, masks)
    assert report["ok"]
    assert report["frames"] == 4


def test_validate_inpainting_rejects_copied_robot_residual(tmp_path) -> None:
    source = tmp_path / "source.mp4"
    masks = tmp_path / "masks"
    masks.mkdir()
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
    for index in range(4):
        frame = np.full((24, 32, 3), 90, dtype=np.uint8)
        frame[6:18, 10:22] = 20
        writer.write(frame)
        mask = np.zeros((24, 32), dtype=np.uint8)
        mask[6:18, 10:22] = 255
        cv2.imwrite(str(masks / f"{index:06d}.png"), mask)
    writer.release()
    report = validate_inpainting(source, source, masks)
    assert not report["ok"]
    assert report["inside_copy_fraction"] > 0.9
    assert len(report["worst_inside_copy_frames"]) == 4
    assert report["worst_inside_copy_frames"][0]["copy_fraction"] > 0.9


def test_inpainting_copy_audit_uses_undilated_source_robot_mask(tmp_path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    removal = tmp_path / "removal"
    robot = tmp_path / "robot"
    removal.mkdir()
    robot.mkdir()
    source_writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (40, 30))
    output_writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (40, 30))
    for index in range(4):
        frame = np.full((30, 40, 3), 90, dtype=np.uint8)
        frame[10:20, 15:25] = 20
        reconstructed = frame.copy()
        reconstructed[10:20, 15:25] = 150
        source_writer.write(frame)
        output_writer.write(reconstructed)
        broad = np.zeros((30, 40), dtype=np.uint8)
        broad[5:25, 8:32] = 255
        exact = np.zeros_like(broad)
        exact[10:20, 15:25] = 255
        cv2.imwrite(str(removal / f"{index:06d}.png"), broad)
        cv2.imwrite(str(robot / f"{index:06d}.png"), exact)
    source_writer.release()
    output_writer.release()
    report = validate_inpainting(
        source,
        output,
        removal,
        source_robot_mask_dir=robot,
    )
    assert report["inside_copy_fraction"] < 0.1
    assert report["source_robot_masks"] == str(robot.resolve())


def test_restore_protected_video_restores_only_tracked_object(tmp_path) -> None:
    source = tmp_path / "source.mp4"
    clean = tmp_path / "clean.mp4"
    masks = tmp_path / "masks"
    masks.mkdir()
    source_writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
    clean_writer = cv2.VideoWriter(str(clean), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
    for index in range(3):
        source_frame = np.full((24, 32, 3), 30, dtype=np.uint8)
        source_frame[7:17, 11:21] = 210
        source_writer.write(source_frame)
        clean_writer.write(np.full_like(source_frame, 80))
        mask = np.zeros((24, 32), dtype=np.uint8)
        mask[7:17, 11:21] = 255
        cv2.imwrite(str(masks / f"{index:06d}.png"), mask)
    source_writer.release()
    clean_writer.release()
    output = restore_protected_video(
        source, clean, masks, tmp_path / "restored.mp4", feather_radius=0
    )
    capture = cv2.VideoCapture(str(output))
    ok, frame = capture.read()
    capture.release()
    assert ok
    assert float(np.mean(frame[9:15, 13:19])) > 180
    assert 65 < float(np.mean(frame[0:5, 0:5])) < 95
    assert output.with_suffix(".restore.json").is_file()


def test_restore_protected_video_excludes_old_robot_overlap(tmp_path) -> None:
    source = tmp_path / "source.mp4"
    clean = tmp_path / "clean.mp4"
    protected = tmp_path / "protected"
    robot = tmp_path / "robot"
    protected.mkdir()
    robot.mkdir()
    source_writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
    clean_writer = cv2.VideoWriter(str(clean), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
    source_frame = np.full((24, 32, 3), 30, dtype=np.uint8)
    source_frame[6:18, 10:22] = 200
    source_frame[6:10, 14:18] = 5
    source_writer.write(source_frame)
    clean_writer.write(np.full_like(source_frame, 80))
    source_writer.release()
    clean_writer.release()
    object_mask = np.zeros((24, 32), dtype=np.uint8)
    object_mask[6:18, 10:22] = 255
    robot_mask = np.zeros_like(object_mask)
    robot_mask[6:10, 14:18] = 255
    cv2.imwrite(str(protected / "000000.png"), object_mask)
    cv2.imwrite(str(robot / "000000.png"), robot_mask)
    output = restore_protected_video(
        source,
        clean,
        protected,
        tmp_path / "excluded.mp4",
        exclude_mask_dir=robot,
        exclude_margin=0,
        feather_radius=0,
    )
    capture = cv2.VideoCapture(str(output))
    ok, frame = capture.read()
    capture.release()
    assert ok
    assert float(np.mean(frame[12:16, 12:20])) > 170
    assert 60 < float(np.mean(frame[7:9, 15:17])) < 100


def test_fragmented_protected_object_is_filled_before_subtraction(tmp_path) -> None:
    video = tmp_path / "source.mp4"
    raw = tmp_path / "raw"
    protected = tmp_path / "protected"
    refined = tmp_path / "refined"
    raw.mkdir()
    protected.mkdir()
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
    writer.write(np.full((24, 32, 3), 80, dtype=np.uint8))
    writer.release()
    removal = np.full((24, 32), 255, dtype=np.uint8)
    fragments = np.zeros_like(removal)
    fragments[6:9, 10:13] = 255
    fragments[14:17, 18:21] = 255
    cv2.imwrite(str(raw / "000000.png"), removal)
    cv2.imwrite(str(protected / "000000.png"), fragments)
    refine_robot_masks(
        video,
        raw,
        refined,
        protected_mask_dir=protected,
        dilation_radius=0,
        closing_radius=0,
        protect_margin=0,
        use_optical_flow=False,
    )
    result = cv2.imread(str(refined / "000000.png"), cv2.IMREAD_GRAYSCALE)
    assert result[11, 15] == 0


def test_refine_masks_uses_future_frame_to_fill_one_frame_gap(tmp_path) -> None:
    video = tmp_path / "source.mp4"
    raw = tmp_path / "raw"
    refined = tmp_path / "refined"
    raw.mkdir()
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (48, 32))
    for index in range(3):
        frame = np.full((32, 48, 3), 70, dtype=np.uint8)
        frame[10:22, 16:32] = 180
        writer.write(frame)
        mask = np.zeros((32, 48), dtype=np.uint8)
        if index != 1:
            mask[10:22, 16:32] = 255
        cv2.imwrite(str(raw / f"{index:06d}.png"), mask)
    writer.release()
    manifest_path = refine_robot_masks(
        video,
        raw,
        refined,
        dilation_radius=0,
        closing_radius=0,
        use_optical_flow=True,
        flow_scale=1.0,
    )
    middle = cv2.imread(str(refined / "000001.png"), cv2.IMREAD_GRAYSCALE)
    assert np.all(middle[11:21, 17:31] == 255)
    manifest = __import__("json").loads(manifest_path.read_text())
    assert manifest["optical_flow_direction"] == "previous+next"
    assert manifest["mean_next_propagated_fraction"] > 0


def test_validate_robot_masks_audits_recall_stability_and_protection(tmp_path) -> None:
    video = tmp_path / "source.mp4"
    raw = tmp_path / "raw"
    masks = tmp_path / "masks"
    protected = tmp_path / "protected"
    raw.mkdir()
    masks.mkdir()
    protected.mkdir()
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
    for index in range(4):
        writer.write(np.full((24, 32, 3), 80, dtype=np.uint8))
        source = np.zeros((24, 32), dtype=np.uint8)
        source[6:18, 7:25] = 255
        keep = np.zeros_like(source)
        keep[10:14, 14:18] = 255
        accepted = source.copy()
        accepted[keep > 0] = 0
        cv2.imwrite(str(raw / f"{index:06d}.png"), source)
        cv2.imwrite(str(masks / f"{index:06d}.png"), accepted)
        cv2.imwrite(str(protected / f"{index:06d}.png"), keep)
    writer.release()
    report = validate_robot_masks(
        video,
        masks,
        source_mask_dir=raw,
        protected_mask_dir=protected,
        flow_scale=1.0,
    )
    assert report["ok"]
    assert report["frames"] == 4
    assert report["mean_source_recall"] == 1.0
    assert report["maximum_protected_overlap"] == 0.0
