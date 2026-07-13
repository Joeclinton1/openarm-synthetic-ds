from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np


GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-tiny"
SAM2_MODEL = "facebook/sam2.1-hiera-small"


def combine_object_masks(masks: np.ndarray) -> np.ndarray:
    """Union every batch/object/channel axis while preserving image height and width."""
    array = np.asarray(masks)
    if array.ndim < 2:
        raise ValueError("Masks must have at least height and width dimensions")
    axes = tuple(range(array.ndim - 2))
    return np.any(array, axis=axes) if axes else array.astype(bool)


def select_robot_boxes(
    detections: Iterable[dict[str, Any]],
    width: int,
    height: int,
    expected_arms: int = 2,
    expansion_fraction: float = 0.15,
    include_edge_components: bool = False,
    allow_partial: bool = False,
) -> list[list[float]]:
    """Select compact, lower-image robot detections and reject scene-sized false positives."""
    if expected_arms not in (1, 2):
        raise ValueError("expected_arms must be one or two")
    candidates: list[tuple[float, list[float]]] = []
    image_area = width * height
    for detection in detections:
        box = detection.get("box", {})
        try:
            x1 = float(box["xmin"])
            y1 = float(box["ymin"])
            x2 = float(box["xmax"])
            y2 = float(box["ymax"])
        except (KeyError, TypeError, ValueError):
            continue
        box_width = max(0.0, x2 - x1)
        box_height = max(0.0, y2 - y1)
        area_fraction = box_width * box_height / image_area
        if not 0.01 <= area_fraction <= 0.35:
            continue
        score = float(detection.get("score", 0.0))
        candidates.append((score, [x1, y1, x2, y2]))

    def expand(box: list[float]) -> list[float]:
        x1, y1, x2, y2 = box
        dx = (x2 - x1) * expansion_fraction
        dy = (y2 - y1) * expansion_fraction
        return [max(0.0, x1 - dx), max(0.0, y1 - dy), min(width, x2 + dx), min(height, y2 + dy)]

    if expected_arms == 1:
        if not candidates:
            raise RuntimeError(
                "No plausible robot-arm detection; lower the threshold or supply masks"
            )
        return [expand(max(candidates, key=lambda item: item[0])[1])]

    # A bimanual ego view should have one detection on each side. Requiring the boxes not to
    # cross the image centre eliminates Grounding DINO's common whole-lower-frame false box.
    left = [item for item in candidates if item[1][2] <= 0.52 * width]
    right = [item for item in candidates if item[1][0] >= 0.48 * width]
    if (not left or not right) and not allow_partial:
        raise RuntimeError("Could not find separated left and right robot-arm detections")
    selected = []
    selected_sides = []
    for side_name, side_candidates in (("left", left), ("right", right)):
        if side_candidates:
            selected.append(max(side_candidates, key=lambda item: item[0]))
            selected_sides.append(side_name)
    if not selected:
        raise RuntimeError("No plausible robot-arm detections in this video chunk")
    if include_edge_components:
        edge_groups = [
            [
                item
                for item in candidates
                if item[1][0] <= 0.05 * width and item[1][2] <= 0.25 * width
            ],
            [
                item
                for item in candidates
                if item[1][0] >= 0.75 * width and item[1][2] >= 0.95 * width
            ],
        ]
        edge_by_side = dict(zip(("left", "right"), edge_groups, strict=True))
        mains = selected.copy()
        for side_name, main in zip(selected_sides, mains, strict=True):
            edge_candidates = edge_by_side[side_name]
            if edge_candidates:
                edge = max(edge_candidates, key=lambda item: item[0])
                if edge[1] != main[1]:
                    selected.append(edge)
    return [expand(item[1]) for item in selected]


def select_prompt_boxes(
    detections: Iterable[dict[str, Any]],
    width: int,
    height: int,
    *,
    max_objects: int = 1,
    expansion_fraction: float = 0.08,
) -> list[list[float]]:
    """Select the highest-confidence compact objects for contact preservation."""
    if max_objects < 1:
        raise ValueError("max_objects must be positive")
    candidates: list[tuple[float, list[float]]] = []
    image_area = width * height
    for detection in detections:
        box = detection.get("box", {})
        try:
            x1, y1 = float(box["xmin"]), float(box["ymin"])
            x2, y2 = float(box["xmax"]), float(box["ymax"])
        except (KeyError, TypeError, ValueError):
            continue
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1) / image_area
        # Prompts frequently produce a higher-scoring union of the object and grasping arm.
        # Contact protection needs the object itself, so reject boxes covering large scene regions.
        if 0.001 <= area <= 0.15:
            candidates.append((float(detection.get("score", 0.0)), [x1, y1, x2, y2]))
    if not candidates:
        raise RuntimeError("No plausible prompted-object detection")
    selected = sorted(candidates, reverse=True)[:max_objects]
    result = []
    for _, (x1, y1, x2, y2) in selected:
        dx, dy = (x2 - x1) * expansion_fraction, (y2 - y1) * expansion_fraction
        result.append(
            [
                max(0.0, x1 - dx),
                max(0.0, y1 - dy),
                min(float(width), x2 + dx),
                min(float(height), y2 + dy),
            ]
        )
    return result


def carry_object_box(
    mask: np.ndarray,
    width: int,
    height: int,
    *,
    minimum_size: tuple[float, float],
    expansion_fraction: float = 0.18,
) -> list[float] | None:
    """Build a stable next-chunk prompt without collapsing onto transparent fragments."""
    binary = np.asarray(mask, dtype=bool)
    if not np.any(binary):
        return None
    ys, xs = np.where(binary)
    observed_width = float(xs.max() + 1 - xs.min())
    observed_height = float(ys.max() + 1 - ys.min())
    box_width = max(observed_width, minimum_size[0]) * (1.0 + 2.0 * expansion_fraction)
    box_height = max(observed_height, minimum_size[1]) * (1.0 + 2.0 * expansion_fraction)
    center_x = float(np.median(xs))
    center_y = float(np.median(ys))
    box_width = min(float(width), box_width)
    box_height = min(float(height), box_height)
    x1 = min(max(0.0, center_x - box_width / 2.0), float(width) - box_width)
    y1 = min(max(0.0, center_y - box_height / 2.0), float(height) - box_height)
    return [x1, y1, x1 + box_width, y1 + box_height]


def _read_video_chunks(
    video_path: str | Path,
    width: int,
    height: int,
    chunk_frames: int,
    max_frames: int | None,
):
    """Decode RGB with software FFmpeg; OpenCV may silently fail on AV1 containers."""
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdout is None:
        raise RuntimeError("Could not open FFmpeg output pipe")
    frame_bytes = width * height * 3
    count = 0
    try:
        while max_frames is None or count < max_frames:
            chunk: list[np.ndarray] = []
            while len(chunk) < chunk_frames and (max_frames is None or count < max_frames):
                buffer = bytearray()
                while len(buffer) < frame_bytes:
                    block = process.stdout.read(frame_bytes - len(buffer))
                    if not block:
                        break
                    buffer.extend(block)
                if not buffer:
                    break
                if len(buffer) != frame_bytes:
                    raise RuntimeError("FFmpeg returned a partial RGB frame")
                chunk.append(np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 3).copy())
                count += 1
            if not chunk:
                break
            yield chunk
    finally:
        if process.poll() is None:
            process.terminate()
        process.communicate()


def segment_robot_video(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    prompt: str = "robotic arm",
    expected_arms: int = 2,
    threshold: float = 0.12,
    box_expansion: float = 0.15,
    include_edge_components: bool = False,
    chunk_frames: int = 300,
    max_frames: int | None = None,
    device: int = 0,
    grounding_model: str = GROUNDING_DINO_MODEL,
    sam_model: str = SAM2_MODEL,
    selection: str = "robot",
    max_objects: int = 1,
    seed_box: list[float] | None = None,
) -> Path:
    """Detect robot arms and track their masks with SAM2 in bounded-memory chunks.

    The heavyweight dependencies are imported lazily so the kinematic converter remains usable
    without the ``media-ai`` extra. Each chunk is independently re-prompted, limiting temporal
    drift and memory use for long demonstrations.
    """
    if chunk_frames < 2:
        raise ValueError("chunk_frames must be at least two")
    try:
        import torch
        from PIL import Image
        from transformers import Sam2VideoModel, Sam2VideoProcessor, pipeline
    except ImportError as error:
        raise RuntimeError("Install the media models with: uv sync --extra media-ai") from error
    if not torch.cuda.is_available():
        raise RuntimeError("SAM2 video segmentation requires a CUDA GPU in this pipeline")

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = pipeline("zero-shot-object-detection", model=grounding_model, device=device)
    torch_device = torch.device(f"cuda:{device}")
    sam = Sam2VideoModel.from_pretrained(sam_model).to(device=torch_device, dtype=torch.bfloat16)
    processor = Sam2VideoProcessor.from_pretrained(sam_model)
    chunks: list[dict[str, Any]] = []
    global_index = 0
    carry_box = seed_box
    initial_object_extent: tuple[float, float] | None = None
    capture.release()
    try:
        for chunk_index, frames in enumerate(
            _read_video_chunks(video_path, width, height, chunk_frames, max_frames)
        ):
            pil_frames = [Image.fromarray(frame) for frame in frames]
            detections = (
                detector(pil_frames[0], candidate_labels=[prompt], threshold=threshold)
                if carry_box is None
                else []
            )
            if selection == "object" and carry_box is not None:
                boxes = [carry_box]
            elif selection == "robot":
                boxes = select_robot_boxes(
                    detections,
                    width,
                    height,
                    expected_arms,
                    box_expansion,
                    include_edge_components,
                    True,
                )
            elif selection == "object":
                boxes = select_prompt_boxes(
                    detections,
                    width,
                    height,
                    max_objects=max_objects,
                    expansion_fraction=box_expansion,
                )
            else:
                raise ValueError("selection must be 'robot' or 'object'")
            if selection == "object" and initial_object_extent is None:
                initial_object_extent = (
                    max(box[2] - box[0] for box in boxes),
                    max(box[3] - box[1] for box in boxes),
                )
            session = processor.init_video_session(
                video=pil_frames,
                inference_device=torch_device,
                dtype=torch.bfloat16,
            )
            object_ids = list(range(1, len(boxes) + 1))
            processor.add_inputs_to_inference_session(
                inference_session=session,
                frame_idx=0,
                obj_ids=object_ids,
                input_boxes=[boxes],
            )
            sam(inference_session=session, frame_idx=0)
            written = 0
            last_combined: np.ndarray | None = None
            for output in sam.propagate_in_video_iterator(session):
                masks = processor.post_process_masks(
                    [output.pred_masks],
                    original_sizes=[[session.video_height, session.video_width]],
                    binarize=True,
                )[0]
                array = masks.detach().to("cpu").numpy()
                combined = combine_object_masks(array)
                last_combined = combined
                cv2.imwrite(
                    str(output_dir / f"{global_index:06d}.png"),
                    combined.astype(np.uint8) * 255,
                )
                global_index += 1
                written += 1
            if written != len(frames):
                raise RuntimeError(f"SAM2 returned {written} masks for a {len(frames)}-frame chunk")
            if selection == "object" and last_combined is not None and np.any(last_combined):
                assert initial_object_extent is not None
                carry_box = carry_object_box(
                    last_combined,
                    width,
                    height,
                    minimum_size=initial_object_extent,
                )
            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "start_frame": global_index - written,
                    "frames": written,
                    "boxes_xyxy": boxes,
                }
            )
            del session, pil_frames
            torch.cuda.empty_cache()
    finally:
        pass

    if global_index == 0:
        raise RuntimeError("Video decoder returned zero frames")

    manifest = {
        "source_video": str(video_path.resolve()),
        "output_format": "six-digit PNG, 0 background, 255 robot",
        "frames": global_index,
        "fps": fps,
        "resolution": [width, height],
        "prompt": prompt,
        "selection": selection,
        "max_objects": max_objects,
        "seed_box_xyxy": seed_box,
        "carry_object_across_chunks": selection == "object",
        "expected_arms": expected_arms,
        "threshold": threshold,
        "box_expansion": box_expansion,
        "include_edge_components": include_edge_components,
        "chunk_frames": chunk_frames,
        "models": {"detector": grounding_model, "tracker": sam_model},
        "chunks": chunks,
    }
    manifest_path = output_dir / "segmentation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path
