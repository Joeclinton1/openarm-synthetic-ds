from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .poses import pose_to_matrix
from .schema import Episode


def write_agibot_openarm_camera(
    episode_path: str | Path,
    intrinsic_path: str | Path,
    aligned_extrinsic_path: str | Path,
    output: str | Path,
    video_path: str | Path | None = None,
    calibration_width: int | None = None,
    calibration_height: int | None = None,
) -> Path:
    """Map AgiBot's aligned camera trajectory through the shared OpenArm registration."""
    episode = Episode.load(episode_path)
    registration = episode.metadata.get("registration")
    if not registration or not registration.get("shared_base_frame"):
        raise ValueError("Episode requires a shared-frame registration")
    bases = registration["openarm_from_source_base"]
    if not np.allclose(bases["right"], bases["left"], atol=1e-10):
        raise ValueError("Right and left arms do not use the same OpenArm base transform")
    openarm_from_source = pose_to_matrix(np.asarray(bases["right"], dtype=np.float64))
    scale = float(registration["position_scale"])

    intrinsic_payload = json.loads(Path(intrinsic_path).read_text())["intrinsic"]
    scale_x = scale_y = 1.0
    output_resolution = None
    if video_path is not None:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Could not open camera video: {video_path}")
        output_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        output_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        output_resolution = [output_width, output_height]
        if (calibration_width is None) != (calibration_height is None):
            raise ValueError("Provide both calibration width and height")
        if calibration_width is not None and calibration_height is not None:
            scale_x = output_width / calibration_width
            scale_y = output_height / calibration_height
    intrinsics = [
        [float(intrinsic_payload["fx"]) * scale_x, 0.0, float(intrinsic_payload["ppx"]) * scale_x],
        [0.0, float(intrinsic_payload["fy"]) * scale_y, float(intrinsic_payload["ppy"]) * scale_y],
        [0.0, 0.0, 1.0],
    ]
    distortion = [
        float(intrinsic_payload.get(name, 0.0)) for name in ("k1", "k2", "p1", "p2", "k3")
    ]
    extrinsics = json.loads(Path(aligned_extrinsic_path).read_text())
    if isinstance(extrinsics, dict):
        extrinsics = [extrinsics]
    if len(extrinsics) not in (1, len(episode.timestamp)):
        raise ValueError(
            f"Camera trajectory has {len(extrinsics)} poses for {len(episode.timestamp)} frames"
        )
    frames = []
    for item in extrinsics:
        values = item["extrinsic"]
        source_world_from_camera = np.eye(4)
        source_world_from_camera[:3, :3] = np.asarray(values["rotation_matrix"])
        source_world_from_camera[:3, 3] = np.asarray(values["translation_vector"])
        transformed = np.eye(4)
        transformed[:3, :3] = openarm_from_source[:3, :3] @ source_world_from_camera[:3, :3]
        transformed[:3, 3] = (
            openarm_from_source[:3, 3]
            + scale * openarm_from_source[:3, :3] @ source_world_from_camera[:3, 3]
        )
        frames.append(transformed.tolist())
    payload = {
        "intrinsics": intrinsics,
        "distortion": distortion,
        "distortion_model": intrinsic_payload.get("distortion_model", "plumb bob"),
        "output_resolution": output_resolution,
        "intrinsic_scale_xy": [scale_x, scale_y],
        "calibration_resolution": (
            [calibration_width, calibration_height] if calibration_width is not None else None
        ),
        "world_from_camera": frames[0],
        "world_from_camera_frames": frames,
        "convention": "OpenCV camera: +x right, +y down, +z forward",
        "source_extrinsic_convention": (
            "AgiBot aligned extrinsics interpreted as source_world_from_camera"
        ),
        "registration_validated": bool(registration.get("validated", False)),
        "warning": (
            "Camera calibration is source-provided; the automatic source-to-OpenArm "
            "registration remains kinematic and unvalidated"
        ),
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
