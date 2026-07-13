from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def fit_rigid_transform(
    source_points: np.ndarray, target_points: np.ndarray
) -> tuple[np.ndarray, float]:
    """Fit target_from_source with the proper-rotation Kabsch solution."""
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source_points and target_points must both have shape [N, 3]")
    if len(source) < 3 or np.linalg.matrix_rank(source - source.mean(axis=0)) < 2:
        raise ValueError("At least three non-collinear point correspondences are required")
    source_centre = source.mean(axis=0)
    target_centre = target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_centre).T @ (target - target_centre))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_centre - rotation @ source_centre
    predicted = (rotation @ source.T).T + translation
    rms = float(np.sqrt(np.mean(np.sum((predicted - target) ** 2, axis=1))))
    pose = np.r_[translation, Rotation.from_matrix(rotation).as_quat()]
    return pose, rms


def calibrate_from_file(input_path: str | Path, output_path: str | Path) -> dict:
    values = json.loads(Path(input_path).read_text())
    pose, rms = fit_rigid_transform(values["source_points"], values["openarm_points"])
    maximum_rms = float(values.get("maximum_rms_m", 0.005))
    result = {
        "method": "corresponding_points_kabsch",
        "openarm_from_source_base": pose.tolist(),
        "source_tool_from_openarm_tool": values.get(
            "source_tool_from_openarm_tool", [0, 0, 0, 0, 0, 0, 1]
        ),
        "rms_m": rms,
        "maximum_rms_m": maximum_rms,
        "validated": bool(rms <= maximum_rms and values.get("tool_offset_validated", False)),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result
