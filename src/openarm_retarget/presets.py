from __future__ import annotations

from pathlib import Path

import numpy as np

from .constants import SIDES
from .ik import OpenArmIK
from .schema import Episode


def fit_workspace_translation(
    episode: Episode,
    model_path: str | Path | None = None,
    sides: tuple[str, ...] = SIDES,
) -> dict[str, list[float] | str]:
    """Fit translation only, preserving documented source axes and metric scale.

    This is an inspection bootstrap, not a metrological calibration. A real conversion must
    validate it using corresponding physical points or robot CAD transforms.
    """
    ik = OpenArmIK(model_path)
    indices = [SIDES.index(side) for side in sides]
    source_centres = np.nanmedian(episode.ee_pose[:, indices, :3], axis=0)
    target_centres = np.stack([ik.forward_pose(side, ik.neutral(side))[:3] for side in sides])
    translation = np.mean(target_centres - source_centres, axis=0)
    return {
        "method": "median_workspace_translation_unvalidated",
        "openarm_from_source_base": [*translation.tolist(), 0.0, 0.0, 0.0, 1.0],
        "source_tool_from_openarm_tool": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    }
