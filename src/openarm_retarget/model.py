from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .constants import OPENARM_MODEL_RELATIVE, OPENARM_MUJOCO_COMMIT, OPENARM_MUJOCO_REPO


def fetch_openarm_model(destination: str | Path = "data/assets/openarm_mujoco") -> Path:
    """Fetch the exact audited OpenArm MuJoCo revision without modifying source files."""
    destination = Path(destination).resolve()
    model_path = destination / OPENARM_MODEL_RELATIVE
    if model_path.exists():
        return model_path
    if not shutil.which("git"):
        raise RuntimeError("git is required to fetch the official OpenArm model")
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            OPENARM_MUJOCO_REPO,
            str(destination),
        ],
        check=True,
    )
    subprocess.run(["git", "checkout", OPENARM_MUJOCO_COMMIT], cwd=destination, check=True)
    if not model_path.exists():
        raise RuntimeError(f"OpenArm model missing after checkout: {model_path}")
    return model_path


def resolve_model(path: str | Path | None = None) -> Path:
    if path:
        result = Path(path).resolve()
        if not result.exists():
            raise FileNotFoundError(result)
        return result
    default = Path("data/assets/openarm_mujoco") / OPENARM_MODEL_RELATIVE
    return default.resolve() if default.exists() else fetch_openarm_model()
