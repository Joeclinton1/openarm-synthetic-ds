import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_cross_dataset_benchmark.py"
SPEC = importlib.util.spec_from_file_location("evaluate_cross_dataset_benchmark", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def _write_render(clip: Path, suffix: str, seconds: float, *, stale: bool = False) -> None:
    scene = clip / "render_scene" / f"scene{suffix}.json"
    scene.parent.mkdir(parents=True, exist_ok=True)
    scene.write_text(json.dumps({"episode_frames": 2, "suffix": suffix}))
    manifest = clip / f"render_raw{suffix}" / "render_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    digest = "stale" if stale else hashlib.sha256(scene.read_bytes()).hexdigest()
    manifest.write_text(
        json.dumps({"scene_sha256": digest, "seconds": seconds, "missing_frames": []})
    )


def test_render_runtime_validates_main_scene_hash(tmp_path: Path) -> None:
    _write_render(tmp_path, "", 1.25)
    assert benchmark.render_runtime(tmp_path) == 1.25


def test_render_runtime_rejects_stale_frames(tmp_path: Path) -> None:
    _write_render(tmp_path, "", 1.25, stale=True)
    with pytest.raises(ValueError, match="Stale render"):
        benchmark.render_runtime(tmp_path)


def test_render_runtime_uses_current_bimanual_scenes(tmp_path: Path) -> None:
    _write_render(tmp_path, "_left", 2.0)
    _write_render(tmp_path, "_right", 3.0)
    assert benchmark.render_runtime(tmp_path) == 3.0
