from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .adapters import load_agibot_lerobot_episode, load_lerobot_episode
from .export import export_lerobot_v3
from .filters import filter_episode
from .ik import OpenArmIK
from .registration import auto_register_episode
from .retarget import apply_registration
from .schema import Episode, SourceConfig


BATCH_CONVERSION_VERSION = 5
_WORKER_STATE: dict = {}


def _conversion_signature(config: SourceConfig) -> str:
    payload = {
        "pipeline_version": BATCH_CONVERSION_VERSION,
        "source_config": config.to_json(),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def _load_episode(table: pa.Table, config: SourceConfig, episode_index: int) -> Episode:
    if config.adapter == "agibot_lerobot":
        return load_agibot_lerobot_episode(table, config, episode_index, allow_uncalibrated=True)
    return load_lerobot_episode(table, config, episode_index, allow_uncalibrated=True)


def _initialize_worker(
    config: SourceConfig,
    sample_parquet: str,
    selected_ids: list[int],
    destination: str,
    model_path: str | None,
    resume: bool,
    conversion_signature: str,
    selected_count: int,
) -> None:
    table = pq.read_table(sample_parquet)
    table = table.filter(np.isin(np.asarray(table["episode_index"]), selected_ids))
    _WORKER_STATE.update(
        {
            "config": config,
            "table": table,
            "episode_dir": Path(destination) / "episodes",
            "model_path": model_path,
            "resume": resume,
            "conversion_signature": conversion_signature,
            "selected_count": selected_count,
            "solver": OpenArmIK(model_path),
        }
    )


def _process_selected_episode(item: tuple[int, dict]) -> tuple[Path, dict, str]:
    number, selected_episode = item
    config = _WORKER_STATE["config"]
    table = _WORKER_STATE["table"]
    episode_dir = _WORKER_STATE["episode_dir"]
    model_path = _WORKER_STATE["model_path"]
    resume = _WORKER_STATE["resume"]
    conversion_signature = _WORKER_STATE["conversion_signature"]
    solver = _WORKER_STATE["solver"]
    episode_index = int(selected_episode["episode_index"])
    output = episode_dir / f"episode_{episode_index:06d}.npz"
    if output.exists() and resume:
        solved = Episode.load(output)
        if (
            solved.feasible is None
            or not np.any(solved.feasible)
            or solved.metadata.get("conversion_signature") != conversion_signature
        ):
            output.unlink()
            solved = None
    else:
        solved = None
    if solved is None:
        # Registration must own both source-to-OpenArm transforms so they are applied once.
        raw_config = replace(
            config,
            openarm_from_source_base=None,
            source_tool_from_openarm_tool=None,
            position_scale=1.0,
            preserve_pinch_center=False,
        )
        raw = _load_episode(table, raw_config, episode_index)
        registration = auto_register_episode(
            raw,
            model_path,
            source_tool_from_openarm_tool=config.source_tool_from_openarm_tool,
            openarm_from_source_base=config.openarm_from_source_base,
            position_scale=(
                config.position_scale if config.openarm_from_source_base is not None else None
            ),
        )
        solved = apply_registration(raw, config, registration)
        solver.solve_episode(solved)
        filter_episode(solved, solver)
        solved.metadata["conversion_signature"] = conversion_signature
        solved.save(output)
    feasible = int(solved.feasible.sum()) if solved.feasible is not None else 0
    quality = {
        "episode_index": episode_index,
        "frames": len(solved.timestamp),
        "seconds": solved.duration,
        "feasible_frames": feasible,
        "feasible_fraction": feasible / max(len(solved.timestamp), 1),
        "output": str(output),
    }
    message = (
        f"[{number}/{_WORKER_STATE['selected_count']}] episode {episode_index}: "
        f"{feasible}/{len(solved.timestamp)} feasible"
    )
    return output, quality, message


def convert_lerobot_hour(
    config: SourceConfig,
    sample_manifest: str | Path,
    sample_parquet: str | Path,
    destination: str | Path,
    model_path: str | Path | None = None,
    max_episodes: int | None = None,
    resume: bool = True,
    progress: Callable[[str], None] | None = None,
    workers: int = 1,
) -> Path:
    """Retarget every selected LeRobot episode with resumable per-episode checkpoints."""
    destination = Path(destination)
    episode_dir = destination / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(sample_manifest).read_text())
    selected = manifest["episodes"][:max_episodes]
    selected_ids = [int(value["episode_index"]) for value in selected]
    if workers < 1:
        raise ValueError("workers must be at least one")
    conversion_signature = _conversion_signature(config)
    work = list(enumerate(selected, start=1))
    initializer_args = (
        config,
        str(sample_parquet),
        selected_ids,
        str(destination),
        str(model_path) if model_path is not None else None,
        resume,
        conversion_signature,
        len(selected),
    )
    if workers == 1:
        _initialize_worker(*initializer_args)
        results = map(_process_selected_episode, work)
    else:
        executor = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_worker,
            initargs=initializer_args,
        )
        results = executor.map(_process_selected_episode, work)
    outputs: list[Path] = []
    quality: list[dict] = []
    try:
        for output, episode_quality, message in results:
            outputs.append(output)
            quality.append(episode_quality)
            if progress:
                progress(message)
    finally:
        if workers != 1:
            executor.shutdown(wait=True, cancel_futures=True)
    episodes = [Episode.load(path) for path in outputs]
    export_path = destination / "lerobot"
    export_lerobot_v3(
        episodes,
        export_path,
        fps=int(config.fps),
        feasible_only=True,
        allow_uncalibrated=True,
    )
    report = {
        "source_manifest": str(Path(sample_manifest).resolve()),
        "source_revision": manifest["revision"],
        "source_repo_id": manifest["repo_id"],
        "registration": "per-episode shared-base-frame kinematic registration",
        "shared_base_frame": True,
        "calibration_validated": False,
        "release_status": "inspection-grade until physical base/tool calibration is validated",
        "episodes_requested": len(selected),
        "input_frames": sum(value["frames"] for value in quality),
        "input_seconds": sum(value["seconds"] for value in quality),
        "feasible_frames": sum(value["feasible_frames"] for value in quality),
        "episodes": quality,
    }
    report["feasible_fraction"] = report["feasible_frames"] / max(report["input_frames"], 1)
    (destination / "quality_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return destination / "quality_report.json"
