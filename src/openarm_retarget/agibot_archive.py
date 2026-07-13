from __future__ import annotations

import json
import os
import re
import tarfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
from huggingface_hub import HfApi, hf_hub_download

from .adapters import load_agibot_h5
from .download import _file_record
from .export import export_lerobot_v3
from .filters import filter_episode
from .ik import OpenArmIK
from .registration import auto_register_episode
from .retarget import apply_registration
from .schema import Episode, SourceConfig

AGIBOT_REPO = "agibot-world/AgiBotWorld-Alpha"


def _episode_frames(annotation: dict) -> int:
    labels = annotation.get("label_info") or annotation.get("lable_info") or {}
    actions = labels.get("action_config") or []
    return max((int(action["end_frame"]) for action in actions), default=0)


def _archive_range(path: str) -> tuple[int, int] | None:
    match = re.search(r"/(\d+)-(\d+)\.tar$", path)
    return (int(match.group(1)), int(match.group(2))) if match else None


def plan_agibot_hour(
    task_json: str | Path,
    destination: str | Path,
    task_id: int,
    seconds: float = 3600,
    token: str | None = None,
) -> Path:
    annotations = json.loads(Path(task_json).read_text())
    elapsed = 0.0
    selected = []
    for annotation in annotations:
        frames = _episode_frames(annotation)
        if frames <= 0:
            continue
        selected.append(
            {
                "episode_index": int(annotation["episode_id"]),
                "task_id": task_id,
                "task": annotation.get("task_name", f"task_{task_id}"),
                "frames": frames,
                "seconds": frames / 30,
                "action_config": (annotation.get("label_info") or {}).get("action_config", []),
            }
        )
        elapsed += frames / 30
        if elapsed >= seconds:
            break
    if elapsed < seconds:
        raise RuntimeError(f"Task {task_id} has only {elapsed:.1f}s")
    ids = {episode["episode_index"] for episode in selected}
    api = HfApi(token=token or os.environ.get("HF_TOKEN"))
    files = api.list_repo_files(AGIBOT_REPO, repo_type="dataset")
    archives = []
    for filename in files:
        archive_range = _archive_range(filename)
        if filename.startswith(f"observations/{task_id}/") and archive_range:
            if any(archive_range[0] <= episode <= archive_range[1] for episode in ids):
                archives.append(filename)
        elif filename.startswith("parameters/") and archive_range:
            if any(archive_range[0] <= episode <= archive_range[1] for episode in ids):
                archives.append(filename)
        elif filename.startswith("proprio_stats/") and filename.endswith(".tar"):
            archives.append(filename)
    info = api.dataset_info(AGIBOT_REPO)
    manifest = {
        "repo_id": AGIBOT_REPO,
        "revision": info.sha,
        "license": "CC-BY-NC-SA-4.0",
        "task_id": task_id,
        "fps": 30,
        "requested_seconds": seconds,
        "selected_seconds": elapsed,
        "selected_frames": sum(episode["frames"] for episode in selected),
        "episodes": selected,
        "archives": sorted(set(archives)),
    }
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "sample_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    return output


def download_agibot_archives(manifest_path: str | Path, token: str | None = None) -> Path:
    path = Path(manifest_path)
    root = path.parent
    manifest = json.loads(path.read_text())
    for filename in manifest["archives"]:
        hf_hub_download(
            AGIBOT_REPO,
            repo_type="dataset",
            filename=filename,
            revision=manifest["revision"],
            local_dir=root,
            token=token or os.environ.get("HF_TOKEN"),
        )
    manifest["files"] = {
        filename: _file_record(root / filename) for filename in manifest["archives"]
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def _safe_selected_members(
    archive: tarfile.TarFile,
    prefixes: tuple[str, ...],
    strip_component: str | None = None,
):
    for member in archive:
        normalized = member.name.removeprefix("./")
        if not normalized.startswith(prefixes):
            continue
        if (
            member.issym()
            or member.islnk()
            or not (member.isfile() or member.isdir())
            or ".." in Path(normalized).parts
        ):
            raise ValueError(f"Unsafe tar member: {member.name}")
        if strip_component is not None:
            parts = list(Path(normalized).parts)
            try:
                parts.remove(strip_component)
            except ValueError:
                pass
            normalized = str(Path(*parts))
        member.name = normalized
        yield member


def extract_agibot_hour(manifest_path: str | Path) -> Path:
    path = Path(manifest_path)
    root = path.parent
    manifest = json.loads(path.read_text())
    task = int(manifest["task_id"])
    ids = [int(episode["episode_index"]) for episode in manifest["episodes"]]
    sample = root / "sample"
    for filename in manifest["archives"]:
        family = Path(filename).parts[0]
        if family == "observations":
            # Observation range archives are rooted directly at episode_id/.
            prefixes = tuple(f"{episode}/" for episode in ids)
            destination = sample / "observations" / str(task)
            strip_component = None
        elif family == "proprio_stats":
            # Proprioception archives are rooted at task_id/episode_id/.
            prefixes = tuple(f"{task}/{episode}/" for episode in ids)
            destination = sample / "proprio_stats"
            strip_component = None
        elif family == "parameters":
            # Parameter archives use task/episode/parameters/* while the released sample
            # dataset uses parameters/task/episode/*.
            prefixes = tuple(f"{task}/{episode}/" for episode in ids)
            destination = sample / "parameters"
            strip_component = "parameters"
        else:
            raise ValueError(f"Unknown AgiBot archive family: {filename}")
        destination.mkdir(parents=True, exist_ok=True)
        with tarfile.open(root / filename, "r:") as archive:
            archive.extractall(
                destination,
                members=_safe_selected_members(archive, prefixes, strip_component),
            )
    missing = [
        episode
        for episode in ids
        if not (sample / f"proprio_stats/{task}/{episode}/proprio_stats.h5").is_file()
    ]
    if missing:
        raise RuntimeError(f"Missing extracted proprioception for {len(missing)} episodes")
    manifest["extracted_root"] = "sample"
    manifest["downloaded_episode_ids"] = ids
    manifest["downloaded_episode_count"] = len(ids)
    selected: list[dict[str, Any]] = []
    elapsed = 0.0
    for episode in manifest["episodes"]:
        item = dict(episode)
        h5_path = sample / (f"proprio_stats/{task}/{int(item['episode_index'])}/proprio_stats.h5")
        with h5py.File(h5_path, "r") as h5:
            timestamps = h5["timestamp"]
            frames = len(timestamps)
            if frames > 1:
                differences = np.diff(np.asarray(timestamps, dtype=np.int64))
                sample_period = float(np.median(differences)) / 1e9
                duration = float(timestamps[-1] - timestamps[0]) / 1e9 + sample_period
            else:
                duration = 0.0
        item["annotation_frames"] = int(item["frames"])
        item["frames"] = frames
        item["seconds"] = duration
        selected.append(item)
        elapsed += duration
        if elapsed >= float(manifest["requested_seconds"]):
            break
    manifest["episodes"] = selected
    manifest["selected_frames"] = sum(episode["frames"] for episode in selected)
    manifest["selected_seconds"] = elapsed
    manifest["sample_rows"] = manifest["selected_frames"]
    manifest["extracted_episodes"] = len(selected)
    manifest["frame_count_basis"] = (
        "HDF5 timestamp rows; duration is timestamp span plus median final sample interval"
    )
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return sample


def convert_agibot_hour(
    config: SourceConfig,
    manifest_path: str | Path,
    destination: str | Path,
    model_path: str | Path | None = None,
    max_episodes: int | None = None,
    resume: bool = True,
    progress: Callable[[str], None] | None = None,
) -> Path:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    source_root = manifest_path.parent / manifest["extracted_root"]
    destination = Path(destination)
    episode_dir = destination / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)
    selected = manifest["episodes"][:max_episodes]
    solver = OpenArmIK(model_path)
    outputs = []
    quality: list[dict[str, Any]] = []
    for number, item in enumerate(selected, start=1):
        episode_index = int(item["episode_index"])
        output = episode_dir / f"episode_{episode_index:06d}.npz"
        if output.exists() and resume:
            solved = Episode.load(output)
            if solved.feasible is None or not np.any(solved.feasible):
                output.unlink()
                solved = None
        else:
            solved = None
        if solved is None:
            h5 = source_root / (
                f"proprio_stats/{manifest['task_id']}/{episode_index}/proprio_stats.h5"
            )
            # Registration must receive flange poses in the untouched source
            # frame. Otherwise the configured base/tool priors are baked in at
            # load time and then applied again by apply_registration.
            raw_config = replace(
                config,
                openarm_from_source_base=None,
                source_tool_from_openarm_tool=None,
                preserve_pinch_center=False,
            )
            raw = load_agibot_h5(h5, raw_config, allow_uncalibrated=True)
            raw.task = item["task"]
            raw.source_episode = str(episode_index)
            registration = auto_register_episode(
                raw,
                model_path,
                source_tool_from_openarm_tool=config.source_tool_from_openarm_tool,
            )
            solved = apply_registration(raw, config, registration)
            solver.solve_episode(solved)
            filter_episode(solved, solver)
            solved.save(output)
        feasible = int(solved.feasible.sum()) if solved.feasible is not None else 0
        quality.append(
            {
                "episode_index": episode_index,
                "frames": len(solved.timestamp),
                "seconds": solved.duration,
                "feasible_frames": feasible,
                "feasible_fraction": feasible / max(len(solved.timestamp), 1),
            }
        )
        outputs.append(output)
        if progress:
            progress(
                f"[{number}/{len(selected)}] episode {episode_index}: "
                f"{feasible}/{len(solved.timestamp)} feasible"
            )
    episodes = [Episode.load(output) for output in outputs]
    export_lerobot_v3(
        episodes,
        destination / "lerobot",
        fps=30,
        feasible_only=True,
        allow_uncalibrated=True,
    )
    report = {
        "source_manifest": str(manifest_path.resolve()),
        "source_revision": manifest["revision"],
        "source_repo_id": manifest["repo_id"],
        "task_id": manifest["task_id"],
        "registration": "per-episode shared-base-frame kinematic registration",
        "shared_base_frame": True,
        "calibration_validated": False,
        "release_status": "inspection-grade until physical base/tool calibration is validated",
        "episodes_requested": len(selected),
        "input_frames": sum(item["frames"] for item in quality),
        "input_seconds": sum(item["seconds"] for item in quality),
        "feasible_frames": sum(item["feasible_frames"] for item in quality),
        "episodes": quality,
    }
    report["feasible_fraction"] = report["feasible_frames"] / max(report["input_frames"], 1)
    output = destination / "quality_report.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    return output
