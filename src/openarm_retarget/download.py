from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError


@dataclass(frozen=True)
class SelectedEpisode:
    episode_index: int
    tasks: list[str]
    frames: int
    seconds: float
    data_file: str
    video_files: list[str]
    video_segments: list[dict[str, Any]]


def _file_record(path: Path) -> dict[str, int | str]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _download(repo_id: str, filename: str, root: Path, token: str | None) -> Path:
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            local_dir=root,
            token=token,
        )
    )


def _metadata_files(repo_id: str, token: str | None, prefix: str = "") -> list[str]:
    files = HfApi(token=token).list_repo_files(repo_id, repo_type="dataset", token=token)
    root = f"{prefix.strip('/')}/" if prefix else ""
    exact = {
        f"{root}README.md",
        f"{root}meta/info.json",
        f"{root}meta/stats.json",
        f"{root}meta/tasks.parquet",
        f"{root}meta/tasks_annotated.parquet",
    }
    return [name for name in files if name in exact or name.startswith(f"{root}meta/episodes/")]


def plan_lerobot_hour(
    repo_id: str,
    destination: str | Path,
    seconds: float = 3600.0,
    task_keywords: list[str] | None = None,
    token: str | None = None,
    prefix: str = "",
) -> tuple[dict[str, Any], list[SelectedEpisode]]:
    """Create a deterministic whole-episode plan totaling at least the requested duration."""
    root = Path(destination) / repo_id.replace("/", "__")
    root.mkdir(parents=True, exist_ok=True)
    token = token or os.environ.get("HF_TOKEN")
    prefix = prefix.strip("/")
    local_base = root / prefix if prefix else root

    def remote(name: str) -> str:
        return f"{prefix}/{name}" if prefix else name

    for filename in _metadata_files(repo_id, token, prefix):
        _download(repo_id, filename, root, token)
    info = json.loads((local_base / "meta/info.json").read_text())
    fps = float(info["fps"])
    episode_tables = [
        pq.read_table(path) for path in sorted((local_base / "meta/episodes").rglob("*.parquet"))
    ]
    episodes = pa.concat_tables(episode_tables, promote_options="default")
    rows = episodes.select(
        [
            name
            for name in episodes.column_names
            if name in {"episode_index", "tasks", "length", "data/chunk_index", "data/file_index"}
            or name.startswith("videos/")
        ]
    ).to_pylist()
    keywords = [value.lower() for value in (task_keywords or [])]
    if keywords:
        matching = [
            row
            for row in rows
            if any(
                keyword in task.lower() for keyword in keywords for task in (row.get("tasks") or [])
            )
        ]
        rows = matching or rows
    selected: list[SelectedEpisode] = []
    elapsed = 0.0
    video_keys = [
        key for key, spec in info.get("features", {}).items() if spec.get("dtype") == "video"
    ]
    for row in rows:
        data_file = remote(
            info["data_path"].format(
                chunk_index=int(row["data/chunk_index"]), file_index=int(row["data/file_index"])
            )
        )
        video_files: list[str] = []
        video_segments: list[dict[str, Any]] = []
        for key in video_keys:
            chunk_key = f"videos/{key}/chunk_index"
            file_key = f"videos/{key}/file_index"
            if row.get(chunk_key) is not None:
                video_file = remote(
                    info["video_path"].format(
                        video_key=key,
                        chunk_index=int(row[chunk_key]),
                        file_index=int(row[file_key]),
                    )
                )
                video_files.append(video_file)
                video_segments.append(
                    {
                        "key": key,
                        "file": video_file,
                        "from_timestamp": float(row[f"videos/{key}/from_timestamp"]),
                        "to_timestamp": float(row[f"videos/{key}/to_timestamp"]),
                    }
                )
        duration = int(row["length"]) / fps
        selected.append(
            SelectedEpisode(
                episode_index=int(row["episode_index"]),
                tasks=list(row.get("tasks") or []),
                frames=int(row["length"]),
                seconds=duration,
                data_file=data_file,
                video_files=video_files,
                video_segments=video_segments,
            )
        )
        elapsed += duration
        if elapsed >= seconds:
            break
    if elapsed < seconds:
        raise RuntimeError(
            f"{repo_id} contains only {elapsed:.1f}s matching the requested selection"
        )
    manifest = {
        "repo_id": repo_id,
        "revision": HfApi(token=token).dataset_info(repo_id, token=token).sha,
        "requested_seconds": seconds,
        "selected_seconds": elapsed,
        "fps": fps,
        "selected_frames": sum(value.frames for value in selected),
        "task_keywords": task_keywords or [],
        "prefix": prefix,
        "episodes": [asdict(value) for value in selected],
    }
    (root / "sample_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest, selected


def download_lerobot_hour(
    repo_id: str,
    destination: str | Path,
    seconds: float = 3600.0,
    task_keywords: list[str] | None = None,
    cameras: list[str] | None = None,
    token: str | None = None,
    metadata_only: bool = False,
    prefix: str = "",
) -> Path:
    """Download exactly selected episodes plus their shared Parquet/video containers."""
    root = Path(destination) / repo_id.replace("/", "__")
    token = token or os.environ.get("HF_TOKEN")
    try:
        manifest, selected = plan_lerobot_hour(
            repo_id,
            destination,
            seconds=seconds,
            task_keywords=task_keywords,
            token=token,
            prefix=prefix,
        )
    except (GatedRepoError, HfHubHTTPError) as error:
        status = {
            "repo_id": repo_id,
            "status": "gated_or_unauthorized",
            "error": str(error),
            "requires": "Accept repository terms and set HF_TOKEN",
        }
        root.mkdir(parents=True, exist_ok=True)
        (root / "download_status.json").write_text(json.dumps(status, indent=2) + "\n")
        raise
    if metadata_only:
        return root / "sample_manifest.json"
    data_files = sorted({episode.data_file for episode in selected})
    video_files = sorted({name for episode in selected for name in episode.video_files})
    if cameras:
        video_files = [
            name for name in video_files if any(f"videos/{camera}/" in name for camera in cameras)
        ]
    for filename in data_files + video_files:
        _download(repo_id, filename, root, token)

    selected_indices = pa.array([episode.episode_index for episode in selected])
    tables: list[pa.Table] = []
    for filename in data_files:
        table = pq.read_table(root / filename)
        tables.append(table.filter(pc.is_in(table["episode_index"], value_set=selected_indices)))
    sample = pa.concat_tables(tables, promote_options="default")
    output = root / "sample" / "data.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(sample, output, compression="zstd")
    manifest["sample_rows"] = sample.num_rows
    manifest["sample_file"] = "sample/data.parquet"
    manifest["downloaded_data_files"] = data_files
    manifest["downloaded_video_files"] = video_files
    file_records: dict[str, dict[str, int | str]] = {}
    for filename in data_files + video_files:
        file_records[filename] = _file_record(root / filename)
    file_records[manifest["sample_file"]] = _file_record(output)
    manifest["files"] = file_records
    (root / "sample_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return root / "sample_manifest.json"


def verify_download(manifest_path: str | Path, rehash: bool = True) -> dict[str, Any]:
    """Verify duration, exact selected rows, file sizes, and optional SHA-256 digests."""
    manifest_path = Path(manifest_path)
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    expected_frames = sum(int(episode["frames"]) for episode in manifest["episodes"])
    errors: list[str] = []
    if float(manifest["selected_seconds"]) < float(manifest["requested_seconds"]):
        errors.append("selected duration is shorter than requested duration")
    if int(manifest.get("selected_frames", expected_frames)) != expected_frames:
        errors.append("selected_frames does not equal the episode frame sum")
    if int(manifest.get("sample_rows", -1)) != expected_frames:
        errors.append("sample_rows does not equal the episode frame sum")
    for filename, record in manifest.get("files", {}).items():
        path = root / filename
        if not path.is_file():
            errors.append(f"missing file: {filename}")
            continue
        if path.stat().st_size != int(record["bytes"]):
            errors.append(f"size mismatch: {filename}")
        elif rehash and _file_record(path)["sha256"] != record["sha256"]:
            errors.append(f"SHA-256 mismatch: {filename}")
    if not manifest.get("files"):
        errors.append("manifest contains no file checksums; rerun download-hour")
    return {
        "ok": not errors,
        "manifest": str(manifest_path),
        "repo_id": manifest.get("repo_id"),
        "revision": manifest.get("revision"),
        "selected_seconds": manifest.get("selected_seconds"),
        "selected_frames": expected_frames,
        "episodes": len(manifest["episodes"]),
        "files": len(manifest.get("files", {})),
        "rehash": rehash,
        "errors": errors,
    }


def probe_repo(repo_id: str, token: str | None = None) -> dict[str, Any]:
    token = token or os.environ.get("HF_TOKEN")
    try:
        info = HfApi(token=token).dataset_info(repo_id, token=token)
        gated = info.gated
        return {
            "repo_id": repo_id,
            "status": "gated" if gated else "accessible",
            "gated": gated,
            "revision": info.sha,
            "license": (info.card_data or {}).get("license") if info.card_data else None,
            **({"requires": "Accept repository terms and set HF_TOKEN"} if gated else {}),
        }
    except GatedRepoError as error:
        return {"repo_id": repo_id, "status": "gated", "error": str(error)}
    except HfHubHTTPError as error:
        return {"repo_id": repo_id, "status": "unavailable", "error": str(error)}
