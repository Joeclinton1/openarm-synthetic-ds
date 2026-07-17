#!/usr/bin/env python3
"""Fetch the minimal RoboMIND supplement needed for a second benchmark task."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from openarm_retarget.download import DEFAULT_MAX_SLICE_BYTES


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ROOT = ROOT / "data/samples/Traly__RoboMIND-lerobot"
PREFIX_ROOT = SAMPLE_ROOT / "agilex_3rgb"
BASE_MANIFEST = SAMPLE_ROOT / "sample_manifest.json"
OUTPUT_MANIFEST = SAMPLE_ROOT / "benchmark_supplement_manifest.json"
CAMERA = "observation.images.camera_front"
EPISODES = tuple(range(210, 216))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    base = json.loads(BASE_MANIFEST.read_text())
    repo_id = str(base["repo_id"])
    revision = str(base["revision"])
    maximum = min(int(base["max_slice_bytes"]), DEFAULT_MAX_SLICE_BYTES)

    columns = [
        "episode_index",
        "tasks",
        "length",
        "data/chunk_index",
        "data/file_index",
        f"videos/{CAMERA}/chunk_index",
        f"videos/{CAMERA}/file_index",
        f"videos/{CAMERA}/from_timestamp",
        f"videos/{CAMERA}/to_timestamp",
    ]
    rows: list[dict] = []
    for path in sorted((PREFIX_ROOT / "meta/episodes").rglob("*.parquet")):
        rows.extend(pq.read_table(path, columns=columns).to_pylist())
    selected = {int(row["episode_index"]): row for row in rows if row["episode_index"] in EPISODES}
    missing = sorted(set(EPISODES) - set(selected))
    if missing:
        raise RuntimeError(f"RoboMIND metadata is missing supplement episodes: {missing}")

    info = json.loads((PREFIX_ROOT / "meta/info.json").read_text())
    fps = float(info["fps"])
    episodes: list[dict] = []
    remote_files: set[str] = set()
    for episode_index in EPISODES:
        row = selected[episode_index]
        data_file = "agilex_3rgb/" + info["data_path"].format(
            chunk_index=int(row["data/chunk_index"]),
            file_index=int(row["data/file_index"]),
        )
        video_file = "agilex_3rgb/" + info["video_path"].format(
            video_key=CAMERA,
            chunk_index=int(row[f"videos/{CAMERA}/chunk_index"]),
            file_index=int(row[f"videos/{CAMERA}/file_index"]),
        )
        if not (SAMPLE_ROOT / data_file).exists():
            raise FileNotFoundError(SAMPLE_ROOT / data_file)
        remote_files.add(video_file)
        episodes.append(
            {
                "episode_index": episode_index,
                "tasks": list(row["tasks"] or []),
                "frames": int(row["length"]),
                "seconds": int(row["length"]) / fps,
                "data_file": data_file,
                "video_files": [video_file],
                "video_segments": [
                    {
                        "key": CAMERA,
                        "file": video_file,
                        "from_timestamp": float(row[f"videos/{CAMERA}/from_timestamp"]),
                        "to_timestamp": float(row[f"videos/{CAMERA}/to_timestamp"]),
                    }
                ],
            }
        )

    token = os.environ.get("HF_TOKEN")
    repo_info = HfApi(token=token).dataset_info(repo_id, files_metadata=True, token=token)
    sizes = {item.rfilename: item.size for item in repo_info.siblings}
    if repo_info.sha != revision:
        raise RuntimeError(f"Dataset revision changed: expected {revision}, found {repo_info.sha}")
    unavailable = [name for name in sorted(remote_files) if sizes.get(name) is None]
    if unavailable:
        raise RuntimeError(f"Repository does not report supplement sizes for: {unavailable}")
    supplement_bytes = sum(int(sizes[name]) for name in remote_files)
    cumulative_bytes = int(base["downloaded_bytes"]) + supplement_bytes
    if cumulative_bytes > maximum:
        raise RuntimeError(
            f"Base slice plus benchmark supplement is {cumulative_bytes:,} bytes, "
            f"above the {maximum:,}-byte limit"
        )

    files: dict[str, dict[str, int | str]] = {}
    for filename in sorted(remote_files):
        path = Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                filename=filename,
                local_dir=SAMPLE_ROOT,
                token=token,
            )
        )
        files[filename] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}

    actual_supplement_bytes = sum(int(value["bytes"]) for value in files.values())
    manifest = {
        "repo_id": repo_id,
        "revision": revision,
        "purpose": "second distinct RoboMIND AgileX task for the cross-dataset benchmark",
        "camera": CAMERA,
        "fps": fps,
        "episodes": episodes,
        "files": files,
        "supplement_downloaded_bytes": actual_supplement_bytes,
        "base_slice_downloaded_bytes": int(base["downloaded_bytes"]),
        "cumulative_slice_bytes": int(base["downloaded_bytes"]) + actual_supplement_bytes,
        "max_slice_bytes": maximum,
    }
    OUTPUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(OUTPUT_MANIFEST)
    print(
        f"downloaded {actual_supplement_bytes:,} supplement bytes; "
        f"cumulative slice {manifest['cumulative_slice_bytes']:,}/{maximum:,} bytes"
    )


if __name__ == "__main__":
    main()
