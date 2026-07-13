#!/usr/bin/env python3
"""Acquire deterministic one-hour slices from public repositories and audit AgiBot access."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from openarm_retarget.download import download_lerobot_hour, probe_repo, verify_download
from openarm_retarget.schema import SourceConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=Path("data/samples"))
    parser.add_argument("--seconds", type=float, default=3600)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    configs = [
        Path("configs/sources/hiw500.yaml"),
        Path("configs/sources/unifolm.yaml"),
        Path("configs/sources/molmoact2_tabletop.yaml"),
        Path("configs/sources/agibot_derived_fallback.yaml"),
    ]
    report: dict[str, object] = {}
    for path in configs:
        config = SourceConfig.from_yaml(path)
        try:
            root = args.destination / config.repo_id.replace("/", "__")
            existing = root / "sample_manifest.json"
            if args.metadata_only and existing.exists() and json.loads(existing.read_text()).get("files"):
                manifest = existing
            else:
                manifest = download_lerobot_hour(
                    config.repo_id,
                    args.destination,
                    args.seconds,
                    config.tabletop_tasks,
                    cameras=[],
                    metadata_only=args.metadata_only,
                    prefix=config.dataset_prefix,
                )
            report[config.name] = (
                verify_download(manifest, rehash=False)
                if json.loads(manifest.read_text()).get("files")
                else {"status": "planned", "manifest": str(manifest)}
            )
        except Exception as error:
            report[config.name] = {"status": "failed", "error": str(error)}
    agibot = SourceConfig.from_yaml("configs/sources/agibot.yaml")
    report[f"{agibot.name} (original)"] = probe_repo(agibot.repo_id)
    args.destination.mkdir(parents=True, exist_ok=True)
    output = args.destination / "acquisition_report.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
