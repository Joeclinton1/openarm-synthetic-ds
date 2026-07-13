from __future__ import annotations

import copy
import csv
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .constants import OPENARM_MUJOCO_COMMIT, SIDES
from .filters import filter_episode
from .ik import OpenArmIK
from .official_ik import OfficialIKConfig, OfficialOpenArmIK
from .schema import Episode
from .viewer import TrajectoryViewer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def summarize_ik(episode: Episode, solve_seconds: float, solver: OpenArmIK) -> dict[str, object]:
    active = [SIDES.index(side) for side in episode.metadata.get("active_sides", SIDES)]
    position = episode.diagnostics["position_error_m"][:, active]
    orientation = episode.diagnostics["orientation_error_rad"][:, active]
    velocity = np.abs(episode.diagnostics["joint_velocity_rad_s"][:, active])
    acceleration = np.abs(episode.diagnostics["joint_acceleration_rad_s2"][:, active])
    per_frame_velocity = np.max(velocity, axis=(1, 2))
    per_frame_acceleration = np.max(acceleration, axis=(1, 2))
    if len(episode.timestamp) > 3:
        dt = np.diff(episode.timestamp)
        acceleration_raw = np.diff(episode.joint_position, n=2, axis=0) / (
            ((dt[1:] + dt[:-1]) / 2)[:, None, None] ** 2
        )
        jerk = np.diff(acceleration_raw, axis=0) / dt[2:, None, None]
        per_frame_jerk = np.max(np.abs(jerk[:, active]), axis=(1, 2))
    else:
        per_frame_jerk = np.zeros(1)
    per_frame_limit_margin = np.full(len(episode.timestamp), np.inf)
    for side_index in active:
        limits = solver.limits(SIDES[side_index])
        q = episode.joint_position[:, side_index]
        per_frame_limit_margin = np.minimum(
            per_frame_limit_margin,
            np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q), axis=1),
        )
    limit_margin = float(np.min(per_frame_limit_margin))
    strict = (position <= solver.config.position_tolerance) & (
        orientation <= solver.config.orientation_tolerance
    )
    failures = episode.diagnostics.get("official_solver_failed", np.zeros(len(episode.timestamp)))
    retries = episode.diagnostics.get("official_target_retries", np.zeros(len(episode.timestamp)))
    limit_violations = episode.diagnostics["invalid_joint_limit"]
    infeasible_frames = np.flatnonzero(~episode.feasible)
    worst_position_frame = int(np.unravel_index(np.argmax(position), position.shape)[0])
    worst_orientation_frame = int(np.unravel_index(np.argmax(orientation), orientation.shape)[0])
    worst_acceleration_frame = int(np.argmax(per_frame_acceleration))
    worst_joint_limit_frame = int(np.argmin(per_frame_limit_margin))
    return {
        "frames": len(episode.timestamp),
        "solve_seconds": float(solve_seconds),
        "solve_fps": float(len(episode.timestamp) / solve_seconds) if solve_seconds else None,
        "strict_arm_frame_success_fraction": float(np.mean(strict)),
        "strict_bimanual_frame_success_fraction": float(np.mean(np.all(strict, axis=1))),
        "feasible_frame_fraction": float(np.mean(episode.feasible)),
        "position_error_m": {
            "mean": float(np.mean(position)),
            "p95": _percentile(position, 95),
            "max": float(np.max(position)),
        },
        "orientation_error_rad": {
            "mean": float(np.mean(orientation)),
            "p95": _percentile(orientation, 95),
            "max": float(np.max(orientation)),
        },
        "peak_joint_velocity_rad_s": {
            "p95": _percentile(per_frame_velocity, 95),
            "max": float(np.max(per_frame_velocity)),
        },
        "peak_joint_acceleration_rad_s2": {
            "p95": _percentile(per_frame_acceleration, 95),
            "max": float(np.max(per_frame_acceleration)),
        },
        "peak_joint_jerk_rad_s3": {
            "p95": _percentile(per_frame_jerk, 95),
            "max": float(np.max(per_frame_jerk)),
        },
        "minimum_joint_limit_margin_rad": limit_margin,
        "joint_limit_violation_frames": int(np.sum(limit_violations)),
        "collision_frames": int(np.sum(episode.diagnostics["invalid_collision"])),
        "solver_failure_frames": int(np.sum(failures)),
        "target_retry_frames": int(np.count_nonzero(retries)),
        "target_retries_total": int(np.sum(retries)),
        "infeasible_frames": [int(value) for value in infeasible_frames],
        "worst_frames": {
            "position_error": worst_position_frame,
            "orientation_error": worst_orientation_frame,
            "joint_acceleration": worst_acceleration_frame,
            "joint_limit": worst_joint_limit_frame,
        },
    }


def _combine_videos(
    videos: list[Path], labels: list[str], output: Path, frame_range: tuple[int, int] | None = None
) -> Path:
    captures = [cv2.VideoCapture(str(path)) for path in videos]
    try:
        if not all(capture.isOpened() for capture in captures):
            raise RuntimeError("Could not open every IK review video")
        widths = [int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) for capture in captures]
        heights = [int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) for capture in captures]
        counts = [int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) for capture in captures]
        fps = float(captures[0].get(cv2.CAP_PROP_FPS))
        if len(set(zip(widths, heights, counts))) != 1:
            raise ValueError("IK review inputs must have identical dimensions and frame counts")
        start, end = frame_range or (0, counts[0])
        start = max(0, start)
        end = min(counts[0], end)
        if end <= start:
            raise ValueError("IK review frame range is empty")
        for capture in captures:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (sum(widths), heights[0]),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not create {output}")
        try:
            for _ in range(start, end):
                frames = []
                for capture, label in zip(captures, labels, strict=True):
                    ok, frame = capture.read()
                    if not ok:
                        raise RuntimeError("IK review input ended early")
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (35, 35, 35), -1)
                    cv2.putText(
                        frame,
                        label,
                        (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (245, 245, 245),
                        1,
                        cv2.LINE_AA,
                    )
                    frames.append(frame)
                writer.write(np.hstack(frames))
        finally:
            writer.release()
    finally:
        for capture in captures:
            capture.release()
    return output


def package_ik_review_videos(
    current_video: str | Path,
    official_video: str | Path,
    destination: str | Path,
    *,
    source_video: str | Path | None = None,
    removed_video: str | Path | None = None,
    contact_start: int = 560,
    contact_end: int = 770,
    still_frames: tuple[int, ...] = (0, 300, 600, 672, 900, 1025),
) -> dict[str, str]:
    destination = Path(destination)
    videos = [Path(current_video), Path(official_video)]
    labels = ["CURRENT DLS IK", "OFFICIAL DORA / MINK IK"]
    prefix = "two_way"
    if (source_video is None) != (removed_video is None):
        raise ValueError("source_video and removed_video must be supplied together")
    if source_video is not None:
        videos = [Path(source_video), Path(removed_video), *videos]
        labels = ["SOURCE", "ARM REMOVED", *labels]
        prefix = "four_way"
    full = _combine_videos(videos, labels, destination / f"full_{prefix}.mp4")
    contact = _combine_videos(
        videos,
        labels,
        destination / f"contact_{prefix}.mp4",
        (contact_start, contact_end),
    )
    capture = cv2.VideoCapture(str(full))
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for frame in sorted(set(value for value in still_frames if 0 <= value < count)):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"Could not extract review frame {frame}")
            cv2.imwrite(str(destination / f"frame_{frame}_{prefix}.png"), image)
    finally:
        capture.release()
    return {"full": str(full), "contact": str(contact)}


def _write_review_files(destination: Path, metrics: dict[str, object]) -> None:
    frames = metrics["frames"]
    current = metrics["current_dls"]
    official = metrics["official_mink"]
    destination.joinpath("HUMAN_REVIEW.md").write_text(
        f"""# IK human review: {metrics["source_episode"]}

This package compares the existing production DLS solver with the pinned official
Dora/OpenArm MuJoCo+Mink solver on the same {frames}-frame target trajectory.
Neither solver has been removed or promoted by this comparison.

## Start here

- `full_two_way.mp4`: synchronized complete trajectory.
- `contact_two_way.mp4`: frames {metrics["contact_frames"][0]}--{metrics["contact_frames"][1]}.
- `full_four_way.mp4` / `contact_four_way.mp4`: source-context versions when packaged.
- `frame_*_two_way.png`: fixed-frame comparisons.
- `metrics.json`: Cartesian, temporal, collision, limit, provenance, and runtime results.
- `review_log.csv`: record one issue per row.

## Automated result

- Current DLS: {current["feasible_frame_fraction"]:.2%} feasible; zero joint-limit violations.
- Official Mink: {official["feasible_frame_fraction"]:.2%} feasible;
  {official["joint_limit_violation_frames"]} joint-limit violation frames and
  {official["peak_joint_acceleration_rad_s2"]["max"]:.2f} rad/s^2 maximum acceleration.
- Official mean position error is {official["position_error_m"]["mean"] * 1000:.3f} mm versus
  {current["position_error_m"]["mean"] * 1000:.3f} mm for current DLS, but it does not pass the
  production safety/temporal gates. Keep the current solver.

The context composites are for posture/motion review. They reuse the accepted arm-removed panel,
but the recovered protected-object masks were unavailable after workspace cleanup, so contact
occlusion in this regenerated package is not a promotion gate.

## Review criteria

- Does either arm visibly lag the intended motion or jump between redundant configurations?
- Are elbow and wrist choices stable through contact and near singular configurations?
- Is there visible high-frequency joint motion, an elbow flip, or a first-frame transient?
- Do the two arms collide, approach limits, or enter implausible postures?
- Is the official result clearly preferable before considering removal of the current fallback?

Promote the official solver only after equivalent checks pass across every retained source,
including worst-error frames; this fixture alone is not sufficient for removal.
"""
    )
    with destination.joinpath("review_log.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "reviewer",
                "timestamp_seconds",
                "frame",
                "solver",
                "severity",
                "category",
                "description",
                "status",
            ]
        )
        limit_frame = official["worst_frames"]["joint_limit"]
        writer.writerow(
            [
                "automated",
                f"{limit_frame / 30:.3f}",
                limit_frame,
                "official_mink",
                "blocker",
                "joint_limit",
                f"Minimum model-limit margin {official['minimum_joint_limit_margin_rad']:.6f} rad",
                "open",
            ]
        )
        acceleration_frame = official["worst_frames"]["joint_acceleration"]
        writer.writerow(
            [
                "automated",
                f"{acceleration_frame / 30:.3f}",
                acceleration_frame,
                "official_mink",
                "major",
                "joint_acceleration",
                f"Peak acceleration {official['peak_joint_acceleration_rad_s2']['max']:.6f} rad/s^2",
                "open",
            ]
        )


def compare_ik_episode(
    source: str | Path,
    destination: str | Path,
    model_path: str | Path | None = None,
    *,
    official_config: OfficialIKConfig | None = None,
    render: bool = True,
    width: int = 640,
    height: int = 480,
    contact_start: int = 560,
    contact_end: int = 770,
) -> dict[str, object]:
    source_path = Path(source).resolve()
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    canonical = Episode.load(source_path)

    current_episode = copy.deepcopy(canonical)
    current_solver = OpenArmIK(model_path)
    started = time.perf_counter()
    current_solver.solve_episode(current_episode)
    current_seconds = time.perf_counter() - started
    filter_episode(current_episode, current_solver)

    official_episode = copy.deepcopy(canonical)
    official_solver = OfficialOpenArmIK(model_path, official_config)
    official_solver.solve_episode(official_episode)
    filter_episode(official_episode, official_solver.evaluator)

    current_path = destination / "current_dls.npz"
    official_path = destination / "official_mink.npz"
    current_episode.save(current_path)
    official_episode.save(official_path)
    current_metrics = summarize_ik(current_episode, current_seconds, current_solver)
    official_metrics = summarize_ik(
        official_episode, official_solver.solve_seconds, official_solver.evaluator
    )
    metrics: dict[str, object] = {
        "schema": "openarm-ik-comparison-v1",
        "source": str(source_path),
        "source_sha256": _sha256(source_path),
        "source_episode": canonical.source_episode,
        "frames": len(canonical.timestamp),
        "model_revision": OPENARM_MUJOCO_COMMIT,
        "contact_frames": [max(0, contact_start), min(len(canonical.timestamp), contact_end)],
        "current_dls": current_metrics,
        "official_mink": official_metrics,
        "official_provenance": official_solver.provenance,
        "promotion_gate": {
            "official_all_frames_feasible": official_metrics["feasible_frame_fraction"] == 1.0,
            "official_all_strict_bimanual_frames": official_metrics[
                "strict_bimanual_frame_success_fraction"
            ]
            == 1.0,
            "official_zero_collisions": official_metrics["collision_frames"] == 0,
            "official_zero_joint_limit_violations": official_metrics["joint_limit_violation_frames"]
            == 0,
            "official_zero_solver_failures": official_metrics["solver_failure_frames"] == 0,
            "official_realtime_throughput": official_metrics["solve_fps"] >= 30.0,
            "automatic_recommendation": "retain current solver pending cross-dataset and human review",
        },
    }
    metrics_path = destination / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    _write_review_files(destination, metrics)

    if render:
        try:
            viewer = TrajectoryViewer(model_path)
            current_video = viewer.render(
                current_episode, destination / "current_dls.mp4", width=width, height=height
            )
            official_video = viewer.render(
                official_episode, destination / "official_mink.mp4", width=width, height=height
            )
        except Exception as exc:
            if "OpenGL platform" in str(exc):
                raise RuntimeError(
                    "Headless MuJoCo review rendering requires `MUJOCO_GL=egl`; "
                    "the solved episodes and metrics were still written"
                ) from exc
            raise
        package_ik_review_videos(
            current_video,
            official_video,
            destination,
            contact_start=contact_start,
            contact_end=contact_end,
            still_frames=tuple(
                sorted(
                    {
                        0,
                        300,
                        600,
                        672,
                        900,
                        len(canonical.timestamp) - 1,
                        *current_metrics["worst_frames"].values(),
                        *official_metrics["worst_frames"].values(),
                    }
                )
            ),
        )
    return metrics
