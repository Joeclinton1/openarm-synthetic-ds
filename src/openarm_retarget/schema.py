from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .constants import FPS, SIDES
from .poses import PoseTransform, make_quaternions_continuous

PoseConfig = list[float] | dict[str, list[float]] | None


@dataclass
class Episode:
    """Intermediate, lossless representation used between source adapters and IK."""

    timestamp: np.ndarray
    ee_pose: np.ndarray  # [T, 2, 7], sides ordered right, left; metres + xyzw
    gripper: np.ndarray  # [T, 2], normalized 0=open, 1=closed
    task: str
    source_dataset: str
    source_episode: str
    gripper_width_m: np.ndarray | None = None  # [T, 2] physical pad separation when known
    joint_position: np.ndarray | None = None  # [T, 2, 7] after OpenArm IK
    feasible: np.ndarray | None = None
    diagnostics: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.timestamp = np.asarray(self.timestamp, dtype=np.float64)
        self.ee_pose = np.asarray(self.ee_pose, dtype=np.float64)
        self.gripper = np.asarray(self.gripper, dtype=np.float64)
        n = len(self.timestamp)
        if self.ee_pose.shape != (n, 2, 7):
            raise ValueError(f"ee_pose must be ({n}, 2, 7), got {self.ee_pose.shape}")
        if self.gripper.shape != (n, 2):
            raise ValueError(f"gripper must be ({n}, 2), got {self.gripper.shape}")
        if not np.all(np.isfinite(self.gripper)) or np.any((self.gripper < 0) | (self.gripper > 1)):
            raise ValueError("gripper must be finite normalized closure in [0,1]")
        if self.gripper_width_m is not None:
            self.gripper_width_m = np.asarray(self.gripper_width_m, dtype=np.float64)
            if self.gripper_width_m.shape != (n, 2):
                raise ValueError(
                    f"gripper_width_m must be ({n}, 2), got {self.gripper_width_m.shape}"
                )
            if np.any(self.gripper_width_m < 0) or not np.all(np.isfinite(self.gripper_width_m)):
                raise ValueError("gripper_width_m must be finite and non-negative")
        if n and (not np.all(np.isfinite(self.timestamp)) or np.any(np.diff(self.timestamp) <= 0)):
            raise ValueError("Timestamps must be finite and strictly increasing")
        for side_index in range(2):
            self.ee_pose[:, side_index, 3:] = make_quaternions_continuous(
                self.ee_pose[:, side_index, 3:]
            )
        if self.joint_position is not None:
            self.joint_position = np.asarray(self.joint_position, dtype=np.float64)
            if self.joint_position.shape != (n, 2, 7):
                raise ValueError("joint_position must have shape [T, 2, 7]")

    @property
    def duration(self) -> float:
        return (
            float(self.timestamp[-1] - self.timestamp[0] + self.sample_period)
            if len(self.timestamp)
            else 0
        )

    @property
    def sample_period(self) -> float:
        return float(np.median(np.diff(self.timestamp))) if len(self.timestamp) > 1 else 1 / FPS

    def sliced(self, start: int = 0, end: int | None = None) -> "Episode":
        selection = slice(start, end)
        timestamp = self.timestamp[selection].copy()
        if len(timestamp):
            timestamp -= timestamp[0]
        result = Episode(
            timestamp=timestamp,
            ee_pose=self.ee_pose[selection].copy(),
            gripper=self.gripper[selection].copy(),
            task=self.task,
            source_dataset=self.source_dataset,
            source_episode=self.source_episode,
            gripper_width_m=(
                self.gripper_width_m[selection].copy() if self.gripper_width_m is not None else None
            ),
            joint_position=(
                self.joint_position[selection].copy() if self.joint_position is not None else None
            ),
            feasible=self.feasible[selection].copy() if self.feasible is not None else None,
            diagnostics={key: value[selection].copy() for key, value in self.diagnostics.items()},
            metadata={**self.metadata, "source_frame_slice": [start, end]},
        )
        result.validate()
        return result

    def save(self, path: str | Path) -> None:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "timestamp": self.timestamp,
            "ee_pose": self.ee_pose,
            "gripper": self.gripper,
            "metadata_json": np.asarray(
                json.dumps(
                    {
                        "task": self.task,
                        "source_dataset": self.source_dataset,
                        "source_episode": self.source_episode,
                        **self.metadata,
                    }
                )
            ),
        }
        if self.joint_position is not None:
            arrays["joint_position"] = self.joint_position
        if self.gripper_width_m is not None:
            arrays["gripper_width_m"] = self.gripper_width_m
        if self.feasible is not None:
            arrays["feasible"] = self.feasible
        arrays.update({f"diagnostic_{k}": v for k, v in self.diagnostics.items()})
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "Episode":
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
            episode = cls(
                timestamp=data["timestamp"],
                ee_pose=data["ee_pose"],
                gripper=data["gripper"],
                task=metadata.pop("task"),
                source_dataset=metadata.pop("source_dataset"),
                source_episode=metadata.pop("source_episode"),
                gripper_width_m=data["gripper_width_m"] if "gripper_width_m" in data else None,
                joint_position=data["joint_position"] if "joint_position" in data else None,
                feasible=data["feasible"] if "feasible" in data else None,
                diagnostics={
                    key.removeprefix("diagnostic_"): data[key]
                    for key in data.files
                    if key.startswith("diagnostic_")
                },
                metadata=metadata,
            )
        episode.validate()
        return episode


@dataclass(frozen=True)
class SourceConfig:
    name: str
    repo_id: str
    adapter: str
    dataset_prefix: str = ""
    fps: float = FPS
    quaternion_order: str = "xyzw"
    rotation_representation: str = "quaternion"
    rotation_euler_order: str = "xyz"
    position_scale: float = 1.0
    openarm_from_source_base: PoseConfig = None
    source_tool_from_openarm_tool: PoseConfig = None
    arm_order: list[str] = field(default_factory=lambda: list(SIDES))
    single_arm_side: str | None = None
    calibrated: bool = False
    gripper_mode: str = "normalized"
    gripper_open_value: float | None = None
    gripper_closed_value: float | None = None
    source_pinch_center_open_m: dict[str, list[float]] | list[float] | None = None
    source_pinch_center_closed_m: dict[str, list[float]] | list[float] | None = None
    preserve_pinch_center: bool = False
    fields: dict[str, str] = field(default_factory=dict)
    tabletop_tasks: list[str] = field(default_factory=list)
    notes: str = ""

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SourceConfig":
        import yaml

        values = yaml.safe_load(Path(path).read_text())
        return cls(**values)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    def pose_transform(self, side: str) -> PoseTransform:
        identity = [0, 0, 0, 0, 0, 0, 1]

        def select(value: PoseConfig) -> list[float]:
            if value is None:
                return identity
            if isinstance(value, dict):
                if side not in value:
                    raise ValueError(f"Missing {side} calibration transform")
                return value[side]
            return value

        return PoseTransform(
            np.asarray(select(self.openarm_from_source_base), dtype=np.float64),
            np.asarray(select(self.source_tool_from_openarm_tool), dtype=np.float64),
            self.position_scale,
        )
