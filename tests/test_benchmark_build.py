import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_cross_dataset_benchmark.py"
SPEC = importlib.util.spec_from_file_location("build_cross_dataset_benchmark", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def test_moving_joint_sides_excludes_nearly_static_arm() -> None:
    joints = np.zeros((20, 2, 7))
    joints[:, 0, 0] = np.linspace(0, 0.01, len(joints))
    joints[:, 1, 0] = np.linspace(0, 1, len(joints))
    assert benchmark._moving_joint_sides(joints, ["right", "left"]) == ["left"]


def test_moving_joint_sides_retains_material_bimanual_motion() -> None:
    joints = np.zeros((20, 2, 7))
    joints[:, 0, 0] = np.linspace(0, 1, len(joints))
    joints[:, 1, 0] = np.linspace(0, 0.2, len(joints))
    assert benchmark._moving_joint_sides(joints, ["right", "left"]) == [
        "right",
        "left",
    ]


def test_new_dataset_families_have_two_distinct_tasks_each() -> None:
    for dataset in ("droid", "rh20t_franka", "robomind_agilex_3rgb"):
        selections = [item for item in benchmark.SELECTIONS if item.dataset == dataset]
        assert len(selections) == 2
        assert len({item.task for item in selections}) == 2
        assert len({item.episode for item in selections}) == 2
