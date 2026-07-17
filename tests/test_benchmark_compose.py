import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/compose_benchmark_outputs.py"
SPEC = importlib.util.spec_from_file_location("compose_benchmark_outputs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
composition = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = composition
SPEC.loader.exec_module(composition)


def test_right_edge_component_ignores_left_arm() -> None:
    mask = np.zeros((120, 160), dtype=np.uint8)
    cv2.rectangle(mask, (0, 20), (50, 80), 255, -1)
    cv2.rectangle(mask, (120, 30), (159, 100), 255, -1)
    component = composition._right_edge_component(mask)
    assert float(component[:, :80].max()) < 0.01
    assert float(component[:, 130:].mean()) > 0.5
