import numpy as np

from openarm_retarget.viewer import TrajectoryViewer


class FakeCamera:
    def __init__(self):
        self.pos = np.zeros(3)
        self.forward = np.zeros(3)
        self.up = np.zeros(3)
        self.frustum_near = 0.0
        self.frustum_far = 0.0
        self.frustum_top = 0.0
        self.frustum_bottom = 0.0
        self.frustum_center = 0.0
        self.frustum_width = 0.0
        self.orthographic = 1


class FakeRenderer:
    def __init__(self):
        self.scene = type("Scene", (), {"camera": [FakeCamera(), FakeCamera()]})()


def test_calibrated_camera_converts_opencv_axes() -> None:
    renderer = FakeRenderer()
    spec = {
        "world_from_camera": np.eye(4).tolist(),
        "intrinsics": [[100, 0, 50], [0, 100, 40], [0, 0, 1]],
    }
    TrajectoryViewer._set_calibrated_camera(renderer, spec, 100, 80)
    camera = renderer.scene.camera[0]
    np.testing.assert_allclose(camera.forward, [0, 0, 1])
    np.testing.assert_allclose(camera.up, [0, -1, 0])
    assert camera.frustum_center == 0
    assert camera.frustum_top == 0.004
    assert camera.frustum_bottom == -0.004
    assert camera.frustum_width == 0.005


def test_calibrated_camera_preserves_anisotropic_focal_lengths() -> None:
    renderer = FakeRenderer()
    spec = {
        "world_from_camera": np.eye(4).tolist(),
        "intrinsics": [[320, 0, 320], [0, 430, 240], [0, 0, 1]],
    }
    TrajectoryViewer._set_calibrated_camera(renderer, spec, 640, 480)
    camera = renderer.scene.camera[0]
    assert camera.frustum_width == 0.01
    np.testing.assert_allclose(camera.frustum_top, 240 * 0.01 / 430)
