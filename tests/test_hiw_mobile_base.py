import numpy as np

from openarm_retarget.presets import integrate_planar_base_velocity


def test_integrate_planar_base_velocity_fixes_height() -> None:
    timestamp = np.array([0.0, 0.5, 1.0])
    velocity = np.array([[1.0, 0.0], [1.0, 2.0], [1.0, 2.0]])
    translation = integrate_planar_base_velocity(timestamp, velocity)
    np.testing.assert_allclose(
        translation,
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [1.0, 1.5, 0.0]],
    )
    np.testing.assert_allclose(translation[:, 2], 0.0)
