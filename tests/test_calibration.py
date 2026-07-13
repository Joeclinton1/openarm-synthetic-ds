import numpy as np
from scipy.spatial.transform import Rotation

from openarm_retarget.calibration import fit_rigid_transform


def test_fit_rigid_transform() -> None:
    source = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=float)
    rotation = Rotation.from_euler("xyz", [0.2, -0.3, 1.0]).as_matrix()
    translation = np.array([0.4, -0.5, 0.2])
    target = (rotation @ source.T).T + translation
    pose, rms = fit_rigid_transform(source, target)
    np.testing.assert_allclose(pose[:3], translation, atol=1e-12)
    np.testing.assert_allclose(Rotation.from_quat(pose[3:]).as_matrix(), rotation, atol=1e-12)
    assert rms < 1e-12


def test_fit_rejects_collinear_points() -> None:
    points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    try:
        fit_rigid_transform(points, points)
    except ValueError as error:
        assert "non-collinear" in str(error)
    else:
        raise AssertionError("collinear calibration accepted")
