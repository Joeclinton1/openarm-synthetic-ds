from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def openarm_model_path() -> Path:
    local = Path("data/assets/openarm_mujoco/v2/openarm_bimanual.xml")
    if local.exists():
        return local.resolve()
    upstream = Path("/tmp/openarm_mujoco/v2/openarm_bimanual.xml")
    if upstream.exists():
        return upstream
    pytest.skip("official OpenArm model not fetched")
