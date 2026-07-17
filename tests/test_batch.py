from dataclasses import replace

from openarm_retarget.batch import _conversion_signature
from openarm_retarget.schema import SourceConfig


def test_conversion_signature_invalidates_changed_tool_convention() -> None:
    config = SourceConfig(name="fixture", repo_id="fixture/repo", adapter="lerobot")
    changed = replace(
        config,
        source_tool_from_openarm_tool=[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    )
    assert _conversion_signature(config) != _conversion_signature(changed)
