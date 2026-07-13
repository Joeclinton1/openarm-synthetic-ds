# OpenArm 2.0 dataset retargeting

Tools for sampling heterogeneous robot datasets, converting end-effector poses into the
official OpenArm 2.0 frame, solving temporally smooth 7-DoF IK, filtering infeasible motion,
and exporting LeRobot v3 datasets. Image-space embodiment replacement is included as a
separate, auditable stage.

The target pose is always the origin of `openarm_{right,left}_ee_base_link` relative to the
root of the official OpenArm 2.0 bimanual MuJoCo model. Positions are metres and quaternions
are `xyzw`. Exported joint vectors match the official OpenArm dataset ordering:
right joints 1-7, right gripper, left joints 1-7, left gripper.

Automatic conversions use one shared source-to-OpenArm base transform for both arms. They are
fully reproducible and suitable for inspection/training experiments, but remain explicitly
`calibration_validated: false` until physical base and flange correspondences are measured.
The exporter never silently promotes automatic workspace registration to metrological truth.

## Quick start

```bash
uv sync --extra dev --extra media-ai --extra video-inpaint --extra robotseg
uv run openarm-retarget fetch-model
uv run openarm-retarget inspect-source configs/sources/molmoact2_tabletop.yaml
uv run pytest
```

The complete coordinate contract, dataset status, workflow, validation results, rendering
benchmarks, and embodiment-transfer decisions are condensed into
[docs/PROJECT.md](docs/PROJECT.md). Raw data and generated media are intentionally Git-ignored.
