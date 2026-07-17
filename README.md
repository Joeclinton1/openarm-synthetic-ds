# OpenArm 2.0 dataset retargeting

Retarget Cartesian demonstrations from AgiBot World Alpha, MolmoAct2 Tabletop, DROID, RH20T
Franka, and RoboMIND AgileX 3RGB to the official OpenArm 2.0 bimanual model. The repository
contains the conversion, temporally smooth 7-DoF IK, feasibility checks, deterministic Blender
rendering, robot removal, compositing, and validation code used for the ten-clip cross-dataset
benchmark.

Generated datasets, source media, model weights, and videos are intentionally excluded from Git.

## Coordinate contract

Canonical end-effector poses are `[x, y, z, qx, qy, qz, qw]` in metres, expressed from the
OpenArm model root to `openarm_{right,left}_ee_base_link`. Joint exports follow:

```text
right joints 1-7, right gripper, left joints 1-7, left gripper
```

Gripper state is `0=open, 1=closed`. Both arms share one source-to-OpenArm base transform;
per-arm tool transforms handle different flange conventions. Automatic registrations remain
marked `calibration_validated: false` until physical base, tool, and camera correspondences are
measured.

## Quick start

Python 3.11+, `uv`, FFmpeg, MuJoCo, and Blender are required. GPU extras are only needed for
learned segmentation or moving-camera removal.

```bash
uv sync --extra dev
uv run openarm-retarget fetch-model
uv run openarm-retarget inspect-source configs/sources/molmoact2_tabletop.yaml
uv run pytest
```

Optional components:

```bash
uv sync --extra media-ai --extra robotseg  # SAM2/RobotSeg masks
uv sync --extra minimax                    # MiniMax moving-camera removal
```

Run `uv run openarm-retarget --help` for the conversion and rendering commands. See
[docs/PROJECT.md](docs/PROJECT.md) for the data and calibration workflow, and
[docs/CROSS_DATASET_BENCHMARK.md](docs/CROSS_DATASET_BENCHMARK.md) for the reproducible visual
evaluation.

## License

The code is Apache-2.0. Source datasets and upstream models retain their own licenses and access
terms; review them before redistributing generated data.
