# Project guide

This is the operational contract for the OpenArm 2.0 retargeting pipeline. Raw datasets and
generated artifacts live under Git-ignored `data/` and `outputs/` directories.

## Supported sources

| Source | Adapter | Retained benchmark tasks | Calibration status |
|---|---|---|---|
| AgiBot World Alpha | Native HDF5 archives | Two water-pouring demonstrations | CAD-informed tool transform and recorded head camera; not physically calibrated |
| MolmoAct2 Tabletop | LeRobot | Close box; flip mug upright | Shared Franka/OpenArm transform and audited fixed-camera fit; not physically calibrated |
| DROID | LeRobot v3, Franka only | Deterministic one-hour slice, one external RGB view | Automatic shared-frame registration; not physically calibrated |
| RH20T | LeRobot v3, cfg5 Franka only | Deterministic one-hour slice, one front RGB view | Absolute TCP state; not physically calibrated |
| RoboMIND | LeRobot v3, AgileX 3RGB only | Deterministic one-hour bimanual slice, all three RGB views | Automatic shared-frame registration; not physically calibrated |

The source definitions are in `configs/sources/`. The Molmo camera fit is in
`configs/cameras/molmoact2_tabletop_fitted.json`. Dataset access, use, and redistribution remain
subject to each upstream license.

## Canonical representation

`Episode.ee_pose` has shape `[time, side, 7]`, where side `0=right` and `1=left`:

```text
[x, y, z, qx, qy, qz, qw]
```

Positions are metres. Quaternions are active, normalized, sign-continuous `xyzw` rotations. The
pose is from the official bimanual model root to `openarm_{side}_ee_base_link`. The model is pinned
to OpenArm MuJoCo commit `9eadf86d5b9a0713fdc097019302e02e4b303083`.

Conversion follows:

```text
openarm_base_from_openarm_tool =
    openarm_base_from_source_base
  * source_base_from_source_tool
  * source_tool_from_openarm_tool
```

Both arms use one shared base transform. A source may have separate left and right tool transforms.
The pipeline never independently recentres the arms.

Every automatic or image-fitted transform remains `calibration_validated: false`. Promotion to a
measured calibration requires at least three non-collinear physical correspondences, verified
flange/TCP links, and a camera-overlay check against a visible target.

## IK and gripper geometry

The damped-least-squares solver tracks Cartesian position and orientation while using the redundant
null space for temporal continuity and joint-centre preference. Warm starts retain the solution
branch. Frames are audited for residuals, conditioning, limits, velocity, acceleration, collision,
gripper aperture, and pinch-midpoint error.

Output joint order is:

```text
right_joint1..7, right_gripper, left_joint1..7, left_gripper
```

Normalized gripper state is `0=open, 1=closed`. The official MJCF is the geometry authority for
finger angles, fingertip aperture, and the moving pinch midpoint. Feasible contiguous runs are
exported as separate LeRobot episodes; invalid gaps are not compressed.

## Installation

```bash
uv sync --extra dev
uv run openarm-retarget fetch-model
uv run pytest
```

Install only the optional GPU path you intend to run:

```bash
uv sync --extra media-ai --extra robotseg
uv sync --extra minimax
```

The external Blender binary and learned model repositories are not vendored in the package or
source distribution.

## Acquisition and conversion

Inspect or acquire the public sources with:

```bash
uv run openarm-retarget inspect-source configs/sources/droid.yaml
uv run openarm-retarget plan-hour configs/sources/droid.yaml
uv run openarm-retarget download-hour configs/sources/droid.yaml --destination data/samples
uv run openarm-retarget convert-hour configs/sources/droid.yaml \
  data/samples/lerobot__droid_1.0.1/sample_manifest.json \
  data/samples/lerobot__droid_1.0.1/sample/data.parquet data/converted/droid_openarm
```

MolmoAct2 uses the same LeRobot workflow. AgiBot uses `plan-agibot-hour`,
`download-agibot-hour`, `extract-agibot-hour`, and `convert-agibot-hour` because its original data
is distributed as gated archives. `scripts/acquire_one_hour.py` records the fixed local acquisition
plan for all supported LeRobot sources and audits access to the original AgiBot repository.

DROID, RH20T Franka cfg5, and RoboMIND AgileX 3RGB use the same LeRobot commands. Their source
configs pin exactly one embodiment and a minimal camera selection. Every plan and download has a
20,000,000,000-byte ceiling by default; planning fails before trajectory/video containers are
downloaded if the selected slice would exceed it:

```bash
uv run openarm-retarget plan-hour configs/sources/droid.yaml
uv run openarm-retarget download-hour configs/sources/droid.yaml --destination data/samples
uv run openarm-retarget convert-hour configs/sources/droid.yaml \
  data/samples/lerobot__droid_1.0.1/sample_manifest.json \
  data/samples/lerobot__droid_1.0.1/sample/data.parquet data/converted/droid_openarm \
  --workers 8
```

Use `rh20t_franka.yaml` or `robomind_agilex_3rgb.yaml` identically. DROID downloads one exterior
view, RH20T downloads one front view, and the explicitly 3RGB RoboMIND subset retains front plus
both wrist views. `--max-bytes` may lower, but not silently bypass, the acquisition ceiling.

Audit converted slices together with:

```bash
uv run openarm-retarget audit-all \
  data/converted/droid_openarm data/converted/rh20t_franka_openarm \
  data/converted/robomind_agilex_openarm --output data/converted/three_dataset_audit.json
```

For a single converted episode:

```bash
uv run openarm-retarget solve canonical.npz solved.npz
uv run openarm-retarget validate-gripper-contact solved.npz --output contact.json
uv run openarm-retarget export solved.npz exported_lerobot
uv run openarm-retarget validate-export exported_lerobot
```

## Visual replacement pipeline

The accepted output path is deterministic:

```text
source video -> robot masks -> robot removal -> Blender RGBA/depth
             -> camera or audited image registration -> composite -> validation
```

RobotSeg, optional SAM2 tracks, and gripper segmentation provide mask evidence. Static-camera clips use a clean-plate
estimator; moving-camera clips use MiniMax removal with background preservation. Protected objects
can be restored only outside the source-robot exclusion mask. Blender EEVEE is the production
renderer and Cycles is available for higher-quality reference frames.

The validator checks frame parity, mask coverage, unchanged background, temporal background error,
depth ordering, render alignment, embodiment alignment, and OpenArm kinematics. The Molmo
fixed-camera fit has 5.13 px median and 9.63 px p90 reprojection error across 240 audited
correspondences, but is still labelled inspection-grade.

## Publication boundary

The repository is reproducible code and configuration, not a redistribution of upstream data.
Before publishing generated datasets:

1. Confirm each upstream dataset's current license and access terms.
2. Replace automatic transforms with physical base/tool calibration where metric accuracy matters.
3. Record camera calibration for any image-space supervision claim.
4. Run `audit-all`, export validation, and the benchmark evaluator on the exact release artifacts.
5. Preserve source revision, config, model revision, and validation manifests with the release.

The current benchmark demonstrates computational retargeting and visual plausibility. It does not
claim metrologically calibrated cross-robot ground truth.
