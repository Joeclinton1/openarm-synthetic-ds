# Project guide

This file is the operational contract, workflow, current status, and design rationale for the
OpenArm 2.0 retargeting pipeline. Generated datasets and audit artifacts live under Git-ignored
`data/` and `outputs/`.

## Status

| Area | State | Boundary |
|---|---|---|
| Four one-hour samples | Complete | Selected whole episodes exceed 3,600 s per source. |
| Cartesian conversion | Complete computationally | Poses use one shared OpenArm base frame and normalized `xyzw` quaternions. |
| 7-DoF IK and filtering | Complete computationally | Limits, collision, residual, conditioning, velocity, and acceleration are audited. |
| LeRobot v3 export | Complete | Feasible contiguous runs are exported without compressing invalid gaps. |
| MuJoCo inspection | Complete | Fast geometry and segmentation reference. |
| Robot removal | Accepted on the 1,026-frame fixture | RobotSeg/Grounding-DINO+SAM2, chunked ProPainter, and protected-object restoration. |
| OpenArm rendering | Accepted on the fixture | Cycles is the reference; EEVEE Next is the production preset. |
| Style refinement | Accepted behind a hard gate | VACE may change robot RGB only; deterministic compositing is the fallback. |
| Physical calibration | Requires hardware | Online priors cannot establish measured source-base and flange/TCP transforms. |

Everything that can be validated without physical source/OpenArm measurements is implemented.
Outputs intentionally retain `calibration_validated: false` and
`physical_release_ready: false` until that final metrology is performed.

## Dataset snapshot

| Source | Local selection | Tabletop | Pose conversion | Main limitation | License |
|---|---:|---:|---:|---|---|
| HIW-500 | 49 ep / 108,335 frames / 3,611.17 s | 3/5 | 4/5 | Hanger manipulation is useful, but the dataset is mostly mobile household work and its Cartesian endpoint link is unnamed. | CC BY 4.0 |
| UnifoLM WBT | 78 ep / 109,134 frames / 3,637.80 s | 3/5 | 4/5 | Dishwasher interaction is not conventional tabletop work; the recorded FK endpoint is ambiguous. | Apache 2.0 |
| AgiBot World Alpha | 110 ep / 107,649 frames / 3,622.75 s | 4/5 | 5/5 | Direct bimanual flange poses, but access is gated and commercial use is restricted. | CC BY-NC-SA 4.0 |
| MolmoAct2 Tabletop | 590 ep / 72,080 frames / 3,604.00 s | 5/5 | 4/5 | Strongest tabletop source; the Franka controller/TCP frame is not published. | Apache 2.0 |

Repository mapping is pinned in `configs/sources/`: BitRobot's HIW LeRobot mirror, Unitree's
`G1_WBT_Brainco_Collect_Plates_Into_Dishwasher`, AllenAI's MolmoAct2 Tabletop member, and the
original gated AgiBot World Alpha archives. The GT-111 AgiBot fallback remains supported in code
but is not part of the primary data or local sample.

The three LeRobot acquisitions pass manifest row, duration, size, and checksum validation. The
AgiBot selection was reconciled against authoritative HDF5 timestamps and all 110 selected
episodes retain proprioception, parameters, and observations. Its three coarse 45--46 GB source
archives were removed after extraction; their revisions and SHA-256 values remain recorded, but
offline archive re-verification now requires re-downloading them.

## Coordinate and calibration contract

`Episode.ee_pose` is `[time, side, 7]`, with side `0=right`, `1=left`:

```text
[x, y, z, qx, qy, qz, qw]
```

Positions are metres. Quaternions are active, normalized, sign-continuous `xyzw` rotations. The
pose is from the official bimanual model root to `openarm_{side}_ee_base_link`, also exposed as
`{side}_ee_control_point`; it is not an unspecified wrist or fingertip pose. The model is pinned
to OpenArm MuJoCo commit `9eadf86d5b9a0713fdc097019302e02e4b303083`.

For a published source pose, conversion is:

```text
openarm_base_from_openarm_tool =
    openarm_base_from_source_base
  * source_base_from_source_tool
  * source_tool_from_openarm_tool
```

Both arms must use the same base transform. Per-side tool transforms are allowed because source
flange conventions can differ. Automatic registration aligns the bimanual midpoint and lateral
axis with one transform; it never translates the arms independently.

Calibration levels are:

- `measured`: physical correspondences pass the residual gate; the only level allowed to set
  `calibrated: true`.
- `cad_prior`: a pinned public model gives a defensible tool transform, but hardware identity is
  unverified.
- `workspace_fit`: visualization-only automatic alignment.

AgiBot has the strongest online prior. Its archived state is documented as left/right flange
pose, and the pinned G1_120s asset supplies the asymmetric gripper-centre transforms used in
`agibot.yaml`. OpenArm's neutral contact centre is approximately
`[-0.0014, 0, -0.16334]` m in the flange. This remains `cad_prior`. Unitree publishes a BrainCo
hand mount, but UnifoLM does not identify whether `ee_state` ends at wrist, hand base, or palm.
HIW does not name its WBC Cartesian endpoint. MolmoAct2 identifies a fixed Franka but not the
controller TCP. Those three therefore remain `workspace_fit`.

To promote a source to `measured`, collect at least three non-collinear source/OpenArm base
correspondences, verify the exact flange/TCP links, run `calibrate-points`, and validate camera
overlay against a target visible in source RGB.

## IK, feasibility, and export

Output joints follow the official OpenArm order:

```text
right_joint1..7, right_gripper, left_joint1..7, left_gripper
```

Normalized gripper state maps from `0=open, 1=closed` to right `0..-0.7854 rad` and left
`0..+0.7854 rad`. Damped least squares solves weighted Cartesian position/orientation error.
The redundant null space follows the constant-velocity prediction `2q[t-1]-q[t-2]` to reduce
joint curvature; a smaller joint-centre objective avoids limits and warm starts retain the
solution branch.

Frames are rejected on Cartesian residual, Jacobian condition, joint limits, velocity,
acceleration, or bimanual collision. Short feasible islands are removed. Export splits remaining
contiguous runs into independent LeRobot episodes.

## Reproducible workflow

Install and fetch the pinned model:

```bash
uv sync --extra dev --extra media-ai --extra video-inpaint --extra robotseg --extra style-ai
uv run openarm-retarget fetch-model
uv run pytest
```

Acquire and verify the LeRobot sources:

```bash
uv run openarm-retarget download-hour configs/sources/hiw500.yaml
uv run openarm-retarget download-hour configs/sources/unifolm.yaml
uv run openarm-retarget download-hour configs/sources/molmoact2_tabletop.yaml
uv run openarm-retarget verify-download PATH/TO/sample_manifest.json
```

Selection keeps the shortest whole-episode prefix of at least 3,600 seconds and records the
revision, episode IDs, exact rows, media intervals, and checksums. AgiBot requires accepted
Hugging Face terms and `HF_TOKEN`:

```bash
uv run openarm-retarget plan-agibot-hour task_410.json \
  data/samples/agibot-world__AgiBotWorld-Alpha --task-id 410
uv run openarm-retarget download-agibot-hour \
  data/samples/agibot-world__AgiBotWorld-Alpha/sample_manifest.json
uv run openarm-retarget extract-agibot-hour \
  data/samples/agibot-world__AgiBotWorld-Alpha/sample_manifest.json
uv run openarm-retarget convert-agibot-hour configs/sources/agibot.yaml \
  data/samples/agibot-world__AgiBotWorld-Alpha/sample_manifest.json \
  data/converted/agibot_openarm
```

Calibrate and convert a normal LeRobot episode:

```bash
uv run openarm-retarget calibrate-points correspondences.json calibration.json
uv run openarm-retarget convert-episode CONFIG SAMPLE_PARQUET canonical.npz \
  --episode-index 0 --calibration calibration.json
uv run openarm-retarget solve canonical.npz solved.npz
uv run openarm-retarget render solved.npz preview.mp4
uv run openarm-retarget export solved.npz converted_dataset/
```

For inspection only, `convert-episode --allow-uncalibrated` followed by `fit-workspace` creates a
reproducible bootstrap; it is never promoted to measured calibration.

## Visual embodiment replacement

The accepted path keeps geometry separate from generative appearance:

```text
source -> robot/object masks -> ProPainter removal
solved joints + source camera -> transparent OpenArm RGBA/depth
clean scene + render + object occlusion -> deterministic composite
deterministic composite -> optional mask-constrained VACE -> validation or fallback
```

Operational sequence:

1. Segment with RobotSeg for high-precision robot masks, or bounded-memory Grounding-DINO +
   SAM2.1 via `segment-robot`. Track manipulated objects separately.
2. Run official ProPainter through `inpaint-propainter`; long videos use overlapping windows.
   Use `restore-protected --exclude-masks` so restoration cannot paste the source gripper back.
3. Map frame-aligned camera calibration through the same OpenArm base registration. AgiBot's
   stored intrinsics are scaled explicitly when calibration and video resolutions differ.
4. Export official meshes and FK with `blender-scene`. Render resumable transparent RGBA and
   optional metric depth with `render-blender-batch`, then run `validate-render` and
   `validate-render-depth`. Apply source lens distortion only after projection validation.
5. Composite with protected-object masks. Occlusion and renderer alpha own geometry.
6. Optionally run `harmonize-render`, or generate VACE windows with `style-vace-batch`. Apply
   `constrain-style-batch`, independently segment the result, and accept only clips passing
   `validate-style-refinement`; otherwise retain the deterministic composite.

Gripper-object contact remains the hardest image region because pixels hidden by the source arm
were never observed. Protected-object masks and depth are therefore release requirements, not
optional polish.

## Executed acceptance

All 397,198 input frames were converted and audited; 366,864 are feasible:

| Source | Feasible frames | Fraction | LeRobot v3 |
|---|---:|---:|---|
| HIW-500 | 93,305 | 86.13% | valid |
| UnifoLM WBT | 102,405 | 93.83% | valid |
| MolmoAct2 Tabletop | 66,298 | 91.98% | valid |
| AgiBot World Alpha | 104,856 | 97.41% | valid |

The full visual fixture is AgiBot episode `649684` at 1,026 frames and 640x480:

| Gate | Accepted result |
|---|---|
| Removal masks | Zero empty frames; 1.0 raw-mask recall; 0.9885 p05 temporal IoU. |
| ProPainter | 159.7 s including extraction/stitching; chunk boundaries below normal p95 frame change. |
| Cycles geometry | 0.9723 mean / 0.9477 p05 MuJoCo coverage agreement; zero depth leakage. |
| EEVEE geometry | 0.9595 mean IoU against Cycles; deterministic decoded pixels. |
| Deterministic composite | 0.00836 outside-region MAE; 0.00821 protected-object MAE. |
| VACE refinement | 0.9057 independent RobotSeg IoU; 0.00837 background MAE; 0.00800 protected-object MAE; 0.0839 px flow error. |

At 640x480, EEVEE Next/16 samples measured 9.99 output fps; two-process EEVEE was slower.
Cycles/8 samples with two OptiX workers measured about 2.01 fps. The corrected full Cycles
RGBA+depth run completed in 487.3 seconds. EEVEE would render the 397,198 sampled frames in about
11 hours; Cycles would require about 55 hours on the measured two-RTX-4090 workstation.

## Design decisions and research

Blender remains the accepted renderer: Cycles is the geometry oracle and EEVEE the fast path.
SAPIEN 3.0.3 is the most credible future high-throughput backend: its complete disposable
environment measured 389 MB and a simple 640x480 smoke scene reached 440 fps raster / 295 fps at
8-spp ray tracing, but those are not OpenArm production numbers. Unreal/URLab remains optional
because URLab has no cooked binary and needs a full editor once; Isaac Sim documents roughly
50 GB minimum storage. RoboTwin uses SAPIEN, so its large asset and policy stack is unnecessary.

The visual architecture follows geometry-preserving embodiment work:

| Work | Relevant conclusion |
|---|---|
| [EgoEngine](https://arxiv.org/abs/2606.12604), [Mirage](https://arxiv.org/abs/2402.19249) | Removal, calibrated render, and object-aware compositing provide the strongest contact/geometry guarantees. |
| [Masquerade](https://arxiv.org/abs/2508.09976), [H2R](https://arxiv.org/abs/2505.11920) | Egocentric human-to-robot rendering is useful for policy data generation. |
| [H2R-Grounder](https://arxiv.org/abs/2512.09406) | Closest learned in-context approach; requires a target-specific LoRA trained on real target-robot video. |
| [EgoDemoGen](https://arxiv.org/abs/2509.22578), [RoVi-Aug](https://rovi-aug.github.io/) | Learned translators improve appearance but require substantial target data or pair-specific training. |
| [VACE](https://github.com/ali-vilab/VACE) | Locally feasible temporal editor, but raw output changes preserved pixels and must remain constrained and rejectable. |

Consequently, calibrated render-and-composite is the accepted output. VACE is only a bounded RGB
refinement beneath exact robot alpha and outside protected objects. A future OpenArm-specific
H2R-Grounder-style LoRA should be trained only after real OpenArm video is available; training it
solely on synthetic renders would preserve the synthetic domain rather than close the gap.

Primary implementation references: [OpenArm MuJoCo](https://github.com/enactic/openarm_mujoco),
[RobotSeg](https://github.com/showlab/RobotSeg), [ProPainter](https://github.com/sczhou/ProPainter),
[SAPIEN](https://sapien-sim.github.io/docs/), and
[UnrealRoboticsLab](https://urlab-sim.github.io/UnrealRoboticsLab/).
