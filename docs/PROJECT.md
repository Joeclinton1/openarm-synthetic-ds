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
| Gripper contact geometry | Complete computationally | One official-MJCF mapping drives MuJoCo, Blender, URLab, and export; physical pinch registration still inherits each source's calibration level. |
| LeRobot v3 export | Complete | Feasible contiguous runs are exported without compressing invalid gaps. |
| MuJoCo inspection | Complete | Fast geometry and segmentation reference. |
| Robot removal | Accepted on the 1,026-frame fixture | RobotSeg/Grounding-DINO+SAM2, chunked ProPainter, and protected-object restoration. |
| OpenArm rendering | Accepted on the fixture | Cycles is the reference; EEVEE Next is the production preset. |
| Unreal/URLab rendering | Integration implemented, acceptance pending | UE 5.7, job-v2, synchronized RGB/depth/instance capture, cooked runtime, and resumable shards are opt-in until the full fixture passes. |
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

Normalized gripper state is `0=open, 1=closed`: open maps to right `-0.7854 rad` and left
`+0.7854 rad`, while closed maps both sides to `0 rad`. This mapping is defined once and consumed
by MuJoCo, Blender/EEVEE/Cycles, URLab, and LeRobot export. When a source publishes physical jaw
width, it is retained in metres and inverted through the pinned OpenArm finger geometry rather
than percentile-normalized per episode. OpenArm's rotating fingers move their pinch midpoint by
up to 15.2 mm over the opening range, so pinch-centre-calibrated sources also receive a local
EE-base compensation that keeps the intended pad midpoint fixed.

Damped least squares solves weighted Cartesian position/orientation error.
The redundant null space follows the constant-velocity prediction `2q[t-1]-q[t-2]` to reduce
joint curvature; a smaller joint-centre objective avoids limits and warm starts retain the
solution branch.

Frames are rejected on Cartesian residual, Jacobian condition, joint limits, velocity,
acceleration, bimanual collision, fingertip aperture error, or pinch-midpoint error. Short
feasible islands are removed. Export splits remaining contiguous runs into independent LeRobot
episodes. Run the independent geometry gate with:

```bash
uv run openarm-retarget validate-gripper-contact solved.npz --output contact.json
```

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

The published Dora/OpenArm MuJoCo+Mink IK remains an opt-in comparison backend. Its Dora
interface and `openarm-control` implementation are pinned independently. Generate quantitative
results and synchronized human-review videos without replacing the production solver:

```bash
uv sync --extra dev --extra official-ik
uv run openarm-retarget compare-ik canonical.npz outputs/ik-review
```

`compare-ik` uses the official solver's published costs and DAQP backend, with 80 iterations per
offline frame and bounded repeated-target warm-up matching Dora's live control cadence. Both
outputs receive the same feasibility gates. Mink output is intentionally left unsmoothed: it can
ride a model joint limit, and the current DLS smoother's inward limit margin changes that valid
redundant solution's tool orientation. On a headless host, set `MUJOCO_GL=egl` when generating
review videos.
Removal of the current solver requires complete cross-dataset validation, not a single fixture.

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

1. Segment with bounded-memory Grounding-DINO + SAM2.1 via `segment-robot`. Robot tracks are
   carried across chunk boundaries from edge-connected components; independent re-detection at
   every boundary can confuse a manipulated object with the arm. Run RobotSeg's `robot` and
   `gripper` categories separately, then use `fuse-robot-gripper-masks`. Gripper components are
   accepted only near the primary arm track, improving small-finger recall without accepting
   disconnected prompt drift. Track manipulated objects separately.
2. For fixed cameras, `inpaint-static-camera` constructs a mask-aware temporal median clean plate
   and reports held-out plate error. Pixels never exposed can use a neural result as a fallback;
   the fallback is restricted to missing or strongly disagreeing plate pixels. For scenes without
   a usable clean plate, run official ProPainter through `inpaint-propainter`; long videos use
   overlapping windows. Use the tighter RobotSeg mask with `restore-protected --exclude-masks` so
   visible object pixels return without pasting the source gripper back.
3. Map frame-aligned camera calibration through the same OpenArm base registration. AgiBot's
   stored intrinsics are scaled explicitly when calibration and video resolutions differ.
4. Export official meshes and FK with `blender-scene`. Render resumable transparent RGBA and
   optional metric depth with `render-blender-batch`, then run `validate-render` and
   `validate-render-depth`. The Blender driver supports aimed, colour-temperature-controlled area
   lights and an optional equirectangular HDR probe. Attach a recovered probe without changing any
   geometry using `configure-blender-hdri`. Apply source lens distortion only after projection
   validation.
5. Run `calibrate-render-lighting` against the source robot track. It fits one bounded
   episode-global tone and white-balance transform in linear light, preserving alpha exactly and
   avoiding temporal pumping. Require a measurable improvement with `validate-render-lighting`.
6. Composite with protected-object masks. Occlusion and renderer alpha own geometry; alpha-over
   is performed in linear light by default. Retained source contact shading is not duplicated with
   an unconditional synthetic shadow catcher.
7. `harmonize-render` remains a weaker local-context fallback. A generative VACE pass is optional:
   constrain it with `constrain-style-batch`, independently segment the result, and accept only
   clips passing `validate-style-refinement`; otherwise retain the deterministic composite.

Gripper-object contact remains the hardest image region because pixels hidden by the source arm
were never observed. Protected-object masks and depth are therefore release requirements, not
optional polish.

### Unreal/URLab candidate backend

Blender remains the default and Cycles remains the oracle. The Unreal candidate is pinned to
URLab `567cbd907a570b820beb87fbddd69c356a6d86da`; its typed bridge is pinned to the matching
remote-stepping revision `bd3a63b15c6430b0b3738a0f5876554d5408644a`. Install Epic's precompiled
UE 5.7 build (not a source checkout), then:

```bash
export UE_ROOT=/path/to/UnrealEngine-5.7
scripts/setup_urlab.sh
scripts/import_urlab_asset.sh       # one persistent 91 MB model import
scripts/package_urlab.sh /new/runtime/path
```

Export, validate, render, and compare a v2 job:

```bash
uv run openarm-retarget urlab-job solved.npz outputs/job camera.json
uv run openarm-retarget validate-urlab-job outputs/job/urlab_job.json
uv run openarm-retarget render-urlab-batch outputs/job/urlab_job.json outputs/unreal \
  --runtime /path/to/OpenArmRenderer --gpu-ids 0,1 --shard-frames 256 --warmup-frames 8
uv run openarm-retarget render-urlab outputs/job/urlab_job.json outputs/unreal-audit \
  --output-mode audit
uv run openarm-retarget validate-urlab outputs/unreal-audit/rgba outputs/cycles \
  --mujoco-rgba outputs/mujoco
```

Each cooked worker uses a distinct URLab step port; shard assignment is stable across resume.
Every shard renders preceding warm-up frames and discards them, writes atomically, and records
checksums, timing, synchronous frame IDs, settings, and failures. Instance segmentation is the
only alpha source. Unreal centimetre depth is converted once to metres. Lens distortion remains
a backend-independent post-process. Batch production output uses bounded writer queues and direct
lossless FFmpeg streams (`rgba.mkv`, `instance.mkv`, and 16-bit millimetre `depth_mm.mkv`);
`--output-mode audit` is the explicit PNG/NPZ path used by validators.

Promotion requires static poses, 10 and 100 moving frames, the complete 1,026-frame fixture, one
hour per source, and full retained data. Hard gates include mean silhouette IoU >= 0.95 with a
strong p05, approximately one-pixel projection agreement, zero alpha/depth leakage, accepted
metric depth tolerance, exact pose/frame synchronization, stable shard boundaries, bounded
repeat renders, existing protected-object/composite audits, end-to-end throughput, and human
review. Until all are recorded, `unreal-lumen` is experimental and Blender remains default.

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
| Removal masks | Carried SAM2 track plus proximity-gated RobotSeg gripper fusion; accepted gripper recall increased from 0.9664 to 1.0 while 685 disconnected components were rejected. |
| Removal candidate | The old result failed the current residual-copy gate (0.3086 p95). The corrected full fixture passes at 0.0183 p95 and 0.00619 mean copied-source fraction. Independent RobotSeg residual-mask area fell 64.0% (0.01416 to 0.00509 mean). |
| Static clean plate | 25.3 s for 1,026 frames at 640x480; 93.37% of pixels had at least three clean observations. A neural fallback was needed for the permanently occluded 6.57%. |
| Cycles geometry | 0.9723 mean / 0.9477 p05 MuJoCo coverage agreement; zero depth leakage. |
| EEVEE geometry | 0.9595 mean IoU against Cycles; deterministic decoded pixels. |
| Gripper contact geometry (corrected replay) | 1,018/1,026 retained frames; 6.91 mm p95 / 9.73 mm maximum pinch-midpoint error; physical jaw aperture error below 1.3e-16 m. |
| Deterministic composite | 0.00836 outside-region MAE; 0.00821 protected-object MAE. |
| Source-calibrated HDRI lighting | A DiffusionLight Turbo probe plus bounded linear-light calibration reduced the active-arm photometric score from 0.10204 to 0.01060 (89.6%); silhouette IoU and post-calibration alpha were exactly 1.0 across all 1,026 frames. |
| VACE refinement | 0.9057 independent RobotSeg IoU; 0.00837 background MAE; 0.00800 protected-object MAE; 0.0839 px flow error. |

The corrected gripper replay uses AgiBot's source-reported millimetre opening and the fixture's
existing automatic workspace registration. It proves internal kinematic/render consistency, not
physical source-to-OpenArm metrology; that registration remains explicitly unvalidated. Eight
frames over the hard 10 mm contact limit are now rejected instead of being rendered silently.
The same corrected replay through the pinned official Mink option retained 751 frames and passed
the contact gate at 1.40 mm p95 / 2.87 mm maximum; its previously documented joint-limit and
temporal failures still prevent it replacing the current solver.

The pinned published Dora/OpenArm Mink IK was also run on all 1,026 fixture frames with its
released DAQP costs, 80 offline iterations, and bounded first-target warm-up:

| IK result | Current DLS | Official Mink |
|---|---:|---:|
| Feasible frames | 100.00% | 73.68% |
| Mean / maximum position error | 2.973 / 4.809 mm | 0.842 / 3.828 mm |
| Maximum orientation error | 0.06465 rad | 0.000252 rad |
| Maximum joint acceleration | 9.37 rad/s² | 38.82 rad/s² |
| Joint-limit violation frames | 0 | 262 |
| Solve throughput | 750 fps | 43 fps |

Mink is more Cartesian-accurate, but `openarm-control` 0.1.0 retries an infeasible constrained QP
without configuration limits. On this fixture that drives the left wrist up to 0.170 rad outside
the audited model range. The current solver remains the production implementation; the official
solver is retained only as a pinned comparison backend until upstream limit handling and temporal
gates pass.

At 640x480, EEVEE Next/16 samples measured 9.99 output fps; two-process EEVEE was slower.
Cycles/8 samples with two OptiX workers measured about 2.01 fps. The corrected full Cycles
RGBA+depth run completed in 487.3 seconds. EEVEE would render the 397,198 sampled frames in about
11 hours; Cycles would require about 55 hours on the measured two-RTX-4090 workstation.

## Design decisions and research

Blender remains the accepted renderer: Cycles is the geometry oracle and EEVEE the fast path.
SAPIEN 3.0.3 is the most credible future high-throughput backend: its complete disposable
environment measured 389 MB and a simple 640x480 smoke scene reached 440 fps raster / 295 fps at
8-spp ray tracing, but those are not OpenArm production numbers. Unreal/URLab now has a local
cooked-runtime path but remains optional until the complete fixture passes; Isaac Sim documents roughly
50 GB minimum storage. RoboTwin uses SAPIEN, so its large asset and policy stack is unnecessary.

The visual architecture follows geometry-preserving embodiment work:

| Work | Relevant conclusion |
|---|---|
| [EgoEngine](https://arxiv.org/abs/2606.12604), [Mirage](https://arxiv.org/abs/2402.19249) | Removal, calibrated render, and object-aware compositing provide the strongest contact/geometry guarantees. |
| [Masquerade](https://arxiv.org/abs/2508.09976), [H2R](https://arxiv.org/abs/2505.11920) | Egocentric human-to-robot rendering is useful for policy data generation. |
| [H2R-Grounder](https://arxiv.org/abs/2512.09406) | Closest learned in-context approach; requires a target-specific LoRA trained on real target-robot video. |
| [EgoDemoGen](https://arxiv.org/abs/2509.22578), [RoVi-Aug](https://rovi-aug.github.io/) | Learned translators improve appearance but require substantial target data or pair-specific training. |
| [VACE](https://github.com/ali-vilab/VACE) | Locally feasible temporal editor, but raw output changes preserved pixels and must remain constrained and rejectable. |
| [Debevec image-based lighting](https://www.pauldebevec.com/Research/IBL/), [Blender colour management](https://docs.blender.org/manual/en/latest/render/color_management.html) | HDR environment lighting, explicit display transforms, and linear-light compositing are the deterministic CGI baseline. |
| [DiffusionLight Turbo](https://github.com/DiffusionLight/DiffusionLight-Turbo) | The full fixture uses its single-image, exposure-bracketed HDR probe for structured metal reflections; the fast source-calibration fallback remains usable without it. |

Consequently, calibrated render-and-composite is the accepted output. VACE is only a bounded RGB
refinement beneath exact robot alpha and outside protected objects. A future OpenArm-specific
H2R-Grounder-style LoRA should be trained only after real OpenArm video is available; training it
solely on synthetic renders would preserve the synthetic domain rather than close the gap.

Primary implementation references: [OpenArm MuJoCo](https://github.com/enactic/openarm_mujoco),
[RobotSeg](https://github.com/showlab/RobotSeg), [ProPainter](https://github.com/sczhou/ProPainter),
[SAPIEN](https://sapien-sim.github.io/docs/), and
[UnrealRoboticsLab](https://urlab-sim.github.io/UnrealRoboticsLab/).
