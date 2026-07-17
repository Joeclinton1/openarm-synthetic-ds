# Cross-dataset OpenArm video benchmark

The benchmark builds six fixed six-second clips under
`outputs/cross_dataset_openarm_benchmark`: two each from AgiBot World Alpha, HIW-500, and MolmoAct2
Tabletop. Generated source video, masks, model weights, renders, and reports are Git-ignored.

## Fixed selection

| Dataset | Clips | Projection |
|---|---|---|
| AgiBot World Alpha | Two water-pouring demonstrations | Recorded moving head camera mapped through the shared OpenArm registration |
| HIW-500 | Hang hanger; hang keys on hook | Audited source-mask registration; not metric camera calibration |
| MolmoAct2 Tabletop | Close box; flip mug upright | One fixed inspection camera fit from 240 correspondences |

HIW uses one base/tool convention and a `0.8` workspace scale for both tasks. Molmo uses one shared
Franka/OpenArm transform, explicit tool-axis correction, and one camera fit for both tasks. No clip
is independently recentered in robot coordinates.

## Reproduction

1. Build source clips and sliced trajectories:

   ```bash
   uv run python scripts/build_cross_dataset_benchmark.py
   ```

2. Generate RobotSeg, optional SAM2 geometry tracks, and gripper masks using the CLI, then fuse the
   audited mask inputs:

   ```bash
   uv run python scripts/prepare_benchmark_masks.py
   ```

   An accepted full-episode AgiBot fixture may be supplied explicitly with
   `--accepted-agibot-fixture` and `--accepted-agibot-start-frame`. Without it, the portable
   RobotSeg and appearance-prior path is used.

3. Remove the source robot. Use `openarm-retarget inpaint-static-camera` for fixed-camera clips.
   Use `scripts/run_minimax_remover.py` for moving-camera clips; it expects the upstream
   implementation at `vendor/MiniMax-Remover`.

4. Export Blender scenes:

   ```bash
   uv run python scripts/prepare_benchmark_scenes.py
   ```

   Render the main scene to `render_raw`. Bimanual uncalibrated clips also use the generated
   per-side scenes in `render_raw_left` and `render_raw_right`.

5. Register renders to the audited image evidence:

   ```bash
   uv run python scripts/align_benchmark_renders.py
   ```

   AgiBot retains its camera-registered projection. Molmo translates the projected pinch centre to
   the audited gripper track at the fixed apparent scale without changing the fitted 3-D tool
   direction. HIW uses the documented image-space registration and remains explicitly uncalibrated.

6. Compose the output and regenerate the metrics:

   ```bash
   uv run python scripts/compose_benchmark_outputs.py
   uv run python scripts/evaluate_cross_dataset_benchmark.py
   ```

Each clip contains `01_source.mp4`, `02_robot_removed.mp4`, `03_openarm_output.mp4`, and the
labelled `source_removed_openarm.mp4` review triplet. AgiBot inserts only the moving OpenArm side and
restores the stationary source arm from the accepted fixture mask.

## Metrics and interpretation

The evaluator records frame parity, background preservation, removal-region change, temporal
background error, unchanged-scene composite error, alpha/mask overlap, tool-anchor image error,
runtime, IK feasibility, Cartesian residual, and orientation residual.

These are diagnostic benchmark metrics. They do not turn inspection-grade image registration into
physical calibration, and they are not a substitute for human review of every final triplet.
