# Cross-dataset OpenArm video benchmark

The benchmark scripts construct the eight-clip visual evaluation under
`outputs/cross_dataset_openarm_benchmark`. Generated videos, masks, renders, model weights, and
downloaded source datasets remain Git-ignored. The scripts and manifests are the reproducible
record committed to the repository.

## Scope

The fixed selection contains two six-second demonstrations from each locally sampled dataset:
AgiBot World Alpha, HIW-500, MolmoAct tabletop, and UnifoLM WBT. HIW and MolmoAct supply two task
labels. The available AgiBot and UnifoLM subsets supply two demonstrations of one task; the
manifest describes them as demonstrations rather than distinct upstream tasks.

Source camera calibration is not sufficiently consistent across the four samples. The resulting
render registration is therefore an explicit visual baseline and must not be used as metric
image-space supervision.

## Pipeline

1. `build_cross_dataset_benchmark.py` selects the highest-motion feasible six-second window and
   writes the source video, sliced trajectory, and clip manifest.
2. RobotSeg, SAM2, and optional manually seeded tracks populate the mask directories described by
   `prepare_benchmark_masks.py`. Run that script to fuse the model results. An accepted AgiBot
   full-episode mask sequence is optional and must be supplied explicitly:

   ```bash
   uv run python scripts/prepare_benchmark_masks.py \
     --accepted-agibot-fixture /path/to/accepted/full_episode_masks \
     --accepted-agibot-start-frame 805
   ```

   Without the option, the portable RobotSeg and appearance-prior path is used for both AgiBot
   clips. No temporary-directory dependency is assumed.
3. Fixed-camera clips use `openarm-retarget inpaint-static-camera`. Moving-camera clips use
   `run_minimax_remover.py`; the script requires the upstream MiniMax-Remover implementation under
   `vendor/MiniMax-Remover` and, by default, model weights in the local Hugging Face cache. Pass
   `--allow-download` for the initial model fetch and `--gpu-id` to select a device.
4. `prepare_benchmark_scenes.py` exports EEVEE render scenes. Render the generated scenes into
   `render_raw`, and for bimanual clips also render the per-side scenes into `render_raw_left` and
   `render_raw_right`.
5. `align_benchmark_renders.py` performs the documented uncalibrated mask registration.
6. Composite the aligned RGBA frames over `02_robot_removed.mp4`, retaining source pixels outside
   target alpha, then run `evaluate_cross_dataset_benchmark.py`.

The benchmark metrics cover removal background error, temporal background error, unchanged-scene
error after compositing, runtime, inverse-kinematics feasibility, and pose error. They are diagnostic
metrics for this benchmark, not VBench scores.
