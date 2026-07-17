from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import typer

from .adapters import load_agibot_h5, load_agibot_lerobot_episode, load_lerobot_episode
from .agibot_archive import (
    convert_agibot_hour,
    download_agibot_archives,
    extract_agibot_hour,
    plan_agibot_hour,
)
from .ai_masks import segment_robot_video
from .audit import audit_all
from .calibration import calibrate_from_file
from .batch import convert_lerobot_hour
from .camera import write_agibot_openarm_camera
from .download import download_lerobot_hour, plan_lerobot_hour, probe_repo, verify_download
from .export import export_lerobot_v3, validate_lerobot_v3
from .filters import filter_episode
from .ik import OpenArmIK
from .gripper_validation import validate_gripper_contact
from .media import (
    calibrate_rgba_photometry,
    composite_video,
    distort_rgba_frames,
    distort_depth_frames,
    harmonize_rgba_frames,
    fuse_robot_gripper_masks,
    inpaint_static_camera,
    inpaint_video,
    refine_protected_masks,
    refine_robot_masks,
    restore_protected_video,
    validate_inpainting,
    validate_harmonized_rgba,
    validate_photometric_calibration,
    validate_depth_render,
    validate_composite_video,
    validate_robot_masks,
    validate_rgba_render,
    validate_render_alignment,
    validate_embodiment_alignment,
)
from .model import fetch_openarm_model
from .presets import fit_workspace_translation, load_hiw_episode
from .raytrace import (
    configure_blender_environment,
    export_blender_scene,
    render_blender_batch,
    render_blender_scene,
)
from .robotseg import segment_robotseg_video
from .registration import auto_register_episode
from .schema import Episode, SourceConfig
from .viewer import TrajectoryViewer

app = typer.Typer(no_args_is_help=True, help="Retarget robot datasets to OpenArm 2.0")


@app.command("audit-all")
def audit_all_command(
    destinations: list[Path],
    output: Path | None = None,
    model: Path | None = None,
) -> None:
    report = audit_all(destinations, output, model)
    typer.echo(json.dumps(report, indent=2))
    if not report["ok"]:
        raise typer.Exit(code=1)


@app.command("fetch-model")
def fetch_model(destination: Path = Path("data/assets/openarm_mujoco")) -> None:
    typer.echo(fetch_openarm_model(destination))


@app.command("calibrate-points")
def calibrate_points(input_path: Path, output: Path) -> None:
    typer.echo(json.dumps(calibrate_from_file(input_path, output), indent=2))


@app.command("inspect-source")
def inspect_source(config_path: Path) -> None:
    config = SourceConfig.from_yaml(config_path)
    result = {"config": config.to_json(), "repository": probe_repo(config.repo_id)}
    typer.echo(json.dumps(result, indent=2))


@app.command("plan-hour")
def plan_hour(
    config_path: Path,
    destination: Path = Path("data/samples"),
    seconds: float = 3600,
) -> None:
    config = SourceConfig.from_yaml(config_path)
    if config.adapter == "agibot_h5":
        typer.echo(json.dumps(probe_repo(config.repo_id), indent=2))
        raise typer.Exit(code=2)
    manifest, _ = plan_lerobot_hour(
        config.repo_id, destination, seconds, config.tabletop_tasks, prefix=config.dataset_prefix
    )
    typer.echo(json.dumps(manifest, indent=2))


@app.command("download-hour")
def download_hour(
    config_path: Path,
    destination: Path = Path("data/samples"),
    seconds: float = 3600,
    camera: list[str] | None = typer.Option(None, help="Video feature to include; repeatable"),
    metadata_only: bool = False,
) -> None:
    config = SourceConfig.from_yaml(config_path)
    if config.adapter == "agibot_h5":
        typer.echo(json.dumps(probe_repo(config.repo_id), indent=2))
        raise typer.Exit(code=2)
    output = download_lerobot_hour(
        config.repo_id,
        destination,
        seconds,
        config.tabletop_tasks,
        cameras=camera,
        metadata_only=metadata_only,
        prefix=config.dataset_prefix,
    )
    typer.echo(output)


@app.command("plan-agibot-hour")
def plan_agibot_hour_command(
    task_json: Path,
    destination: Path = Path("data/samples/agibot-world__AgiBotWorld-Alpha"),
    task_id: int = 410,
    seconds: float = 3600,
) -> None:
    typer.echo(plan_agibot_hour(task_json, destination, task_id, seconds))


@app.command("download-agibot-hour")
def download_agibot_hour_command(manifest: Path) -> None:
    typer.echo(download_agibot_archives(manifest))


@app.command("extract-agibot-hour")
def extract_agibot_hour_command(manifest: Path) -> None:
    typer.echo(extract_agibot_hour(manifest))


@app.command("verify-download")
def verify_download_command(manifest: Path, rehash: bool = True) -> None:
    report = verify_download(manifest, rehash)
    typer.echo(json.dumps(report, indent=2))
    if not report["ok"]:
        raise typer.Exit(code=1)


@app.command("convert-episode")
def convert_episode(
    config_path: Path,
    source: Path,
    output: Path,
    episode_index: int = 0,
    allow_uncalibrated: bool = typer.Option(
        False, help="Inspection only; output is marked uncalibrated"
    ),
    calibration: Path | None = typer.Option(
        None, help="JSON containing validated rigid transforms"
    ),
    max_frames: int | None = typer.Option(None, help="Limit frames for a smoke-test conversion"),
    model: Path | None = None,
) -> None:
    config = SourceConfig.from_yaml(config_path)
    if calibration:
        values = json.loads(calibration.read_text())
        config = replace(
            config,
            openarm_from_source_base=values["openarm_from_source_base"],
            source_tool_from_openarm_tool=values["source_tool_from_openarm_tool"],
            position_scale=float(values.get("position_scale", config.position_scale)),
            calibrated=values.get("validated", False),
        )
    if config.adapter == "agibot_h5":
        episode = load_agibot_h5(source, config, allow_uncalibrated)
    elif config.adapter == "agibot_lerobot":
        episode = load_agibot_lerobot_episode(source, config, episode_index, allow_uncalibrated)
    elif config.name == "HIW-500":
        episode = load_hiw_episode(source, config, episode_index, allow_uncalibrated, model)
    else:
        episode = load_lerobot_episode(source, config, episode_index, allow_uncalibrated, model)
    if max_frames is not None:
        episode = episode.sliced(0, max_frames)
    episode.save(output)
    typer.echo(f"{output}: {len(episode.timestamp)} frames, {episode.duration:.2f}s")


@app.command("convert-hour")
def convert_hour(
    config_path: Path,
    sample_manifest: Path,
    sample_parquet: Path,
    destination: Path,
    model: Path | None = None,
    max_episodes: int | None = typer.Option(None, help="Acceptance-test subset"),
    resume: bool = True,
) -> None:
    config = SourceConfig.from_yaml(config_path)
    if config.adapter == "agibot_h5":
        raise typer.BadParameter("Use the original AgiBot archive pipeline for HDF5 sources")
    typer.echo(
        convert_lerobot_hour(
            config,
            sample_manifest,
            sample_parquet,
            destination,
            model,
            max_episodes,
            resume,
            progress=typer.echo,
        )
    )


@app.command("convert-agibot-hour")
def convert_agibot_hour_command(
    config_path: Path,
    sample_manifest: Path,
    destination: Path,
    model: Path | None = None,
    max_episodes: int | None = typer.Option(None, help="Acceptance-test subset"),
    resume: bool = True,
) -> None:
    config = SourceConfig.from_yaml(config_path)
    if config.adapter != "agibot_h5":
        raise typer.BadParameter("Expected an agibot_h5 source configuration")
    typer.echo(
        convert_agibot_hour(
            config,
            sample_manifest,
            destination,
            model,
            max_episodes,
            resume,
            progress=typer.echo,
        )
    )


@app.command("fit-workspace")
def fit_workspace(
    source: Path,
    output: Path,
    side: list[str] | None = typer.Option(None, help="Active arm; repeat for bimanual"),
    model: Path | None = None,
) -> None:
    values = fit_workspace_translation(
        Episode.load(source), model, tuple(side or ["right", "left"])
    )
    values["validated"] = False
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(values, indent=2) + "\n")
    typer.echo(output)


@app.command("auto-register")
def auto_register(
    source: Path,
    output: Path,
    model: Path | None = None,
    config_path: Path | None = None,
) -> None:
    config = SourceConfig.from_yaml(config_path) if config_path else None
    values = auto_register_episode(
        Episode.load(source),
        model,
        source_tool_from_openarm_tool=(
            config.source_tool_from_openarm_tool if config is not None else None
        ),
        openarm_from_source_base=(
            config.openarm_from_source_base if config is not None else None
        ),
        position_scale=(
            config.position_scale
            if config is not None and config.openarm_from_source_base is not None
            else None
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(values, indent=2) + "\n")
    typer.echo(output)


@app.command("solve")
def solve(source: Path, output: Path, model: Path | None = None, apply_filter: bool = True) -> None:
    episode = Episode.load(source)
    ik = OpenArmIK(model)
    ik.solve_episode(episode)
    if apply_filter:
        filter_episode(episode, ik)
    episode.save(output)
    feasible = (
        int(episode.feasible.sum()) if episode.feasible is not None else len(episode.timestamp)
    )
    typer.echo(f"{output}: {feasible}/{len(episode.timestamp)} feasible frames")


@app.command("view")
def view(source: Path, model: Path | None = None, realtime: bool = True) -> None:
    TrajectoryViewer(model).interactive(Episode.load(source), realtime)


@app.command("validate-gripper-contact")
def validate_gripper_contact_command(
    source: Path,
    output: Path | None = None,
    model: Path | None = None,
    maximum_pinch_error_m: float = 0.01,
    maximum_aperture_error_m: float = 1e-6,
) -> None:
    report = validate_gripper_contact(
        Episode.load(source),
        model,
        maximum_pinch_error_m=maximum_pinch_error_m,
        maximum_aperture_error_m=maximum_aperture_error_m,
    )
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
    typer.echo(json.dumps(report, indent=2))
    if not report["ok"]:
        raise typer.Exit(code=1)


@app.command("render")
def render(
    source: Path,
    output: Path,
    model: Path | None = None,
    transparent_frames: Path | None = None,
    depth_frames: Path | None = None,
    camera_json: Path | None = None,
    width: int = 960,
    height: int = 720,
    backend: str = typer.Option(
        "mujoco", help="mujoco, blender-eevee, or blender-cycles"
    ),
    blender: Path = Path("blender"),
    device: str = "CPU",
) -> None:
    episode = Episode.load(source)
    backend = backend.lower()
    if backend == "mujoco":
        typer.echo(
            TrajectoryViewer(model).render(
                episode,
                output,
                width=width,
                height=height,
                transparent_frames=transparent_frames,
                depth_frames=depth_frames,
                camera_json=camera_json,
            )
        )
        return
    if camera_json is None:
        raise typer.BadParameter("--camera-json is required for calibrated backend rendering")
    if backend in {"blender-eevee", "blender-cycles"}:
        scene = export_blender_scene(
            episode,
            output / "scene",
            model,
            camera_json,
            width,
            height,
            samples=0 if backend == "blender-eevee" else 32,
        )
        typer.echo(
            render_blender_scene(
                scene,
                output / "frames",
                blender,
                device=device.upper(),
                write_depth=depth_frames is not None,
            )
        )
        return
    raise typer.BadParameter("backend must be mujoco, blender-eevee, or blender-cycles")


@app.command("blender-scene")
def blender_scene(
    source: Path,
    destination: Path,
    model: Path | None = None,
    camera_json: Path | None = None,
    width: int = 960,
    height: int = 720,
    max_frames: int | None = None,
    samples: int = 32,
    eevee_samples: int = 16,
    png_compression: int = 15,
) -> None:
    typer.echo(
        export_blender_scene(
            Episode.load(source),
            destination,
            model,
            camera_json,
            width,
            height,
            max_frames,
            samples,
            eevee_samples,
            png_compression,
        )
    )


@app.command("render-blender")
def render_blender(
    scene: Path,
    output: Path,
    blender: Path = Path("blender"),
    max_frames: int | None = None,
    device: str = "CPU",
) -> None:
    typer.echo(render_blender_scene(scene, output, blender, max_frames, device.upper()))


@app.command("configure-blender-hdri")
def configure_blender_hdri_command(
    scene: Path,
    output: Path,
    environment: Path,
    strength: float = typer.Option(0.7, min=0.01),
    rotation_rad: float = 0.0,
    area_light_scale: float = typer.Option(0.25, min=0.0),
) -> None:
    typer.echo(
        configure_blender_environment(
            scene,
            output,
            environment,
            strength=strength,
            rotation_rad=rotation_rad,
            area_light_scale=area_light_scale,
        )
    )


@app.command("render-blender-batch")
def render_blender_batch_command(
    scene: Path,
    output: Path,
    blender: Path = Path("blender"),
    device: str = "OPTIX",
    gpu_ids: str = "0,1",
    workers: int | None = None,
    resume: bool = True,
    max_frames: int | None = None,
    write_depth: bool = False,
) -> None:
    parsed_gpu_ids = tuple(int(value) for value in gpu_ids.split(",") if value.strip())
    typer.echo(
        render_blender_batch(
            scene,
            output,
            blender,
            device=device.upper(),
            gpu_ids=parsed_gpu_ids,
            workers=workers,
            resume=resume,
            max_frames=max_frames,
            write_depth=write_depth,
        )
    )


@app.command("agibot-camera")
def agibot_camera(
    episode: Path,
    intrinsics: Path,
    aligned_extrinsics: Path,
    output: Path,
    video: Path | None = None,
    calibration_width: int | None = None,
    calibration_height: int | None = None,
) -> None:
    typer.echo(
        write_agibot_openarm_camera(
            episode,
            intrinsics,
            aligned_extrinsics,
            output,
            video,
            calibration_width,
            calibration_height,
        )
    )


@app.command("export")
def export(
    sources: list[Path],
    destination: Path,
    feasible_only: bool = True,
    fps: int = 30,
    allow_uncalibrated: bool = typer.Option(False, help="Inspection only; never for release data"),
) -> None:
    typer.echo(
        export_lerobot_v3(
            [Episode.load(source) for source in sources],
            destination,
            fps,
            feasible_only,
            allow_uncalibrated,
        )
    )


@app.command("validate-export")
def validate_export(destination: Path) -> None:
    report = validate_lerobot_v3(destination)
    typer.echo(json.dumps(report, indent=2))
    if not report["ok"]:
        raise typer.Exit(code=1)


@app.command("inpaint")
def inpaint(video: Path, masks: Path, output: Path, method: str = "telea") -> None:
    typer.echo(inpaint_video(video, masks, output, method))


@app.command("fuse-robot-gripper-masks")
def fuse_robot_gripper_masks_command(
    robot_masks: Path,
    gripper_masks: Path,
    output: Path,
    proximity_radius: int = typer.Option(24, min=0),
    minimum_component_area: int = typer.Option(12, min=1),
) -> None:
    typer.echo(
        fuse_robot_gripper_masks(
            robot_masks,
            gripper_masks,
            output,
            proximity_radius=proximity_radius,
            minimum_component_area=minimum_component_area,
        )
    )


@app.command("inpaint-static-camera")
def inpaint_static_camera_command(
    video: Path,
    masks: Path,
    output: Path,
    protected_masks: Path | None = None,
    fallback_video: Path | None = typer.Option(
        None, help="Frame-aligned neural inpaint used only where the real plate is unreliable"
    ),
    reference_image: Path | None = typer.Option(
        None, help="Reusable empty-scene or one-time-inpainted clean reference image"
    ),
    maximum_clean_mad: float = typer.Option(
        0.04, min=0, max=1, help="Maximum normalized temporal MAD for trusting the real plate"
    ),
    fallback_context_radius: int = typer.Option(
        24, min=0, help="Full-strength then feathered fallback context around unreliable pixels"
    ),
    sample_stride: int = typer.Option(10, min=1),
    minimum_clean_observations: int = typer.Option(3, min=1),
    fallback_radius: int = typer.Option(7, min=1),
    feather_radius: int = typer.Option(1, min=0),
) -> None:
    typer.echo(
        inpaint_static_camera(
            video,
            masks,
            output,
            protected_mask_dir=protected_masks,
            fallback_video=fallback_video,
            reference_image=reference_image,
            maximum_clean_mad=maximum_clean_mad,
            fallback_context_radius=fallback_context_radius,
            sample_stride=sample_stride,
            minimum_clean_observations=minimum_clean_observations,
            fallback_radius=fallback_radius,
            feather_radius=feather_radius,
        )
    )


@app.command("refine-removal-masks")
def refine_removal_masks(
    video: Path,
    masks: Path,
    output: Path,
    protected_masks: Path | None = typer.Option(
        None, help="Manipulated-object masks to preserve at robot/object contacts"
    ),
    dilation_radius: int = 7,
    closing_radius: int = 3,
    protect_margin: int = 2,
    optical_flow: bool = True,
    protect_convex_hull: bool = True,
    subtract_protected_masks: bool = typer.Option(
        False,
        help="Legacy mode: exclude protected pixels before inpainting instead of restoring later",
    ),
) -> None:
    typer.echo(
        refine_robot_masks(
            video,
            masks,
            output,
            protected_mask_dir=protected_masks,
            dilation_radius=dilation_radius,
            closing_radius=closing_radius,
            protect_margin=protect_margin,
            use_optical_flow=optical_flow,
            protect_convex_hull=protect_convex_hull,
            subtract_protected_masks=subtract_protected_masks,
        )
    )


@app.command("refine-protected-masks")
def refine_protected_masks_command(
    masks: Path,
    output: Path,
    closing_radius: int = typer.Option(3, min=0),
    dilation_radius: int = typer.Option(2, min=0),
    minimum_hull_area: int = typer.Option(20, min=0),
) -> None:
    typer.echo(
        refine_protected_masks(
            masks,
            output,
            closing_radius=closing_radius,
            dilation_radius=dilation_radius,
            minimum_hull_area=minimum_hull_area,
        )
    )


@app.command("validate-removal-masks")
def validate_removal_masks(
    video: Path,
    masks: Path,
    source_masks: Path | None = typer.Option(
        None, help="Raw segmenter masks whose non-protected pixels must be retained"
    ),
    protected_masks: Path | None = typer.Option(
        None, help="Manipulated-object masks that removal must not overlap"
    ),
) -> None:
    report = validate_robot_masks(
        video,
        masks,
        source_mask_dir=source_masks,
        protected_mask_dir=protected_masks,
    )
    typer.echo(json.dumps(report, indent=2))
    if not report["ok"]:
        raise typer.Exit(code=1)


@app.command("restore-protected")
def restore_protected(
    source: Path,
    clean: Path,
    protected_masks: Path,
    output: Path,
    exclude_masks: Path | None = typer.Option(
        None, help="Old-robot masks whose pixels must never be restored"
    ),
    exclude_margin: int = typer.Option(2, min=0),
    feather_radius: int = typer.Option(2, min=0),
    minimum_mask_area_ratio: float = typer.Option(0.25, min=0, max=1),
) -> None:
    typer.echo(
        restore_protected_video(
            source,
            clean,
            protected_masks,
            output,
            exclude_mask_dir=exclude_masks,
            exclude_margin=exclude_margin,
            feather_radius=feather_radius,
            minimum_mask_area_ratio=minimum_mask_area_ratio,
        )
    )


@app.command("validate-inpainting")
def validate_inpainting_command(
    source: Path,
    inpainted: Path,
    masks: Path,
    source_robot_masks: Path | None = typer.Option(
        None, help="Undilated robot masks used only for copied-source residual detection"
    ),
    protected_masks: Path | None = None,
) -> None:
    report = validate_inpainting(
        source,
        inpainted,
        masks,
        source_robot_mask_dir=source_robot_masks,
        protected_mask_dir=protected_masks,
    )
    typer.echo(json.dumps(report, indent=2))
    if not report["ok"]:
        raise typer.Exit(code=1)


@app.command("segment-robot")
def segment_robot(
    video: Path,
    output: Path,
    prompt: str = "robotic arm",
    expected_arms: int = typer.Option(2, min=1, max=2),
    threshold: float = 0.12,
    box_expansion: float = 0.15,
    include_edge_components: bool = typer.Option(
        False, help="Also segment narrow robot housings at the image edges in ego views"
    ),
    chunk_frames: int = 300,
    max_frames: int | None = None,
    device: int = 0,
    carry_robot_across_chunks: bool = typer.Option(
        True, help="Reuse edge-connected arm tracks instead of re-detecting every chunk"
    ),
) -> None:
    typer.echo(
        segment_robot_video(
            video,
            output,
            prompt=prompt,
            expected_arms=expected_arms,
            threshold=threshold,
            box_expansion=box_expansion,
            include_edge_components=include_edge_components,
            chunk_frames=chunk_frames,
            max_frames=max_frames,
            device=device,
            carry_robot_across_chunks=carry_robot_across_chunks,
        )
    )


@app.command("segment-robotseg")
def segment_robotseg(
    video: Path,
    output: Path,
    repository: Path,
    checkpoint: Path,
    category: str = "robot",
    chunk_frames: int = 120,
    device: int = 0,
) -> None:
    typer.echo(
        segment_robotseg_video(
            video,
            output,
            repository,
            checkpoint,
            category=category,
            chunk_frames=chunk_frames,
            device=device,
        )
    )


@app.command("segment-object")
def segment_object(
    video: Path,
    output: Path,
    prompt: str,
    max_objects: int = typer.Option(1, min=1),
    threshold: float = 0.12,
    box_expansion: float = 0.08,
    chunk_frames: int = 300,
    max_frames: int | None = None,
    device: int = 0,
    seed_box: list[float] | None = typer.Option(
        None, help="Optional initial xyxy box; repeat this option four times"
    ),
) -> None:
    if seed_box is not None and len(seed_box) != 4:
        raise typer.BadParameter(
            "--seed-box must be repeated exactly four times: xmin ymin xmax ymax"
        )
    typer.echo(
        segment_robot_video(
            video,
            output,
            prompt=prompt,
            expected_arms=1,
            threshold=threshold,
            box_expansion=box_expansion,
            chunk_frames=chunk_frames,
            max_frames=max_frames,
            device=device,
            selection="object",
            max_objects=max_objects,
            seed_box=seed_box,
        )
    )


@app.command("composite")
def composite(
    background: Path,
    rgba_frames: Path,
    output: Path,
    protected_masks: Path | None = typer.Option(
        None, help="Manipulated-object masks that must remain in front of the rendered arm"
    ),
    source_depth: Path | None = typer.Option(
        None, help="Per-frame source-scene metric depth (.npy or .npz)"
    ),
    render_depth: Path | None = typer.Option(
        None, help="Per-frame Blender metric depth (.npy or .npz)"
    ),
    depth_tolerance_m: float = 0.01,
    linear_light: bool = typer.Option(
        True, help="Composite display-encoded inputs in scene-linear light"
    ),
    protected_feather_radius: int = typer.Option(
        0, min=0, help="Soft transition radius for translucent protected objects"
    ),
) -> None:
    typer.echo(
        composite_video(
            background,
            rgba_frames,
            output,
            protected_masks,
            source_depth,
            render_depth,
            depth_tolerance_m,
            linear_light=linear_light,
            protected_feather_radius=protected_feather_radius,
        )
    )


@app.command("validate-render")
def validate_render(
    rgba_frames: Path,
    expected_frames: int | None = None,
    minimum_coverage: float = 0.002,
    minimum_visible_fraction: float = 0.95,
) -> None:
    report = validate_rgba_render(
        rgba_frames, expected_frames, minimum_coverage, minimum_visible_fraction
    )
    typer.echo(json.dumps(report, indent=2))
    if not report["ok"]:
        raise typer.Exit(code=1)


@app.command("validate-composite")
def validate_composite_command(
    background: Path,
    composited: Path,
    rgba_frames: Path,
    protected_masks: Path | None = None,
) -> None:
    report = validate_composite_video(background, composited, rgba_frames, protected_masks)
    typer.echo(json.dumps(report, indent=2))
    if not report["ok"]:
        raise typer.Exit(code=1)


@app.command("validate-render-depth")
def validate_render_depth_command(
    rgba_frames: Path,
    depth_frames: Path,
    minimum_p05_alpha_coverage: float = 0.85,
    maximum_depth_m: float = 20.0,
) -> None:
    report = validate_depth_render(
        rgba_frames, depth_frames, minimum_p05_alpha_coverage, maximum_depth_m
    )
    typer.echo(json.dumps(report, indent=2))
    if not report["ok"]:
        raise typer.Exit(code=1)


@app.command("validate-render-alignment")
def validate_render_alignment_command(
    reference_rgba: Path,
    candidate_rgba: Path,
    minimum_mean_iou: float = 0.9,
) -> None:
    report = validate_render_alignment(reference_rgba, candidate_rgba, minimum_mean_iou)
    typer.echo(json.dumps(report, indent=2))
    if not report["ok"]:
        raise typer.Exit(code=1)


@app.command("validate-embodiment-alignment")
def validate_embodiment_alignment_command(
    source_robot_masks: Path,
    rendered_rgba: Path,
    minimum_mean_containment: float = 0.6,
    minimum_p05_containment: float = 0.35,
) -> None:
    report = validate_embodiment_alignment(
        source_robot_masks,
        rendered_rgba,
        minimum_mean_containment,
        minimum_p05_containment,
    )
    typer.echo(json.dumps(report, indent=2))
    if not report["ok"]:
        raise typer.Exit(code=1)


@app.command("distort-render")
def distort_render(rgba_frames: Path, camera_json: Path, output: Path) -> None:
    typer.echo(distort_rgba_frames(rgba_frames, camera_json, output))


@app.command("distort-render-depth")
def distort_render_depth_command(depth_frames: Path, camera_json: Path, output: Path) -> None:
    typer.echo(distort_depth_frames(depth_frames, camera_json, output))


@app.command("harmonize-render")
def harmonize_render_command(
    background: Path,
    rgba_frames: Path,
    output: Path,
    strength: float = 0.65,
    context_radius: int = 18,
    temporal_smoothing: float = 0.9,
) -> None:
    typer.echo(
        harmonize_rgba_frames(
            background,
            rgba_frames,
            output,
            strength=strength,
            context_radius=context_radius,
            temporal_smoothing=temporal_smoothing,
        )
    )


@app.command("calibrate-render-lighting")
def calibrate_render_lighting_command(
    source_video: Path,
    source_robot_masks: Path,
    rgba_frames: Path,
    output: Path,
    protected_masks: Path | None = None,
    sample_stride: int = typer.Option(15, min=1),
    strength: float = typer.Option(0.9, min=0.0, max=1.0),
) -> None:
    typer.echo(
        calibrate_rgba_photometry(
            source_video,
            source_robot_masks,
            rgba_frames,
            output,
            protected_mask_dir=protected_masks,
            sample_stride=sample_stride,
            strength=strength,
        )
    )


@app.command("validate-render-lighting")
def validate_render_lighting_command(
    source_video: Path,
    source_robot_masks: Path,
    reference_rgba: Path,
    candidate_rgba: Path,
    protected_masks: Path | None = None,
    sample_stride: int = typer.Option(15, min=1),
    minimum_relative_improvement: float = 0.1,
) -> None:
    report = validate_photometric_calibration(
        source_video,
        source_robot_masks,
        reference_rgba,
        candidate_rgba,
        protected_mask_dir=protected_masks,
        sample_stride=sample_stride,
        minimum_relative_improvement=minimum_relative_improvement,
    )
    typer.echo(json.dumps(report, indent=2))
    if not report["ok"]:
        raise typer.Exit(code=1)


@app.command("validate-harmonization")
def validate_harmonization_command(reference_rgba: Path, candidate_rgba: Path) -> None:
    report = validate_harmonized_rgba(reference_rgba, candidate_rgba)
    typer.echo(json.dumps(report, indent=2))
    if not report["ok"]:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
