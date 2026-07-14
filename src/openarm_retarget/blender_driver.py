"""Blender-side Cycles renderer. Invoked by ``openarm-retarget render-blender``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _arguments() -> argparse.Namespace:
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("scene", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("CPU", "CUDA", "OPTIX"), default="CPU")
    parser.add_argument("--write-depth", action="store_true")
    return parser.parse_args(arguments)


def _matrix(values):
    from mathutils import Matrix

    return Matrix(values)


def _srgb_to_linear(value):
    value = float(value)
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def _enable_device(scene, requested: str) -> None:
    import bpy

    scene.cycles.device = "CPU"
    if requested == "CPU":
        return
    preferences = bpy.context.preferences.addons["cycles"].preferences
    try:
        preferences.compute_device_type = requested
        preferences.get_devices()
        enabled = False
        for device in preferences.devices:
            device.use = device.type == requested
            enabled |= device.use
        if enabled:
            scene.cycles.device = "GPU"
    except Exception as error:
        print(f"GPU setup failed; falling back to CPU: {error}")


def main() -> None:
    import bpy

    args = _arguments()
    spec = json.loads(args.scene.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    engine = spec.get("engine", "CYCLES")
    scene.render.engine = engine
    if engine == "CYCLES":
        scene.cycles.samples = int(spec["samples"])
        scene.cycles.use_denoising = True
    elif engine == "BLENDER_EEVEE_NEXT":
        scene.eevee.taa_render_samples = int(spec.get("eevee_samples", 16))
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = int(spec.get("png_compression", 15))
    color_management = spec.get("color_management", {})
    scene.view_settings.view_transform = color_management.get("view_transform", "AgX")
    scene.view_settings.look = color_management.get("look", "AgX - Medium High Contrast")
    scene.view_settings.exposure = float(color_management.get("exposure", 0.0))
    scene.render.resolution_x, scene.render.resolution_y = spec["resolution"]
    scene.render.resolution_percentage = 100
    scene.render.fps = round(spec["fps"])
    depth_output = None
    if args.write_depth:
        scene.view_layers[0].use_pass_z = True
        (args.output / "depth").mkdir(exist_ok=True)
        depth_temp = args.output / ".depth_tmp"
        depth_temp.mkdir(exist_ok=True)
        scene.use_nodes = True
        nodes = scene.node_tree.nodes
        nodes.clear()
        render_layers = nodes.new("CompositorNodeRLayers")
        depth_output = nodes.new("CompositorNodeOutputFile")
        depth_output.base_path = str(depth_temp)
        depth_output.format.file_format = "OPEN_EXR"
        depth_output.format.color_mode = "BW"
        depth_output.format.color_depth = "32"
        depth_output.format.exr_codec = "ZIP"
        scene.node_tree.links.new(render_layers.outputs["Depth"], depth_output.inputs[0])
    if engine == "CYCLES":
        _enable_device(scene, args.device)

    world = bpy.data.worlds.new("transparent_world") if scene.world is None else scene.world
    scene.world = world
    world.use_nodes = True
    lighting = spec.get("lighting", {})
    world_nodes = world.node_tree.nodes
    world_links = world.node_tree.links
    background = world_nodes.get("Background")
    background.inputs["Color"].default_value = lighting.get(
        "world_color_linear", (0.12, 0.12, 0.12, 1)
    )
    background.inputs["Strength"].default_value = lighting.get("world_strength", 0.35)
    environment_path = lighting.get("environment_path")
    if environment_path:
        environment = world_nodes.new("ShaderNodeTexEnvironment")
        resolved_environment = Path(environment_path)
        if not resolved_environment.is_absolute():
            resolved_environment = args.scene.parent / resolved_environment
        environment.image = bpy.data.images.load(str(resolved_environment), check_existing=True)
        environment.interpolation = "Linear"
        mapping = world_nodes.new("ShaderNodeMapping")
        coordinates = world_nodes.new("ShaderNodeTexCoord")
        mapping.inputs["Rotation"].default_value[2] = float(
            lighting.get("environment_rotation_rad", 0.0)
        )
        world_links.new(coordinates.outputs["Generated"], mapping.inputs["Vector"])
        world_links.new(mapping.outputs["Vector"], environment.inputs["Vector"])
        world_links.new(environment.outputs["Color"], background.inputs["Color"])

    imported = []
    for item in spec["objects"]:
        existing = set(bpy.data.objects)
        bpy.ops.wm.obj_import(filepath=str(args.scene.parent / item["mesh"]))
        candidates = list(set(bpy.data.objects) - existing)
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected one object from {item['mesh']}, received {len(candidates)}"
            )
        obj = candidates[0]
        obj.name = item["name"]
        material_spec = item["material"]
        rgba = material_spec["rgba"]
        if material_spec.get("color_space") == "sRGB":
            shader_rgba = tuple(_srgb_to_linear(value) for value in rgba[:3]) + (rgba[3],)
        else:
            shader_rgba = rgba
        material = bpy.data.materials.new(f"{item['name']}_material")
        material.diffuse_color = shader_rgba
        material.use_nodes = True
        principled = material.node_tree.nodes.get("Principled BSDF")
        principled.inputs["Base Color"].default_value = shader_rgba
        principled.inputs["Metallic"].default_value = material_spec["metallic"]
        principled.inputs["Roughness"].default_value = material_spec["roughness"]
        obj.data.materials.clear()
        obj.data.materials.append(material)
        imported.append((obj, item["world_from_object_frames"]))

    camera_data = bpy.data.cameras.new("OpenCV calibrated camera")
    camera = bpy.data.objects.new("OpenCV calibrated camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    width, height = spec["resolution"]
    intrinsics = spec["camera"]["intrinsics"]
    fx, fy = float(intrinsics[0][0]), float(intrinsics[1][1])
    cx, cy = float(intrinsics[0][2]), float(intrinsics[1][2])
    camera_data.type = "PERSP"
    camera_data.sensor_fit = "HORIZONTAL"
    camera_data.sensor_width = 36.0
    camera_data.lens = fx * camera_data.sensor_width / width
    camera_data.shift_x = (width / 2 - cx) / width
    pixel_aspect_ratio = fx / fy
    camera_data.shift_y = (cy - height / 2) * pixel_aspect_ratio / width
    # Blender clamps each pixel-aspect component to >= 1. Encode a ratio below one by
    # increasing the x component rather than assigning an invalid y component.
    scene.render.pixel_aspect_x = fy / fx
    scene.render.pixel_aspect_y = 1.0
    camera_data.lens_unit = "MILLIMETERS"

    # Broad softboxes give useful metallic highlights while preserving transparent alpha.
    area_lights = lighting.get(
        "area_lights",
        [
            {"location": (1.4, -1.0, 1.8), "energy_w": 160, "size_m": 2.0},
            {"location": (-1.2, 0.8, 1.2), "energy_w": 100, "size_m": 1.5},
        ],
    )
    for index, light_spec in enumerate(area_lights):
        light_data = bpy.data.lights.new(f"softbox_{index}", "AREA")
        light_data.energy = float(light_spec["energy_w"])
        light_data.shape = "DISK"
        light_data.size = float(light_spec["size_m"])
        color = light_spec.get("color_srgb", (1.0, 1.0, 1.0))
        light_data.color = tuple(_srgb_to_linear(value) for value in color)
        light = bpy.data.objects.new(f"softbox_{index}", light_data)
        light.location = light_spec["location"]
        if "target" in light_spec:
            from mathutils import Vector

            direction = Vector(light_spec["target"]) - light.location
            light.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        scene.collection.objects.link(light)

    frame_count = int(spec["episode_frames"])
    if args.max_frames is not None:
        frame_count = min(frame_count, args.max_frames)
    camera_poses = spec["camera"]["world_from_camera_frames"]
    cv_to_blender = _matrix(((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1)))
    start_frame = max(0, args.start_frame)
    end_frame = min(frame_count, args.end_frame if args.end_frame is not None else frame_count)
    if start_frame >= end_frame:
        raise ValueError(f"Empty Blender frame range [{start_frame}, {end_frame})")
    rendered = 0
    skipped = 0
    started = time.perf_counter()
    for frame in range(start_frame, end_frame):
        output_path = args.output / f"{frame:06d}.png"
        if args.resume and output_path.is_file() and output_path.stat().st_size > 100:
            skipped += 1
            continue
        for obj, transforms in imported:
            obj.matrix_world = _matrix(transforms[frame])
        pose = camera_poses[0 if len(camera_poses) == 1 else frame]
        camera.matrix_world = _matrix(pose) @ cv_to_blender
        scene.render.filepath = str(output_path)
        scene.frame_set(frame + 1)
        if depth_output is not None:
            depth_output.file_slots[0].path = f"{frame:06d}_"
        bpy.ops.render.render(write_still=True)
        if args.write_depth:
            import numpy as np

            matches = sorted((args.output / ".depth_tmp").glob(f"{frame:06d}_*.exr"))
            if len(matches) != 1:
                raise RuntimeError(f"Expected one depth EXR for frame {frame}; got {matches}")
            depth_image = bpy.data.images.load(str(matches[0]), check_existing=False)
            depth = np.asarray(depth_image.pixels[:], dtype=np.float32)
            depth = depth.reshape(height, width, 4)[..., 0]
            combined_image = bpy.data.images.load(str(output_path), check_existing=False)
            combined = np.asarray(combined_image.pixels[:], dtype=np.float32).reshape(
                height, width, 4
            )
            # Blender image arrays use a bottom-left origin; NumPy consumers use top-left.
            depth = np.flipud(depth).copy()
            alpha = np.flipud(combined[..., 3])
            depth[(alpha <= 1e-6) | (depth >= camera_data.clip_end * 0.99)] = np.inf
            np.savez_compressed(args.output / "depth" / f"{frame:06d}.npz", depth_m=depth)
            bpy.data.images.remove(depth_image)
            bpy.data.images.remove(combined_image)
            matches[0].unlink()
        rendered += 1
        print(f"Rendered frame {frame + 1}/{frame_count}")
    elapsed = time.perf_counter() - started
    report = {
        "engine": engine,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "rendered": rendered,
        "skipped": skipped,
        "seconds": elapsed,
        "render_fps": rendered / elapsed if elapsed else 0.0,
    }
    report_path = args.output / f"worker_{start_frame:06d}_{end_frame:06d}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if args.write_depth:
        (args.output / ".depth_tmp").rmdir()


if __name__ == "__main__":
    main()
