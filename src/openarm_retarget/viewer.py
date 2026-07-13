from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

from .constants import ARM_JOINT_NAMES, SIDES
from .gripper import closure_to_finger_qpos, finger_qpos_addresses
from .model import resolve_model
from .schema import Episode


class TrajectoryViewer:
    def __init__(self, model_path: str | Path | None = None):
        self.model_path = resolve_model(model_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.qpos = {}
        for side in SIDES:
            ids = [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in ARM_JOINT_NAMES[side]
            ]
            self.qpos[side] = self.model.jnt_qposadr[ids]
        self.finger_qpos = finger_qpos_addresses(self.model)

    def set_frame(self, episode: Episode, frame: int) -> None:
        if episode.joint_position is None:
            raise ValueError("Viewer requires an IK-solved episode")
        for side_index, side in enumerate(SIDES):
            self.data.qpos[self.qpos[side]] = episode.joint_position[frame, side_index]
        self.data.qpos[self.finger_qpos] = closure_to_finger_qpos(episode.gripper[frame])
        mujoco.mj_forward(self.model, self.data)

    def interactive(self, episode: Episode, realtime: bool = True) -> None:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            start = time.monotonic()
            for frame in range(len(episode.timestamp)):
                if not viewer.is_running():
                    break
                self.set_frame(episode, frame)
                viewer.sync()
                if realtime:
                    target = start + episode.timestamp[frame] - episode.timestamp[0]
                    time.sleep(max(0, target - time.monotonic()))

    def render(
        self,
        episode: Episode,
        output: str | Path,
        width: int = 960,
        height: int = 720,
        fps: float | None = None,
        transparent_frames: str | Path | None = None,
        depth_frames: str | Path | None = None,
        camera_json: str | Path | None = None,
    ) -> Path:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fps = fps or 1 / episode.sample_period
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, width)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, height)
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = [0.0, 0.0, -0.2]
        camera.distance = 1.35
        camera.azimuth = 145
        camera.elevation = -18
        camera_spec = json.loads(Path(camera_json).read_text()) if camera_json else None
        writer = cv2.VideoWriter(
            str(output), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
        )
        alpha_dir = Path(transparent_frames) if transparent_frames else None
        if alpha_dir:
            alpha_dir.mkdir(parents=True, exist_ok=True)
        depth_dir = Path(depth_frames) if depth_frames else None
        if depth_dir:
            depth_dir.mkdir(parents=True, exist_ok=True)
        try:
            for frame in range(len(episode.timestamp)):
                self.set_frame(episode, frame)
                renderer.update_scene(self.data, camera=camera)
                if camera_spec:
                    frame_spec = camera_spec
                    if "world_from_camera_frames" in camera_spec:
                        poses = camera_spec["world_from_camera_frames"]
                        if len(poses) not in (1, len(episode.timestamp)):
                            raise ValueError("Per-frame camera count does not match the episode")
                        frame_spec = {
                            **camera_spec,
                            "world_from_camera": poses[0 if len(poses) == 1 else frame],
                        }
                    self._set_calibrated_camera(renderer, frame_spec, width, height)
                rgb = renderer.render()
                if episode.feasible is not None and not episode.feasible[frame]:
                    cv2.putText(
                        rgb, "INFEASIBLE", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 30, 30), 3
                    )
                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                if alpha_dir:
                    renderer.enable_segmentation_rendering()
                    segmentation = renderer.render()
                    renderer.disable_segmentation_rendering()
                    alpha = np.where(segmentation[..., 0] >= 0, 255, 0).astype(np.uint8)
                    rgba = np.dstack([rgb, alpha])
                    cv2.imwrite(
                        str(alpha_dir / f"{frame:06d}.png"), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
                    )
                if depth_dir:
                    renderer.enable_depth_rendering()
                    depth = renderer.render()
                    renderer.disable_depth_rendering()
                    np.save(depth_dir / f"{frame:06d}.npy", depth.astype(np.float32))
        finally:
            writer.release()
            renderer.close()
        return output

    @staticmethod
    def _set_calibrated_camera(renderer, spec: dict, width: int, height: int) -> None:
        world_from_camera = np.asarray(spec["world_from_camera"], dtype=np.float64)
        intrinsics = np.asarray(spec["intrinsics"], dtype=np.float64)
        if world_from_camera.shape != (4, 4) or intrinsics.shape != (3, 3):
            raise ValueError("Camera requires 4x4 world_from_camera and 3x3 intrinsics")
        position = world_from_camera[:3, 3]
        # OpenCV camera axes are right/down/forward; MuJoCo uses right/up/back frame axes.
        forward = world_from_camera[:3, 2]
        up = -world_from_camera[:3, 1]
        near = float(spec.get("near", 0.01))
        far = float(spec.get("far", 10.0))
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        for gl_camera in renderer.scene.camera:
            gl_camera.pos[:] = position
            gl_camera.forward[:] = forward / np.linalg.norm(forward)
            gl_camera.up[:] = up / np.linalg.norm(up)
            gl_camera.frustum_near = near
            gl_camera.frustum_far = far
            gl_camera.frustum_top = cy * near / fy
            gl_camera.frustum_bottom = -(height - cy) * near / fy
            gl_camera.frustum_center = (width - 2 * cx) * near / (2 * fx)
            gl_camera.frustum_width = width * near / (2 * fx)
            gl_camera.orthographic = 0
