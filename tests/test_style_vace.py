import json

import cv2
import numpy as np

from openarm_retarget.media import apply_mask_constrained_style_batch
from openarm_retarget.style_vace import plan_style_chunks


def test_style_chunk_plan_covers_every_frame_once() -> None:
    jobs = plan_style_chunks(120, chunk_frames=49, overlap=8)
    assert jobs[0]["start"] == jobs[0]["keep_start"] == 0
    assert jobs[-1]["end"] == jobs[-1]["keep_end"] == 120
    assert all(first["keep_end"] == second["keep_start"] for first, second in zip(jobs, jobs[1:]))
    retained = [
        frame
        for job in jobs
        for frame in range(int(job["keep_start"]), int(job["keep_end"]))
    ]
    assert retained == list(range(120))


def test_batch_constraint_center_stitches_complete_video(tmp_path) -> None:
    width, height, total = 32, 24, 60
    reference = tmp_path / "reference.mp4"
    reference_writer = cv2.VideoWriter(
        str(reference), cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height)
    )
    rgba = tmp_path / "rgba"
    rgba.mkdir()
    for index in range(total):
        reference_writer.write(np.full((height, width, 3), 40, dtype=np.uint8))
        layer = np.zeros((height, width, 4), dtype=np.uint8)
        layer[6:18, 8:24, 3] = 255
        cv2.imwrite(str(rgba / f"{index:06d}.png"), layer)
    reference_writer.release()

    jobs = plan_style_chunks(total, chunk_frames=49, overlap=8)
    for job in jobs:
        candidate = tmp_path / f"candidate_{job['index']}.mp4"
        writer = cv2.VideoWriter(
            str(candidate), cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height)
        )
        value = 100 + int(job["index"]) * 40
        for _ in range(49):
            writer.write(np.full((height, width, 3), value, dtype=np.uint8))
        writer.release()
        job["candidate"] = str(candidate)
    manifest = tmp_path / "batch.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "openarm-vace-style-batch-v1",
                "input_video": str(reference),
                "rgba_geometry_authority": str(rgba),
                "protected_masks": None,
                "total_frames": total,
                "jobs": jobs,
            }
        )
    )
    output = tmp_path / "styled.mp4"
    report = apply_mask_constrained_style_batch(
        manifest, output, strength=1, maximum_channel_delta=50
    )
    assert report["frames"] == total
    capture = cv2.VideoCapture(str(output))
    decoded = 0
    while capture.read()[0]:
        decoded += 1
    capture.release()
    assert decoded == total
