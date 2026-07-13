import hashlib
import json

from openarm_retarget.download import verify_download


def test_verify_download(tmp_path):
    sample = tmp_path / "sample" / "data.parquet"
    sample.parent.mkdir()
    sample.write_bytes(b"sample")
    digest = hashlib.sha256(b"sample").hexdigest()
    manifest = {
        "repo_id": "example/data",
        "revision": "abc",
        "requested_seconds": 1.0,
        "selected_seconds": 1.0,
        "selected_frames": 2,
        "sample_rows": 2,
        "episodes": [{"frames": 2}],
        "files": {"sample/data.parquet": {"bytes": 6, "sha256": digest}},
    }
    path = tmp_path / "sample_manifest.json"
    path.write_text(json.dumps(manifest))
    assert verify_download(path)["ok"]


def test_verify_download_detects_tamper(tmp_path):
    sample = tmp_path / "data.bin"
    sample.write_bytes(b"changed")
    manifest = {
        "requested_seconds": 1,
        "selected_seconds": 1,
        "sample_rows": 1,
        "episodes": [{"frames": 1}],
        "files": {"data.bin": {"bytes": 7, "sha256": "0" * 64}},
    }
    path = tmp_path / "sample_manifest.json"
    path.write_text(json.dumps(manifest))
    report = verify_download(path)
    assert not report["ok"]
    assert "SHA-256 mismatch: data.bin" in report["errors"]
