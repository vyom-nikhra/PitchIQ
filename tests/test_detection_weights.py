"""resolve_weights contract: existing file wins; absent file tries the URL;
offline degrades to COCO (None) — never to a crash or the blob detector."""

import pytest

from pitchiq.config import DetectionConfig
from pitchiq.perception.detection import ultralytics_backend as ub


def test_existing_weights_used_directly(tmp_path):
    w = tmp_path / "model.pt"
    w.write_bytes(b"x")
    cfg = DetectionConfig(weights=str(w), weights_url="https://example.com/m.pt")
    assert ub.resolve_weights(cfg) == str(w)


def test_no_weights_configured_is_coco():
    assert ub.resolve_weights(DetectionConfig(weights=None)) is None


def test_missing_file_downloads_from_url(tmp_path, monkeypatch):
    w = tmp_path / "model.pt"

    def fake_download(url, dest, timeout=60):
        assert url == "https://example.com/m.pt"
        w.write_bytes(b"weights")

    monkeypatch.setattr(ub, "download_weights", fake_download)
    cfg = DetectionConfig(weights=str(w), weights_url="https://example.com/m.pt")
    assert ub.resolve_weights(cfg) == str(w)
    assert w.exists()


def test_download_failure_degrades_to_coco(tmp_path, monkeypatch):
    def failing_download(url, dest, timeout=60):
        raise OSError("offline")

    monkeypatch.setattr(ub, "download_weights", failing_download)
    cfg = DetectionConfig(weights=str(tmp_path / "missing.pt"),
                          weights_url="https://example.com/m.pt")
    assert ub.resolve_weights(cfg) is None


def test_missing_file_no_url_degrades_to_coco(tmp_path):
    cfg = DetectionConfig(weights=str(tmp_path / "missing.pt"))
    assert ub.resolve_weights(cfg) is None


def test_partial_download_never_leaves_weights_file(tmp_path, monkeypatch):
    """download_weights writes to a .part temp; a mid-stream failure must not
    leave anything at the destination path."""
    import io
    import urllib.request

    class Broken(io.RawIOBase):
        def read(self, n=-1):
            raise OSError("connection reset")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda url, timeout=60: Broken())
    dest = tmp_path / "model.pt"
    with pytest.raises(OSError):
        ub.download_weights("https://example.com/m.pt", dest)
    assert not dest.exists()
