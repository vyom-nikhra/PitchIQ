"""TrackNet ball-tracker unit tests (architecture + heatmap decoding).

Training/real inference need torch + weights; these validate the shape
contract and the peak-decoding maths without either, so they run in CI.
"""

import numpy as np
import pytest

from pitchiq.perception.detection.tracknet import gaussian_heatmap


def test_gaussian_heatmap_peaks_at_ball():
    hm = gaussian_heatmap(72, 128, cx=40.0, cy=30.0, sigma=3.0)
    iy, ix = np.unravel_index(int(hm.argmax()), hm.shape)
    assert (ix, iy) == (40, 30)
    assert hm.max() == pytest.approx(1.0, abs=1e-5)
    # falls off away from the centre
    assert hm[30, 40] > hm[30, 60] > hm[30, 100]


def test_tracknet_forward_shape():
    torch = pytest.importorskip("torch")
    from pitchiq.perception.detection.tracknet import build_tracknet

    model = build_tracknet(in_frames=3).eval()
    x = torch.zeros(2, 9, 288, 512)  # 3 RGB frames stacked
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 1, 288, 512)


def test_peak_decode_subpixel():
    """The centroid decoder recovers a sub-pixel ball location from a heatmap."""
    torch = pytest.importorskip("torch")
    from pitchiq.perception.detection.tracknet import TrackNetBall

    # build an instance without loading weights, then exercise _peak directly
    bt = TrackNetBall.__new__(TrackNetBall)
    bt.peak_threshold = 0.5
    heat = gaussian_heatmap(72, 128, cx=50.4, cy=20.6, sigma=2.5)
    # map from heatmap grid (128x72) to a 'full-res' 256x144 frame
    res = bt._peak(heat, w0=256, h0=144)
    assert res is not None
    x, y, conf = res
    assert conf > 0.9
    assert abs(x - 50.4 / 128 * 256) < 3
    assert abs(y - 20.6 / 72 * 144) < 3
    # below-threshold heatmap returns None
    assert bt._peak(np.zeros((72, 128), np.float32), 256, 144) is None
