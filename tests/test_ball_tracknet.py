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


def test_training_heatmap_has_exact_peak():
    """The focal loss defines positives as target == 1.0 exactly; a Gaussian
    at a fractional centre never reaches 1.0, so the training target must
    stamp the rounded peak pixel — otherwise the model collapses to zeros."""
    from pitchiq.perception.detection.ball_dataset import gaussian_heatmap as gh

    hm = gh(72, 128, cx=50.4, cy=20.6, sigma=3.0)
    assert (hm == 1.0).sum() == 1
    iy, ix = np.unravel_index(int(hm.argmax()), hm.shape)
    assert (ix, iy) == (50, 21)


def test_focal_loss_prefers_correct_heatmap():
    torch = pytest.importorskip("torch")
    from pitchiq.perception.detection.ball_dataset import (
        focal_heatmap_loss,
        gaussian_heatmap as gh,
    )

    target = torch.from_numpy(gh(36, 64, cx=20.0, cy=10.0, sigma=3.0))[None, None]
    logit_good = (target * 12.0) - 6.0          # sigmoid ~= target
    logit_flat = torch.full_like(target, -6.0)  # all-background prediction
    good = focal_heatmap_loss(logit_good, target)
    flat = focal_heatmap_loss(logit_flat, target)
    assert float(good) < float(flat)
    assert float(good) < 1.0


def test_ball_window_dataset_end_to_end(tmp_path):
    """Windows listed from a fake MOT sequence produce correct tensors."""
    torch = pytest.importorskip("torch")
    cv2 = pytest.importorskip("cv2")
    from pitchiq.perception.detection.ball_dataset import (
        BallWindowDataset,
        list_ball_windows,
    )

    seq = tmp_path / "SNMOT-000"
    (seq / "img1").mkdir(parents=True)
    (seq / "gt").mkdir()
    for i in range(1, 6):
        cv2.imwrite(str(seq / "img1" / f"{i:06d}.jpg"),
                    np.full((90, 160, 3), 60, np.uint8))
    # MOT rows: frame,id,x,y,w,h,conf,cls,vis — id 9 tiny box = the ball
    rows = []
    for f in range(1, 6):
        rows.append(f"{f},1,10,10,12,30,1,1,1")          # a player
        rows.append(f"{f},9,{40 + f},45,4,4,1,1,1")       # the ball
    (seq / "gt" / "gt.txt").write_text("\n".join(rows))

    wins = list_ball_windows(seq)
    assert len(wins) == 3  # frames 3..5 as window ends
    paths, frac = wins[0]
    assert len(paths) == 3 and frac is not None
    assert 0.0 < frac[0] < 1.0 and 0.0 < frac[1] < 1.0

    ds = BallWindowDataset(wins, input_size=(64, 36))
    stack, heat, has, px = ds[0]
    assert stack.shape == (9, 36, 64)
    assert heat.shape == (1, 36, 64)
    assert float(has) == 1.0
    assert float(heat.max()) == 1.0
    # heatmap peak sits at the scaled ball pixel
    iy, ix = np.unravel_index(int(heat[0].numpy().argmax()), (36, 64))
    assert abs(ix - float(px[0])) <= 1 and abs(iy - float(px[1])) <= 1
