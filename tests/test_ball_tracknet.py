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


def test_safe_convs_block_fp16_overflow_and_keep_state_dict_keys():
    """Regression: on fp16-only GPUs (Kaggle T4), pre-BatchNorm activations
    grew past fp16's 65504 ceiling and the infs poisoned BN running stats on
    the very forward pass that produced them. Every conv output is clamped in
    fp16; the subclasses must not change checkpoint keys, and fp32 behaviour
    must be untouched."""
    torch = pytest.importorskip("torch")
    from pitchiq.perception.detection.tracknet import build_tracknet

    model = build_tracknet(in_frames=3)
    # 1) state_dict keys identical to the original plain-module layout
    keys = set(model.state_dict().keys())
    assert "e1.0.0.weight" in keys and "up3.weight" in keys and "head.weight" in keys

    # 2) engineered blow-up: huge first-layer weights so conv outputs exceed
    #    the fp16 max; in fp16 the clamp must keep everything finite
    conv = model.e1[0][0]
    with torch.no_grad():
        conv.weight.fill_(500.0)
    x16 = torch.full((1, 9, 32, 64), 100.0, dtype=torch.float16)
    try:
        out16 = conv.half()(x16)
    except (RuntimeError, NotImplementedError):
        pytest.skip("fp16 conv unsupported on this CPU build")
    assert torch.isfinite(out16).all()
    assert float(out16.abs().max()) <= 30000.0

    # 3) fp32 path is a no-op: values may exceed the clamp bound freely
    out32 = conv.float()(x16.float())
    assert float(out32.abs().max()) > 30000.0


def test_refine_ball_track_outliers_smoothing_and_cuts():
    """Outlier peaks are dropped (not smoothed toward), jitter shrinks, and
    smoothing never crosses a scene cut (two camera views are unrelated)."""
    import pandas as pd

    from pitchiq.perception.detection.ball import refine_ball_track

    rng = np.random.default_rng(0)
    n = 60
    frames = np.arange(n)
    x = 100.0 + 8.0 * frames + rng.normal(0, 2.0, n)   # linear motion + jitter
    y = 200.0 + 2.0 * frames + rng.normal(0, 2.0, n)
    x[30] += 500.0                                     # teleport outlier
    # scene cut at frame 45: camera jumps, position legitimately teleports
    x[45:] += 900.0
    rows = [dict(frame=int(f), timestamp=f / 25, entity_id=-1,
                 x_pixel=float(xi), y_pixel=float(yi),
                 x_pitch=float(xi) / 10, y_pitch=float(yi) / 10, conf=0.9)
            for f, xi, yi in zip(frames, x, y)]
    rows.append(dict(frame=0, timestamp=0.0, entity_id=7, x_pixel=5.0,
                     y_pixel=5.0, x_pitch=0.5, y_pitch=0.5, conf=0.9))
    df = pd.DataFrame(rows)

    out = refine_ball_track(df, ball_id=-1, cut_frames={45})
    ball = out[out.entity_id == -1].set_index("frame")

    assert 30 not in ball.index                       # outlier dropped
    assert 7 in out.entity_id.values                  # players untouched
    # jitter reduced on the clean stretch (compare residuals to the true line)
    seg = ball.loc[5:25]
    resid = seg.x_pixel - (100.0 + 8.0 * seg.index.to_numpy())
    assert resid.abs().mean() < 2.0
    # the cut boundary stays sharp: frame 45 is not dragged toward pre-cut
    assert ball.loc[45, "x_pixel"] > 1200.0


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


def test_focal_loss_finite_at_fp16_confidence():
    """Regression: under autocast, confident fp16 logits saturate sigmoid to
    exactly 1.0, and a log(1 - pred) formulation returns -inf — one such batch
    NaN-poisoned a real training run. The logsigmoid form must stay finite for
    arbitrarily confident predictions in half precision."""
    torch = pytest.importorskip("torch")
    from pitchiq.perception.detection.ball_dataset import (
        focal_heatmap_loss,
        gaussian_heatmap as gh,
    )

    target = torch.from_numpy(gh(36, 64, cx=20.0, cy=10.0, sigma=3.0))[None, None]
    # fp16 logits confident enough that sigmoid(x) == 1.0 exactly in half
    logits = torch.full_like(target, 30.0).half()
    assert float(torch.sigmoid(logits).max()) == 1.0  # the trap is armed
    loss = focal_heatmap_loss(logits, target)
    assert torch.isfinite(loss)
    # and gradients flow without NaN
    logits32 = torch.full((1, 1, 36, 64), 30.0, requires_grad=True)
    focal_heatmap_loss(logits32, target).backward()
    assert torch.isfinite(logits32.grad).all()


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
