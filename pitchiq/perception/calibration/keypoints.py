"""Learned pitch-keypoint calibration (the optional stronger path).

A heatmap CNN predicts the image locations of the pitch's ~33 semantic
keypoints (see :class:`pitchiq.core.pitch.Pitch`); homography follows from
RANSAC over detected keypoints. More robust than line matching on partial
views because each keypoint is an independent, semantically unambiguous
correspondence.

PitchIQ ships the architecture + inference + training script
(``scripts/train_pitch_keypoints.py``, trains on SoccerNet-Calibration), but
no pretrained weights are bundled — without ``calibration.keypoint_weights``
the calibrator transparently uses the line-based estimator. This is a
documented fallback, not silent degradation: the chosen method is recorded
per frame in the homography table.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from pitchiq.core.pitch import Pitch

log = logging.getLogger(__name__)


def build_keypoint_net(n_keypoints: int = 33):
    """Small U-Net-ish heatmap regressor (torch). Input 3x270x480, output
    ``n_keypoints`` heatmaps at 1/4 resolution."""
    import torch.nn as nn

    def block(cin, cout, stride=1):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    class KeypointNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.enc1 = nn.Sequential(block(3, 32), block(32, 32))
            self.enc2 = nn.Sequential(block(32, 64, 2), block(64, 64))
            self.enc3 = nn.Sequential(block(64, 128, 2), block(128, 128))
            self.enc4 = nn.Sequential(block(128, 256, 2), block(256, 256))
            self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
            self.dec3 = block(256, 128)
            self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
            self.dec2 = block(128, 64)
            self.head = nn.Conv2d(64, n_keypoints, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(e1)
            e3 = self.enc3(e2)
            e4 = self.enc4(e3)
            d3 = self.dec3(__import__("torch").cat([self.up3(e4), e3], dim=1))
            d2 = self.dec2(__import__("torch").cat([self.up2(d3), e2], dim=1))
            return self.head(d2)

    return KeypointNet()


class KeypointCalibrator:
    """Inference wrapper: frame → keypoints → RANSAC homography."""

    INPUT_SIZE = (480, 272)  # (w, h) — both divisible by 8 for the U-Net skips

    def __init__(self, pitch: Pitch, weights_path: str, min_conf: float = 0.35) -> None:
        import torch

        self.torch = torch
        self.pitch = pitch
        self.min_conf = min_conf
        self.names, self.world = pitch.keypoint_array()
        try:  # torchscript export preferred; state_dict accepted
            self.model = torch.jit.load(weights_path, map_location="cpu").eval()
        except RuntimeError:
            self.model = build_keypoint_net(len(self.names))
            self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
            self.model.eval()
        log.info("keypoint calibrator loaded: %s", weights_path)

    def detect_keypoints(self, frame_bgr: np.ndarray) -> dict[str, tuple[float, float, float]]:
        """name -> (x_px, y_px, confidence) at original frame resolution."""
        h0, w0 = frame_bgr.shape[:2]
        img = cv2.resize(frame_bgr, self.INPUT_SIZE)[:, :, ::-1].astype(np.float32) / 255.0
        t = self.torch.from_numpy(img.transpose(2, 0, 1)[None].copy())
        with self.torch.no_grad():
            heat = self.torch.sigmoid(self.model(t))[0].numpy()  # (K, h/4, w/4)
        out = {}
        hh, hw = heat.shape[1:]
        for k, name in enumerate(self.names):
            idx = int(np.argmax(heat[k]))
            y, x = divmod(idx, hw)
            conf = float(heat[k, y, x])
            if conf < self.min_conf:
                continue
            out[name] = (x / hw * w0, y / hh * h0, conf)
        return out

    def estimate(self, frame_bgr: np.ndarray):
        """Return (H, reproj_error_px, n_inliers) or None."""
        kps = self.detect_keypoints(frame_bgr)
        if len(kps) < 4:
            return None
        img_pts = np.array([[v[0], v[1]] for v in kps.values()])
        world_pts = np.array([self.pitch.keypoints[n] for n in kps])
        H, inliers = cv2.findHomography(img_pts, world_pts, cv2.RANSAC, 3.0)
        if H is None or inliers is None or inliers.sum() < 4:
            return None
        inl = inliers.ravel() == 1
        try:
            back = cv2.perspectiveTransform(
                world_pts[inl].reshape(-1, 1, 2).astype(np.float64), np.linalg.inv(H)
            ).reshape(-1, 2)
        except np.linalg.LinAlgError:
            return None
        err = float(np.sqrt(np.mean(np.sum((back - img_pts[inl]) ** 2, axis=1))))
        return H, err, int(inl.sum())
