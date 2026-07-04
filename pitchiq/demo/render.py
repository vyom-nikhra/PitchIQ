"""Broadcast-style renderer for simulated matches.

Renders the simulator's tracking table as a TV-plausible clip: perspective
main-camera view (a genuine 3D pinhole camera on the sideline that pans and
zooms with the ball), mowing stripes, white markings, kit-coloured player
sprites with jersey numbers, ball with shadow.

Because the camera is synthetic we also emit **ground truth** per frame:
the exact pixel→pitch homography and every entity's bounding box. That makes
the rendered clip a complete Layer-1 validation asset: run the real CV
pipeline on the video and score calibration error in metres, MOTA/IDF1 and
team-assignment accuracy against truth (see ``scripts/validate_synthetic.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pitchiq.core.pitch import Pitch
from pitchiq.core.schema import BALL_ID
from pitchiq.core.video import VideoSink

PLAYER_HEIGHT = 1.80  # metres, for bbox ground truth + sprite size


@dataclass
class RenderResult:
    video_path: Path
    homographies: list[dict]      # frame, timestamp, H (pixel->pitch), method='gt'
    boxes: pd.DataFrame           # frame, entity_id, x1, y1, x2, y2, class, team


class BroadcastCamera:
    """Pinhole camera on the y<0 sideline, panning/zooming to follow the ball."""

    def __init__(self, pitch: Pitch, width: int, height: int, seed: int = 0) -> None:
        self.pitch = pitch
        self.w = width
        self.h = height
        self.rng = np.random.default_rng(seed)
        self.cam_x = pitch.length / 2
        self.look_x = pitch.length / 2
        self.fov = 27.0
        self.cam_y = -28.0
        self.cam_z = 18.0

    def update(self, ball_xy: np.ndarray) -> None:
        """Smoothly track the ball with limited pan speed and gentle zoom."""
        target = float(np.clip(ball_xy[0], 15.0, self.pitch.length - 15.0))
        self.look_x += np.clip(target - self.look_x, -0.35, 0.35)
        self.cam_x += np.clip(self.look_x - self.cam_x, -0.22, 0.22)
        # zoom in a touch when play nears either box
        near_box = min(self.look_x, self.pitch.length - self.look_x) < 25
        target_fov = 23.0 if near_box else 27.0
        self.fov += np.clip(target_fov - self.fov, -0.15, 0.15)
        # subtle operator shake
        self.look_x += float(self.rng.normal(0, 0.02))

    def matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (K, R, C): intrinsics, world→camera rotation, camera centre."""
        C = np.array([self.cam_x, self.cam_y, self.cam_z])
        look = np.array([self.look_x, self.pitch.width * 0.42, 0.0])
        fwd = look - C
        fwd = fwd / np.linalg.norm(fwd)
        up_w = np.array([0.0, 0.0, 1.0])
        right = np.cross(fwd, up_w)
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        R = np.stack([right, -up, fwd])  # rows: image x, image y (down), depth
        f = 0.5 * self.w / np.tan(np.radians(self.fov / 2))
        K = np.array([[f, 0, self.w / 2], [0, f, self.h / 2], [0, 0, 1.0]])
        return K, R, C

    def project(self, pts_world: np.ndarray) -> np.ndarray:
        """Project (N,3) world points to (N,2) pixels (NaN behind camera)."""
        K, R, C = self.matrices()
        pc = (np.atleast_2d(pts_world) - C) @ R.T
        out = np.full((len(pc), 2), np.nan)
        ok = pc[:, 2] > 0.1
        proj = (pc[ok] @ K.T)
        out[ok] = proj[:, :2] / proj[:, 2:3]
        return out

    def ground_homography(self) -> np.ndarray:
        """Exact pixel→pitch homography for the z=0 plane."""
        K, R, C = self.matrices()
        t = -R @ C
        H_world_to_px = K @ np.column_stack([R[:, 0], R[:, 1], t])
        return np.linalg.inv(H_world_to_px)


# ------------------------------------------------------------------ drawing


def _hex_bgr(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


GRASS_DARK = (38, 118, 44)
GRASS_LIGHT = (52, 138, 58)
OFF_PITCH = (30, 72, 34)
LINE_WHITE = (245, 245, 245)
GK_COLORS = {"home": (60, 220, 220), "away": (60, 160, 250)}  # BGR
REF_COLOR = (40, 40, 40)
SKIN = (150, 190, 235)


class BroadcastRenderer:
    def __init__(
        self,
        pitch: Pitch,
        kit_colors: dict[str, str],
        width: int = 1024,
        height: int = 576,
        seed: int = 0,
    ) -> None:
        self.pitch = pitch
        self.w = width
        self.h = height
        self.cam = BroadcastCamera(pitch, width, height, seed)
        self.kit = {k: _hex_bgr(v) for k, v in kit_colors.items()}
        self._circle_cache = {
            name: self._circle_pts(c) for name, c in pitch.circles.items()
        }

    def _circle_pts(self, c) -> np.ndarray:
        t0, t1 = c.theta_range if c.theta_range else (0, 2 * np.pi)
        th = np.linspace(t0, t1, 72)
        return np.stack([c.cx + c.r * np.cos(th), c.cy + c.r * np.sin(th),
                         np.zeros_like(th)], axis=1)

    # ------------------------------------------------------------- frame
    def render_frame(self, entities: pd.DataFrame, ball_xy: np.ndarray | None) -> tuple[np.ndarray, list[dict]]:
        """Draw one frame. ``entities``: rows of the tracking table for this
        frame (players/GK/ref). Returns (image, gt_boxes)."""
        img = np.full((self.h, self.w, 3), OFF_PITCH, dtype=np.uint8)
        self._draw_pitch(img)
        boxes: list[dict] = []

        # draw far-to-near so occlusion looks right (smaller y_pitch = farther?
        # camera sits at y<0, so larger y_pitch = farther from camera)
        persons = entities[entities["class"] != "ball"].copy()
        persons = persons.sort_values("y_pitch", ascending=False)
        for _, row in persons.iterrows():
            box = self._draw_person(img, row)
            if box is not None:
                boxes.append(box)

        if ball_xy is not None and np.all(np.isfinite(ball_xy)):
            self._draw_ball(img, ball_xy, boxes)
        return img, boxes

    def _draw_pitch(self, img: np.ndarray) -> None:
        L, W = self.pitch.length, self.pitch.width
        # mowing stripes: filled projected quads
        n_stripes = 14
        xs = np.linspace(0, L, n_stripes + 1)
        for i in range(n_stripes):
            quad = np.array(
                [[xs[i], 0, 0], [xs[i + 1], 0, 0], [xs[i + 1], W, 0], [xs[i], W, 0]]
            )
            px = self.cam.project(quad)
            if np.isnan(px).any():
                continue
            color = GRASS_LIGHT if i % 2 == 0 else GRASS_DARK
            cv2.fillPoly(img, [px.astype(np.int32)], color)
        # markings
        for (a, b) in self.pitch.lines.values():
            pts = self.cam.project(np.array([[a[0], a[1], 0], [b[0], b[1], 0]]))
            if np.isnan(pts).any():
                continue
            cv2.line(img, tuple(pts[0].astype(int)), tuple(pts[1].astype(int)), LINE_WHITE, 2,
                     cv2.LINE_AA)
        for pts3 in self._circle_cache.values():
            px = self.cam.project(pts3)
            ok = ~np.isnan(px).any(axis=1)
            if ok.sum() >= 2:
                cv2.polylines(img, [px[ok].astype(np.int32)], False, LINE_WHITE, 2, cv2.LINE_AA)
        # spots
        for name in ("penalty_spot_left", "penalty_spot_right", "center_spot"):
            x, y = self.pitch.keypoints[name]
            p = self.cam.project(np.array([[x, y, 0]]))[0]
            if not np.isnan(p).any():
                cv2.circle(img, tuple(p.astype(int)), 2, LINE_WHITE, -1, cv2.LINE_AA)

    def _person_color(self, row) -> tuple[int, int, int]:
        cls = row["class"]
        team = str(row["team"])
        if cls == "referee":
            return REF_COLOR
        if cls == "goalkeeper":
            return GK_COLORS.get(team, (0, 200, 200))
        return self.kit.get(team, (200, 200, 200))

    def _draw_person(self, img: np.ndarray, row) -> dict | None:
        x, y = float(row["x_pitch"]), float(row["y_pitch"])
        foot = self.cam.project(np.array([[x, y, 0.0]]))[0]
        head = self.cam.project(np.array([[x, y, PLAYER_HEIGHT]]))[0]
        if np.isnan(foot).any() or np.isnan(head).any():
            return None
        h_px = abs(foot[1] - head[1])
        if h_px < 4 or foot[0] < -30 or foot[0] > self.w + 30 or foot[1] < 0 or foot[1] > self.h + 30:
            return None
        w_px = 0.42 * h_px
        color = self._person_color(row)

        # shadow
        cv2.ellipse(img, (int(foot[0]), int(foot[1])), (int(w_px * 0.6), max(2, int(h_px * 0.07))),
                    0, 0, 360, (25, 60, 28), -1, cv2.LINE_AA)
        # legs (shorts darker)
        leg_top = foot[1] - 0.45 * h_px
        cv2.rectangle(img, (int(foot[0] - w_px * 0.28), int(leg_top)),
                      (int(foot[0] + w_px * 0.28), int(foot[1])), tuple(int(c * 0.5) for c in color), -1)
        # torso
        torso_top = foot[1] - 0.82 * h_px
        cv2.rectangle(img, (int(foot[0] - w_px * 0.42), int(torso_top)),
                      (int(foot[0] + w_px * 0.42), int(leg_top)), color, -1)
        # head
        cv2.circle(img, (int(foot[0]), int(foot[1] - 0.90 * h_px)), max(2, int(h_px * 0.10)),
                   SKIN, -1, cv2.LINE_AA)
        # jersey number
        jno = row.get("jersey_no")
        if jno is not None and not pd.isna(jno) and h_px >= 26:
            txt = str(int(jno))
            scale = h_px / 110.0
            tw = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
            cv2.putText(img, txt, (int(foot[0] - tw / 2), int(foot[1] - 0.60 * h_px)),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)

        x1 = float(foot[0] - w_px / 2)
        y1 = float(head[1])
        return {
            "entity_id": int(row["entity_id"]), "x1": x1, "y1": y1,
            "x2": x1 + w_px, "y2": float(foot[1]),
            "class": row["class"], "team": str(row["team"]),
        }

    def _draw_ball(self, img: np.ndarray, ball_xy: np.ndarray, boxes: list[dict]) -> None:
        p = self.cam.project(np.array([[ball_xy[0], ball_xy[1], 0.12]]))[0]
        shadow = self.cam.project(np.array([[ball_xy[0], ball_xy[1], 0.0]]))[0]
        if np.isnan(p).any():
            return
        # scale ball radius with local pixel height
        head = self.cam.project(np.array([[ball_xy[0], ball_xy[1], PLAYER_HEIGHT]]))[0]
        r = max(2, int(abs(shadow[1] - head[1]) * 0.06))
        if not np.isnan(shadow).any():
            cv2.ellipse(img, (int(shadow[0]), int(shadow[1])), (r, max(1, r // 2)), 0, 0, 360,
                        (25, 60, 28), -1, cv2.LINE_AA)
        cv2.circle(img, (int(p[0]), int(p[1])), r, (250, 250, 250), -1, cv2.LINE_AA)
        cv2.circle(img, (int(p[0]), int(p[1])), r, (60, 60, 60), 1, cv2.LINE_AA)
        boxes.append({"entity_id": BALL_ID, "x1": float(p[0] - r), "y1": float(p[1] - r),
                      "x2": float(p[0] + r), "y2": float(p[1] + r), "class": "ball", "team": "none"})


def render_match(
    tracking: pd.DataFrame,
    pitch: Pitch,
    kit_colors: dict[str, str],
    out_path: str | Path,
    fps: float,
    width: int = 1024,
    height: int = 576,
    progress_cb=None,
) -> RenderResult:
    """Render a whole tracking table to video + ground-truth homographies/boxes."""
    renderer = BroadcastRenderer(pitch, kit_colors, width, height)
    out_path = Path(out_path)
    frames = tracking.groupby("frame")
    n = tracking["frame"].nunique()
    hrecs: list[dict] = []
    box_rows: list[dict] = []

    with VideoSink(out_path, fps, (width, height)) as sink:
        for frame_idx, group in frames:
            ball = group[group["entity_id"] == BALL_ID]
            ball_xy = ball[["x_pitch", "y_pitch"]].to_numpy()[0] if len(ball) else None
            if ball_xy is not None:
                renderer.cam.update(ball_xy)
            img, boxes = renderer.render_frame(group, ball_xy)
            sink.write(img)
            ts = float(group["timestamp"].iloc[0])
            hrecs.append({"frame": int(frame_idx), "timestamp": ts,
                          "H": renderer.cam.ground_homography(),
                          "reproj_error_px": 0.0, "method": "gt", "is_scene_cut": False})
            for b in boxes:
                b.update({"frame": int(frame_idx)})
                box_rows.append(b)
            if progress_cb and frame_idx % 100 == 0:
                progress_cb(frame_idx / max(n, 1), f"rendering frame {frame_idx}/{n}")

    return RenderResult(out_path, hrecs, pd.DataFrame(box_rows))
