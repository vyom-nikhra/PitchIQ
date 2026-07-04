"""Agent-based football match simulator with ground-truth events.

Not a physics-perfect football model — a *behaviourally plausible* one, rich
enough that every PitchIQ analytics module has real signal to detect:
formation shape with in/out-of-possession morphing, pressing driven by a
press-intensity profile, man vs zonal marking with known assignments, passing
with openness-aware target choice, interceptions, turnovers, halves with
direction flip.

Everything downstream treats its output like any other match: the tracking
table is schema-identical to the CV pipeline's. Ground truth (passes,
possession, marking map) is returned separately for validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pitchiq.config import SimulatorConfig
from pitchiq.core.formations import formation_slots
from pitchiq.core.pitch import Pitch
from pitchiq.core.schema import BALL_ID, MatchMeta, validate_tracking_table
from pitchiq.demo.profiles import TeamProfile

MAX_ACCEL = 4.5          # m/s^2
GK_IDX = 10              # index of the goalkeeper within a team's arrays
REFEREE_ID = 200

EVENT_COLUMNS = [
    "frame", "timestamp", "type", "team", "from_id", "to_id",
    "x", "y", "end_x", "end_y", "outcome",
]


@dataclass
class SimResult:
    tracking: pd.DataFrame
    events: pd.DataFrame          # ground-truth passes/shots/turnovers
    meta: MatchMeta
    marking_gt: dict              # team -> {defender_entity_id: attacker_entity_id}
    possession_gt: np.ndarray     # per-frame team in possession: 'home'/'away'/'none'


class _TeamState:
    def __init__(self, key: str, profile: TeamProfile, pitch: Pitch, attack_sign: int,
                 id_base: int) -> None:
        self.key = key
        self.profile = profile
        self.pitch = pitch
        self.attack_sign = attack_sign
        self.id_base = id_base  # entity ids: base+1 = GK jersey1, base+2.. outfield
        self.slots = formation_slots(profile.formation, pitch.length, pitch.width, attack_sign)
        self.pos = self.slots.copy() + np.random.uniform(-1, 1, (10, 2))
        self.vel = np.zeros((10, 2))
        self.gk_pos = np.array([self._own_goal_x() + attack_sign * 8.0, pitch.width / 2])
        self.gk_vel = np.zeros(2)
        st = [profile.style(i) for i in range(10)]
        self.speed = np.array([s.speed for s in st])
        self.roam = np.array([s.roam for s in st])
        self.fwd_bias = np.array([s.forward_bias for s in st])
        self.press_eager = np.array([s.press_eagerness for s in st])
        self.mark_map = np.full(10, -1, dtype=int)  # opponent outfield index

    def _own_goal_x(self) -> float:
        return 0.0 if self.attack_sign > 0 else self.pitch.length

    def flip(self) -> None:
        self.attack_sign *= -1
        self.slots = formation_slots(
            self.profile.formation, self.pitch.length, self.pitch.width, self.attack_sign
        )
        self.pos = self.slots.copy() + np.random.uniform(-1, 1, (10, 2))
        self.vel[:] = 0
        self.gk_pos = np.array([self._own_goal_x() + self.attack_sign * 8.0, self.pitch.width / 2])

    def entity_id(self, idx: int) -> int:
        """idx 0..9 outfield -> base+2.., idx GK_IDX -> base+1."""
        return self.id_base + 1 if idx == GK_IDX else self.id_base + 2 + idx

    def all_pos(self) -> np.ndarray:
        return np.vstack([self.pos, self.gk_pos[None]])


class MatchSimulator:
    def __init__(self, cfg: SimulatorConfig, home: TeamProfile, away: TeamProfile,
                 pitch: Pitch | None = None) -> None:
        self.cfg = cfg
        self.pitch = pitch or Pitch()
        self.rng = np.random.default_rng(cfg.seed)
        np.random.seed(cfg.seed)  # _TeamState uses np.random for init jitter
        self.home = _TeamState("home", home, self.pitch, +1, id_base=0)
        self.away = _TeamState("away", away, self.pitch, -1, id_base=100)
        self._assign_marking(self.home, self.away)
        self._assign_marking(self.away, self.home)
        self.dt = 1.0 / cfg.fps

        # ball state
        self.ball = np.array([self.pitch.length / 2, self.pitch.width / 2])
        self.ball_vel = np.zeros(2)
        self.mode = "held"                 # held | flight | loose
        self.holder: tuple[str, int] | None = ("home", 5)
        self.flight_receiver: int | None = None
        self.flight_passer: tuple[str, int] | None = None
        self.flight_event_row: int | None = None
        self._flight_time = 0.0
        self.decision_t = 1.0

        self.rows: list[tuple] = []
        self.events: list[dict] = []
        self.possession_seq: list[str] = []

    # ------------------------------------------------------------ marking
    def _assign_marking(self, defending: _TeamState, attacking: _TeamState) -> None:
        """Fixed man-marking assignments (only used when scheme == 'man'):
        each defender/mid marks the nearest opponent by kickoff slot."""
        from scipy.optimize import linear_sum_assignment

        cost = np.linalg.norm(defending.slots[:, None] - attacking.slots[None], axis=2)
        rows, cols = linear_sum_assignment(cost)
        defending.mark_map[rows] = cols

    # ------------------------------------------------------------- helpers
    def _team(self, key: str) -> _TeamState:
        return self.home if key == "home" else self.away

    def _opponent(self, key: str) -> _TeamState:
        return self.away if key == "home" else self.home

    def possession_team(self) -> str | None:
        if self.holder is not None:
            return self.holder[0]
        if self.flight_passer is not None:
            return self.flight_passer[0]
        return None

    def _goal_center(self, team: _TeamState) -> np.ndarray:
        gx = self.pitch.length if team.attack_sign > 0 else 0.0
        return np.array([gx, self.pitch.width / 2])

    # ------------------------------------------------------------ movement
    def _player_targets(self, team: _TeamState, in_possession: bool) -> np.ndarray:
        p = team.profile
        L, W = self.pitch.length, self.pitch.width
        anchors = team.slots.copy()
        sign = team.attack_sign

        if in_possession:
            push = (8.0 + 14.0 * p.line_height) * sign
            anchors[:, 0] += push * (0.6 + 0.8 * np.clip(team.fwd_bias + 0.3, 0, 1))
            anchors[:, 1] = W / 2 + (anchors[:, 1] - W / 2) * (0.9 + 0.5 * p.width)
            # drift with ball x to support play
            anchors[:, 0] = 0.75 * anchors[:, 0] + 0.25 * self.ball[0]
        else:
            block_x = (18.0 + 30.0 * p.line_height) if sign > 0 else L - (18.0 + 30.0 * p.line_height)
            anchors[:, 0] = 0.45 * anchors[:, 0] + 0.55 * block_x
            anchors[:, 1] = W / 2 + (anchors[:, 1] - W / 2) * 0.75  # compact
            # lateral shift with ball
            anchors[:, 1] += 0.35 * (self.ball[1] - W / 2)
            anchors[:, 0] += 0.25 * (self.ball[0] - anchors[:, 0])

            if p.marking_scheme == "man":
                opp = self._opponent(team.key)
                # own goal = the goal the opponent attacks
                own_goal = self._goal_center(opp)
                marked = team.mark_map >= 0
                marked_pos = opp.pos[team.mark_map[marked]]
                # goal-side: stand 1.8m between your man and your own goal
                offsets = own_goal[None] - marked_pos
                norms = np.linalg.norm(offsets, axis=1, keepdims=True) + 1e-9
                tgt = marked_pos + 1.8 * offsets / norms
                anchors[marked] = 0.15 * anchors[marked] + 0.85 * tgt

            # pressing: most eager nearby defenders attack the ball
            dists = np.linalg.norm(team.pos - self.ball, axis=1)
            drive = p.press_intensity * team.press_eager / (1.0 + dists / 12.0)
            drive[dists > 30.0] = 0
            n_press = 1 + int(p.press_intensity * 2.0)
            for idx in np.argsort(-drive)[:n_press]:
                if drive[idx] > 0.12:
                    anchors[idx] = self.ball

        # ball attraction for everyone (small) + personal roam noise
        anchors += 0.08 * (self.ball[None] - anchors)
        anchors += self.rng.normal(0, 1.2, anchors.shape) * team.roam[:, None]

        # ball pursuit overrides shape-keeping:
        if self.mode == "flight" and self.flight_passer is not None:
            if team.key == self.flight_passer[0] and self.flight_receiver is not None \
                    and self.flight_receiver < 10:
                # intended receiver attacks the ball's path
                anchors[self.flight_receiver] = self.ball + self.ball_vel * 0.25
        elif self.mode == "loose":
            d = np.linalg.norm(team.pos - self.ball, axis=1)
            for idx in np.argsort(d)[:2]:  # two nearest chase
                anchors[idx] = self.ball

        anchors[:, 0] = np.clip(anchors[:, 0], 1.0, L - 1.0)
        anchors[:, 1] = np.clip(anchors[:, 1], 1.0, W - 1.0)
        return anchors

    def _step_kinematics(self, team: _TeamState, targets: np.ndarray) -> None:
        to_t = targets - team.pos
        dist = np.linalg.norm(to_t, axis=1, keepdims=True) + 1e-9
        desired_speed = np.clip(dist / 1.2, 0, team.speed[:, None])
        desired_vel = to_t / dist * desired_speed
        dv = desired_vel - team.vel
        dv_norm = np.linalg.norm(dv, axis=1, keepdims=True) + 1e-9
        dv = dv / dv_norm * np.minimum(dv_norm, MAX_ACCEL * self.dt)
        team.vel += dv
        # separation: repel from very close teammates
        diff = team.pos[:, None] - team.pos[None]
        d = np.linalg.norm(diff, axis=2) + np.eye(10) * 99
        close = d < 1.5
        if close.any():
            push = (diff / (d[..., None] + 1e-9) * close[..., None]).sum(axis=1)
            team.vel += 0.8 * push
        speed = np.linalg.norm(team.vel, axis=1, keepdims=True) + 1e-9
        team.vel = team.vel / speed * np.minimum(speed, team.speed[:, None])
        team.pos += team.vel * self.dt

        # goalkeeper
        own_gx = 0.0 if team.attack_sign > 0 else self.pitch.length
        depth = np.clip(abs(self.ball[0] - own_gx) * 0.12, 5.0, 13.0)
        gk_target = np.array(
            [own_gx + team.attack_sign * depth,
             self.pitch.width / 2 + np.clip(self.ball[1] - self.pitch.width / 2, -9, 9) * 0.65]
        )
        gv = np.clip(gk_target - team.gk_pos, -6 * self.dt * 10, 6 * self.dt * 10)
        team.gk_vel = 0.8 * team.gk_vel + 0.2 * gv
        step = team.gk_vel * self.dt * 6.0
        n = np.linalg.norm(step)
        if n > 6.0 * self.dt:
            step *= 6.0 * self.dt / n
        team.gk_pos += step

    # ------------------------------------------------------------ ball I/O
    def _holder_pos(self) -> np.ndarray:
        team = self._team(self.holder[0])
        return team.gk_pos if self.holder[1] == GK_IDX else team.pos[self.holder[1]]

    def _emit_event(self, frame: int, ts: float, **kw) -> int:
        row = dict(frame=frame, timestamp=ts, type="", team="", from_id=-1, to_id=-1,
                   x=np.nan, y=np.nan, end_x=np.nan, end_y=np.nan, outcome="")
        row.update(kw)
        self.events.append(row)
        return len(self.events) - 1

    def _choose_pass(self, team: _TeamState, holder_idx: int) -> tuple[int, np.ndarray] | None:
        """Pick a receiver (may be GK_IDX). Returns (receiver_idx, target_xy)."""
        opp = self._opponent(team.key)
        mates = team.all_pos()
        goal = self._goal_center(team)
        holder = mates[holder_idx]
        scores = np.full(11, -1e9)
        for j in range(11):
            if j == holder_idx:
                continue
            d = np.linalg.norm(mates[j] - holder)
            if d < 3.0 or d > (55.0 if team.profile.possession_style == "direct" else 38.0):
                continue
            # openness: distance from nearest opponent to receiver & to pass lane
            opp_all = np.vstack([opp.pos, opp.gk_pos[None]])
            recv_open = np.min(np.linalg.norm(opp_all - mates[j], axis=1))
            lane = _min_dist_to_segment(opp_all, holder, mates[j])
            fwd_gain = (np.linalg.norm(holder - goal) - np.linalg.norm(mates[j] - goal))
            direct_w = 0.9 if team.profile.possession_style == "direct" else 0.45
            scores[j] = (
                0.9 * np.clip(recv_open, 0, 12)
                + 1.1 * np.clip(lane, 0, 10)
                + direct_w * np.clip(fwd_gain, -20, 30) / 3.0
                - 0.12 * d
                + (2.0 if j == GK_IDX and self.ball[0] * team.attack_sign < 30 else -3.0 if j == GK_IDX else 0)
            )
        scores += self.rng.normal(0, 1.5, 11)
        j = int(np.argmax(scores))
        if scores[j] <= -1e8:
            return None
        return j, mates[j] + self.rng.normal(0, 1.0, 2)

    def _start_flight(self, frame: int, ts: float, team: _TeamState, holder_idx: int,
                      receiver_idx: int, target: np.ndarray) -> None:
        origin = self._holder_pos().copy()
        d = np.linalg.norm(target - origin)
        speed = np.clip(9.0 + 0.28 * d, 10.0, 19.0)
        self.ball_vel = _unit(target - origin) * speed
        self.mode = "flight"
        self._flight_time = 0.0
        self.flight_passer = (team.key, holder_idx)
        self.flight_receiver = receiver_idx
        self.holder = None
        self.flight_event_row = self._emit_event(
            frame, ts, type="pass", team=team.key,
            from_id=team.entity_id(holder_idx), to_id=team.entity_id(receiver_idx),
            x=origin[0], y=origin[1], end_x=target[0], end_y=target[1], outcome="pending",
        )

    def _ball_step(self, frame: int, ts: float) -> None:
        if self.mode == "held":
            team = self._team(self.holder[0])
            hp = self._holder_pos()
            self.ball = hp + _unit(self._goal_center(team) - hp) * 0.5
            self.ball_vel[:] = 0
            self.decision_t -= self.dt
            # tackle risk: opponent very close for a while
            opp = self._opponent(team.key)
            if np.min(np.linalg.norm(opp.pos - hp, axis=1)) < 1.1 and self.rng.random() < 0.03:
                tackler = int(np.argmin(np.linalg.norm(opp.pos - hp, axis=1)))
                self._emit_event(frame, ts, type="turnover", team=opp.key,
                                 from_id=team.entity_id(self.holder[1]),
                                 to_id=opp.entity_id(tackler), x=hp[0], y=hp[1], outcome="tackle")
                self.holder = (opp.key, tackler)
                self.decision_t = self._new_decision_time()
                return
            if self.decision_t <= 0:
                self._decide(frame, ts, team)
        elif self.mode == "flight":
            self.ball = self.ball + self.ball_vel * self.dt
            self.ball_vel *= 0.975
            passer_team = self._team(self.flight_passer[0])
            opp = self._opponent(self.flight_passer[0])
            self._flight_time += self.dt
            # reception has priority over interception
            mates = passer_team.all_pos()
            recv = self.flight_receiver
            if self._flight_time > 0.2 and np.linalg.norm(mates[recv] - self.ball) < 2.0:
                self._resolve_pass("complete")
                self.holder = (passer_team.key, recv if recv < 10 else GK_IDX)
                self.mode = "held"
                self.decision_t = self._new_decision_time()
                return
            # interception window
            d_opp = np.linalg.norm(opp.pos - self.ball, axis=1)
            close = int(np.argmin(d_opp))
            if self._flight_time > 0.3 and d_opp[close] < 0.9 and self.rng.random() < 0.12:
                self._resolve_pass("intercepted")
                self._emit_event(frame, ts, type="turnover", team=opp.key,
                                 from_id=passer_team.entity_id(self.flight_passer[1]),
                                 to_id=opp.entity_id(close),
                                 x=self.ball[0], y=self.ball[1], outcome="interception")
                self.holder = (opp.key, close)
                self.mode = "held"
                self.decision_t = self._new_decision_time()
                return
            # ball slowed down / overran target -> loose
            if np.linalg.norm(self.ball_vel) < 4.0:
                self._resolve_pass("incomplete")
                self.mode = "loose"
            self._keep_in_bounds(frame, ts)
        else:  # loose
            self.ball = self.ball + self.ball_vel * self.dt
            self.ball_vel *= 0.94
            for key in ("home", "away"):
                team = self._team(key)
                d = np.linalg.norm(team.all_pos() - self.ball, axis=1)
                j = int(np.argmin(d))
                if d[j] < 1.5:
                    self.holder = (key, j if j < 10 else GK_IDX)
                    self.mode = "held"
                    self.decision_t = self._new_decision_time()
                    return
            self._keep_in_bounds(frame, ts)

    def _resolve_pass(self, outcome: str) -> None:
        if self.flight_event_row is not None:
            self.events[self.flight_event_row]["outcome"] = outcome
        self.flight_event_row = None

    def _keep_in_bounds(self, frame: int, ts: float) -> None:
        L, W = self.pitch.length, self.pitch.width
        if 0 <= self.ball[0] <= L and 0 <= self.ball[1] <= W:
            return
        # simplify restarts: ball goes loose just inside, velocity damped
        self.ball[0] = float(np.clip(self.ball[0], 1.0, L - 1.0))
        self.ball[1] = float(np.clip(self.ball[1], 1.0, W - 1.0))
        self.ball_vel *= -0.2
        if self.mode == "flight":
            self._resolve_pass("out")
            self.mode = "loose"

    def _decide(self, frame: int, ts: float, team: _TeamState) -> None:
        holder_idx = self.holder[1]
        hp = self._holder_pos()
        goal = self._goal_center(team)
        dist_goal = np.linalg.norm(hp - goal)
        r = self.rng.random()
        shot_p = 0.35 if dist_goal < 20 else (0.12 if dist_goal < 28 else 0.0)
        if holder_idx != GK_IDX and r < shot_p:
            self._shoot(frame, ts, team, holder_idx, goal)
            return
        if r < shot_p + 0.72 or holder_idx == GK_IDX:
            choice = self._choose_pass(team, holder_idx)
            if choice is not None:
                self._start_flight(frame, ts, team, holder_idx, *choice)
                return
        # dribble: keep the ball, shorter timer
        self.decision_t = 0.5 + self.rng.random() * 0.8

    def _shoot(self, frame: int, ts: float, team: _TeamState, holder_idx: int,
               goal: np.ndarray) -> None:
        target = goal + np.array([0.0, self.rng.uniform(-3.4, 3.4)])
        r = self.rng.random()
        outcome = "goal" if r < 0.12 else ("saved" if r < 0.75 else "miss")
        self._emit_event(frame, ts, type="shot", team=team.key,
                         from_id=team.entity_id(holder_idx),
                         x=self.ball[0], y=self.ball[1],
                         end_x=target[0], end_y=target[1], outcome=outcome)
        opp = self._opponent(team.key)
        if outcome == "goal":
            self._kickoff(opp.key)
        else:  # save or miss -> opposition GK restarts
            self.holder = (opp.key, GK_IDX)
            self.mode = "held"
            self.ball = opp.gk_pos.copy()
            self.decision_t = self._new_decision_time() + 0.8

    def _kickoff(self, in_possession: str) -> None:
        self.ball = np.array([self.pitch.length / 2, self.pitch.width / 2])
        self.ball_vel[:] = 0
        self.mode = "held"
        self.holder = (in_possession, 5)
        self.decision_t = 1.0
        for t in (self.home, self.away):
            t.pos = t.slots.copy() + self.rng.normal(0, 1.0, (10, 2))
            t.vel[:] = 0

    def _new_decision_time(self) -> float:
        return 0.6 + self.rng.random() * 1.6

    # ---------------------------------------------------------------- run
    def run(self) -> SimResult:
        fps = self.cfg.fps
        half_frames = int(self.cfg.half_minutes * 60 * fps)
        total = half_frames * 2
        ref_pos = np.array([self.pitch.length / 2 - 8, self.pitch.width / 2 - 6])
        ref_vel = np.zeros(2)

        for frame in range(total):
            ts = frame / fps
            if frame == half_frames:  # halftime: flip directions, away kicks off
                self.home.flip()
                self.away.flip()
                self._assign_marking(self.home, self.away)
                self._assign_marking(self.away, self.home)
                self._kickoff("away")
                self._emit_event(frame, ts, type="kickoff", team="away")

            poss = self.possession_team()
            for team in (self.home, self.away):
                targets = self._player_targets(team, in_possession=(team.key == poss))
                self._step_kinematics(team, targets)
            self._ball_step(frame, ts)

            # referee: trail the ball diagonally
            ref_target = self.ball + np.array([-7.0, -5.0])
            rv = np.clip(ref_target - ref_pos, -1, 1) * 5.5
            ref_vel = 0.85 * ref_vel + 0.15 * rv
            sp = np.linalg.norm(ref_vel)
            if sp > 6.5:
                ref_vel *= 6.5 / sp
            ref_pos = np.clip(ref_pos + ref_vel * self.dt, 2, [self.pitch.length - 2, self.pitch.width - 2])

            self.possession_seq.append(poss or "none")
            self._record(frame, ts, ref_pos)

        return self._package(total, fps)

    def _record(self, frame: int, ts: float, ref_pos: np.ndarray) -> None:
        for team in (self.home, self.away):
            for i in range(10):
                self.rows.append((frame, ts, team.entity_id(i), "player", team.key,
                                  i + 2, team.pos[i, 0], team.pos[i, 1]))
            self.rows.append((frame, ts, team.entity_id(GK_IDX), "goalkeeper", team.key,
                              1, team.gk_pos[0], team.gk_pos[1]))
        self.rows.append((frame, ts, REFEREE_ID, "referee", "none", None, ref_pos[0], ref_pos[1]))
        self.rows.append((frame, ts, BALL_ID, "ball", "none", None, self.ball[0], self.ball[1]))

    def _package(self, total: int, fps: float) -> SimResult:
        df = pd.DataFrame(
            self.rows,
            columns=["frame", "timestamp", "entity_id", "class", "team", "jersey_no",
                     "x_pitch", "y_pitch"],
        )
        df["x_pixel"] = np.nan
        df["y_pixel"] = np.nan
        df["conf"] = 1.0
        tracking = validate_tracking_table(df)

        events = pd.DataFrame(self.events, columns=EVENT_COLUMNS)
        marking_gt = {}
        for team in (self.home, self.away):
            if team.profile.marking_scheme == "man":
                opp = self._opponent(team.key)
                marking_gt[team.key] = {
                    int(team.entity_id(i)): int(opp.entity_id(int(team.mark_map[i])))
                    for i in range(10) if team.mark_map[i] >= 0
                }
        meta = MatchMeta(
            fps=fps,
            n_frames=total,
            pitch_length=self.pitch.length,
            pitch_width=self.pitch.width,
            team_names={"home": self.home.profile.name, "away": self.away.profile.name},
            kit_colors={"home": self.home.profile.kit_color, "away": self.away.profile.kit_color},
            attack_direction={"home": 1, "away": -1},
            source="synthetic-simulator",
            extras={
                "halftime_frame": total // 2,
                "profiles": {
                    "home": self.home.profile.__dict__ | {"player_styles": None},
                    "away": self.away.profile.__dict__ | {"player_styles": None},
                },
            },
        )
        return SimResult(tracking, events, meta, marking_gt,
                         np.array(self.possession_seq))


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros_like(v)


def _min_dist_to_segment(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Min distance from any of pts to segment ab."""
    ab = b - a
    denom = float(ab @ ab) + 1e-9
    t = np.clip(((pts - a) @ ab) / denom, 0, 1)
    proj = a + t[:, None] * ab
    return float(np.min(np.linalg.norm(pts - proj, axis=1)))


def simulate_demo_match(cfg: SimulatorConfig) -> SimResult:
    """The bundled fixture (see :func:`pitchiq.demo.profiles.demo_profiles`)."""
    from pitchiq.demo.profiles import demo_profiles

    home, away = demo_profiles()
    return MatchSimulator(cfg, home, away).run()
