"""Typed, YAML-driven configuration for every PitchIQ component.

Usage::

    from pitchiq.config import load_config
    cfg = load_config()                          # configs/default.yaml
    cfg = load_config("configs/demo.yaml")       # deep-merged over defaults
    cfg = load_config(overrides={"video": {"max_frames": 500}})

The tree is validated into pydantic models so components fail fast on typos
and every module receives only its own section (e.g. ``cfg.calibration``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


class PitchConfig(BaseModel):
    length_m: float = 105.0
    width_m: float = 68.0


class VideoConfig(BaseModel):
    target_fps: Optional[float] = 25.0
    max_frames: Optional[int] = None


class BallConfig(BaseModel):
    roi_inference: bool = True
    roi_size: int = 640
    conf_threshold: float = 0.15
    max_gap_interpolate: int = 25


class DetectionConfig(BaseModel):
    backend: Literal["auto", "yolo", "rtdetr", "blob"] = "auto"
    weights: Optional[str] = None
    model: str = "yolo11n.pt"
    conf_threshold: float = 0.30
    imgsz: int = 1280
    device: str = "auto"
    coco_fallback: bool = True
    ball: BallConfig = Field(default_factory=BallConfig)


class AppearanceConfig(BaseModel):
    enabled: bool = True
    weight: float = 0.30
    backend: Literal["auto", "osnet", "colorhist"] = "auto"


class TrackingConfig(BaseModel):
    high_thresh: float = 0.5
    low_thresh: float = 0.10
    new_track_thresh: float = 0.60
    match_thresh: float = 0.80
    second_match_thresh: float = 0.50
    unconfirmed_match_thresh: float = 0.70
    max_lost: int = 30
    min_track_len: int = 5
    appearance: AppearanceConfig = Field(default_factory=AppearanceConfig)
    camera_motion_compensation: bool = True


class TeamsConfig(BaseModel):
    method: Literal["kmeans_lab", "embed"] = "kmeans_lab"
    embed_backend: Literal["auto", "siglip", "cnn", "none"] = "auto"
    torso_top: float = 0.12
    torso_bottom: float = 0.52
    torso_inset: float = 0.22
    min_samples_per_track: int = 3
    gk_by_position: bool = True
    min_torso_height_px: int = 22  # skip crops too small for a reliable kit signature
    min_non_grass_px: int = 40     # min non-grass pixels required in a torso crop


class JerseyConfig(BaseModel):
    enabled: bool = True
    backend: Literal["auto", "easyocr", "none"] = "auto"
    min_conf: float = 0.55
    min_votes: int = 2
    min_height_px: int = 40


class SmoothingConfig(BaseModel):
    enabled: bool = True
    alpha: float = 0.5


class CalibrationConfig(BaseModel):
    method: Literal["auto", "lines", "keypoints", "manual"] = "auto"
    keypoint_weights: Optional[str] = None
    every_n_frames: int = 2
    max_reproj_error_px: float = 25.0
    min_line_score: float = 0.25
    scene_cut_threshold: float = 0.45
    propagate_with_flow: bool = True
    smoothing: SmoothingConfig = Field(default_factory=SmoothingConfig)


class PossessionConfig(BaseModel):
    control_radius_m: float = 2.0
    hysteresis_frames: int = 6
    dead_ball_speed_mps: float = 0.5


class KinematicsConfig(BaseModel):
    smooth_window_s: float = 0.5
    max_speed_mps: float = 11.0
    sprint_speed_mps: float = 7.0
    hi_speed_mps: float = 5.5
    min_sprint_duration_s: float = 0.7


class HeatmapConfig(BaseModel):
    nx: int = 52
    ny: int = 34
    sigma_cells: float = 1.5


class FormationsConfig(BaseModel):
    window_s: float = 20.0
    min_window_players: int = 9


class PitchControlConfig(BaseModel):
    grid_nx: int = 52
    grid_ny: int = 34
    reaction_time_s: float = 0.7
    max_speed_mps: float = 7.8
    kappa: float = 0.45


class PassesConfig(BaseModel):
    min_pass_dist_m: float = 3.0
    max_pass_time_s: float = 4.0
    min_carry_dist_m: float = 5.0


class XTConfig(BaseModel):
    grid_nx: int = 16
    grid_ny: int = 12
    n_iterations: int = 60
    grid_path: Optional[str] = None


class PressingConfig(BaseModel):
    press_radius_m: float = 4.0
    closing_speed_mps: float = 0.5
    ppda_zone_frac: float = 0.60
    intense_press_dist_m: float = 2.5


class PhasesConfig(BaseModel):
    dead_ball_speed_mps: float = 0.3
    dead_ball_secs: float = 1.5
    transition_window_s: float = 5.0
    third_boundaries_m: tuple[float, float] = (35.0, 70.0)
    high_block_x_m: float = 65.0
    low_block_x_m: float = 40.0


class LearnedEmbeddingConfig(BaseModel):
    enabled: Literal["auto", True, False] = "auto"
    weights: Optional[str] = None
    dim: int = 64


class EmbeddingsConfig(BaseModel):
    heatmap_nx: int = 24
    heatmap_ny: int = 16
    pca_dim: int = 16
    learned: LearnedEmbeddingConfig = Field(default_factory=LearnedEmbeddingConfig)


class RolesConfig(BaseModel):
    n_clusters: int | Literal["auto"] = "auto"
    min_minutes: float = 1.0


class SimilarityConfig(BaseModel):
    backend: Literal["auto", "faiss", "sklearn"] = "auto"
    top_k: int = 8


class MarkingConfig(BaseModel):
    window_s: float = 4.0
    step_s: float = 1.0
    max_pair_dist_m: float = 15.0
    stability_threshold: float = 0.60
    min_defensive_frames: int = 50


class LLMConfig(BaseModel):
    provider: Literal["auto", "gemini", "anthropic", "none"] = "auto"
    model: str = "auto"  # 'auto' = provider default (gemini-flash-latest / claude-sonnet-5)
    max_tokens: int = 3000
    temperature: float = 0.2


class ReportConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)


class AppConfig(BaseModel):
    artifacts_root: str = "data/jobs"
    demo_root: str = "data/demo"
    api_url: str = "http://localhost:8000"
    radar_fps: float = 12.5


class SimulatorConfig(BaseModel):
    fps: float = 25.0
    half_minutes: float = 2.0
    seed: int = 7


class Config(BaseModel):
    """Root configuration object handed to the pipeline."""

    pitch: PitchConfig = Field(default_factory=PitchConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    teams: TeamsConfig = Field(default_factory=TeamsConfig)
    jersey: JerseyConfig = Field(default_factory=JerseyConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    possession: PossessionConfig = Field(default_factory=PossessionConfig)
    kinematics: KinematicsConfig = Field(default_factory=KinematicsConfig)
    heatmaps: HeatmapConfig = Field(default_factory=HeatmapConfig)
    formations: FormationsConfig = Field(default_factory=FormationsConfig)
    pitch_control: PitchControlConfig = Field(default_factory=PitchControlConfig)
    passes: PassesConfig = Field(default_factory=PassesConfig)
    xt: XTConfig = Field(default_factory=XTConfig)
    pressing: PressingConfig = Field(default_factory=PressingConfig)
    phases: PhasesConfig = Field(default_factory=PhasesConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    roles: RolesConfig = Field(default_factory=RolesConfig)
    similarity: SimilarityConfig = Field(default_factory=SimilarityConfig)
    marking: MarkingConfig = Field(default_factory=MarkingConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    app: AppConfig = Field(default_factory=AppConfig)
    simulator: SimulatorConfig = Field(default_factory=SimulatorConfig)

    def config_hash(self) -> str:
        """Stable hash of the perception-relevant config, used to invalidate caches."""
        import hashlib
        import json

        relevant = {
            "video": self.video.model_dump(),
            "detection": self.detection.model_dump(),
            "tracking": self.tracking.model_dump(),
            "teams": self.teams.model_dump(),
            "jersey": self.jersey.model_dump(),
            "calibration": self.calibration.model_dump(),
        }
        blob = json.dumps(relevant, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` (override wins)."""
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Load configuration: defaults <- optional file <- optional overrides dict."""
    tree: dict[str, Any] = {}
    if DEFAULT_CONFIG_PATH.exists():
        with open(DEFAULT_CONFIG_PATH, encoding="utf-8") as fh:
            tree = yaml.safe_load(fh) or {}
    if path is not None and Path(path) != DEFAULT_CONFIG_PATH:
        with open(path, encoding="utf-8") as fh:
            tree = _deep_merge(tree, yaml.safe_load(fh) or {})
    if overrides:
        tree = _deep_merge(tree, overrides)
    return Config.model_validate(tree)
