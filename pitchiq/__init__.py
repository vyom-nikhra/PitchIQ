"""PitchIQ — football tactical intelligence from broadcast video.

Three layers:
  1. Perception   — detection, tracking, team assignment, jersey OCR, homography
  2. Analytics    — possession, kinematics, formations, pitch control, passes, pressing, xT
  3. Intelligence — style embeddings, role discovery, similar-player search, marking analysis

Layer 1 emits the *tracking table* (parquet): every entity's real-pitch (x, y)
per frame. Layers 2–3 are pure analysis over that table and never touch video.
"""

__version__ = "0.1.0"
