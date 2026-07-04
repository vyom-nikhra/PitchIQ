"""Synthetic match generation: agent-based simulator + broadcast renderer.

Purpose:
1. **Bundled demo** — the app ships with a pre-computed synthetic match so the
   full dashboard is explorable without waiting for CV processing.
2. **Ground truth** — the simulator knows every position, pass and marking
   assignment, and the renderer knows the exact per-frame camera homography,
   so Layer-1 (calibration error in metres, MOTA/IDF1) and Layer-2/3 (pass
   precision/recall, marking recovery) are validated quantitatively without
   external datasets. Synthetic validation is necessary but not sufficient —
   real-broadcast caveats are documented in the README.
"""
