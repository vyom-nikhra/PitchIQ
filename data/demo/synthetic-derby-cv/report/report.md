# Match Report — Crimson City vs Azure United

## Executive Summary
Azure United controlled the ball (81% possession, field tilt 0.71 in favour of Crimson City). The more aggressive press came from Crimson City (PPDA 0.2 vs 0.2).
Data confidence: **medium**. Only ~9 players visible per frame (broadcast framing) — team-shape and pitch-control metrics describe the visible players.

## Tactical Narrative
**Crimson City** set up in a 3-5-2 with the ball (width 30.9 m) and a 4-3-3 without it — 3-5-2 in possession → 4-3-3 out of possession.
Off the ball they defended in a high line (defensive line 55.3 m from goal, depth 19.7 m).
**Azure United** set up in a 3-4-3 with the ball (width 38.8 m) and a 4-5-1 without it — 3-4-3 in possession → 4-5-1 out of possession.
Off the ball they defended in a mid block (defensive line 30.3 m from goal, depth 23.5 m).
Crimson City registered 24 pressures (6.0/min), average press height 66.1 m, converting 21% into turnovers within 3 s.
Azure United registered 19 pressures (4.75/min), average press height 26.2 m, converting 42% into turnovers within 3 s.

## Key Players
- player 50 (Azure United) created the most threat (xT +0.028 over 1 moves).
- player 243 (Azure United) created the most threat (xT +0.010 over 1 moves).
- player 96 (Azure United) created the most threat (xT +0.001 over 1 moves).
- player 25 (Azure United) was Azure United's passing hub (highest betweenness centrality).

## Defensive Schemes
**Crimson City**: hybrid (man-score 0.532).
  - player 148 (Crimson City) marked player 146 (Azure United) (100% of defensive samples).
  - player 154 (Crimson City) marked player 171 (Azure United) (100% of defensive samples).
  - player 107 (Crimson City) marked player 166 (Azure United) (83% of defensive samples).
  - player 140 (Crimson City) marked player 149 (Azure United) (80% of defensive samples).
**Azure United**: zonal (man-score 0.438).
  - player 701 (Azure United) marked player 696 (Crimson City) (80% of defensive samples).

## Watch-outs & Data Notes
- Events (passes/turnovers) are derived from tracking, not a manual event feed; ball-dependent metrics inherit ball-tracking noise.
- Only ~9 players visible per frame (broadcast framing) — team-shape and pitch-control metrics describe the visible players.
- Detection ran on the fallback stack (blob-fallback) — the trained football detector was not available for this run.

## Metrics Appendix

Auto-generated from `facts.json`; every figure above traces here.

- **possession**: `{"share": {"away": 0.8095, "home": 0.1905}, "n_spells": 41, "avg_spell_s": {"away": 6.46, "home": 4.14}, "longest_spell_s": {"away": 24.0, "home": 13.6}}`
- **field_tilt**: `{"away": {"final_third_share": 0.218, "own_third_share": 0.487, "mean_ball_x_att": 44.3}, "home": {"final_third_share": 0.545, "own_third_share": 0.312, "mean_ball_x_att": 59.3}, "tilt_home": 0.714}`
- **ppda**: `{"home": {"ppda": 0.2, "opp_buildup_passes": 5, "defensive_actions": 25}, "away": {"ppda": 0.25, "opp_buildup_passes": 1, "defensive_actions": 4}}`
- **pitch_control**: `{"home_mean_control": 0.497, "home_final_third_control": 0.474}`
- **team_distance_m**: `{"home": 2344.0, "away": 3468.0}`
- **line_breaking**: `{"total": 0, "by_team": {}}`
- **events**: `{"n_passes": 8, "n_completed_passes": 6, "n_turnovers": 13, "n_carries": 1}`
- **data_quality**: `{"overall": "medium", "is_cv": true, "levels": {"calibration": "high", "ball": "high", "tracking": "medium", "teams": "medium"}, "components": {"calibration_coverage": 1.0, "calibration_reproj_px_median": 0.9, "calibration_method_mix": {"flow": 4819, "lines": 1181}, "ball_frame_coverage": 1.0, "ball_observed_share": 0.965, "ball_observed_is_estimated": true, "ball_conf_median": 0.5, "players_per_frame_median": 9.0, "n_player_tracks": 550, "track_len_s_median": 2.0, "team_separability": 2.05, "de`
