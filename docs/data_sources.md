# Data sources & licensing

PitchIQ's bundled demo is **fully synthetic** (our own simulator + renderer),
so the public repository and deployment carry no third-party data. External
datasets plug in under these rules:

| Source | What | Licence / constraint | Where it lives |
|---|---|---|---|
| Synthetic simulator | demo matches + ground truth | ours (MIT) | `data/demo/` (committed) |
| Metrica Sports sample data | real tracking (validation) | public GitHub sample | `data/downloads/metrica/` (ignored) |
| StatsBomb open data | event data (validation) | free w/ attribution, [terms](https://github.com/statsbomb/open-data) | `data/downloads/statsbomb/` (ignored) |
| Roboflow `football-players-detection` | detection fine-tune | open licence (CC BY 4.0) — Kaggle training OK | `data/downloads/roboflow/` (ignored) |
| SoccerNet | calibration/tracking/jersey data | **NDA — see below** | `data/soccernet/` (hard-ignored) |

## SoccerNet NDA — operating rules

The SoccerNet password grants access under a KAUST non-disclosure agreement:
*non-commercial research use only, no redistribution to third parties.*
Concretely, in this project:

1. **Nothing from SoccerNet enters git or any public artifact** — videos,
   annotation JSONs, labels, crops, frames, or file listings. The
   `data/soccernet/` path is blocked in `.gitignore`; keep it that way.
2. **No third-party uploads** — not to Kaggle (even private datasets), cloud
   drives, or CI. Training on SoccerNet happens **locally only**
   (`scripts/train_pitch_keypoints.py`; a consumer GPU suffices).
3. **Derived weights** are kept out of the public repo by default. Using them
   *server-side* in a non-commercial research demo is within the permitted
   purpose; redistributing the weight files is a judgement call the
   repository owner must make explicitly, not a default.
4. The Roboflow detector fine-tune is the Kaggle-friendly training task —
   its licence permits it.

## Secrets

API keys (`GEMINI_API_KEY`, `ROBOFLOW_API_KEY`, `SOCCERNET_PASSWORD`) live in
`.env` (gitignored; template in `.env.example`) and are read only through
`pitchiq.core.env.get_secret`. Error paths scrub key values before logging.
For deployment, inject them as platform secrets (e.g. HF Spaces → Settings →
Variables and secrets) — never bake them into images or commits. **Rotate any
key that has ever been shared over chat/email once the project stabilises.**
