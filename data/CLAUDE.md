# data/ — all gitignored; part cache, part licensed material

- `soccernet/`, `raw/soccernet_*.mp4` — NDA material (rules in root CLAUDE.md).
- `raw/` — input clips; copyrighted broadcast footage, local analysis only.
- `jobs/<name>/` — pipeline outputs, fully regenerable with
  `scripts/process_clip.py`; deleting a job dir is the clean way to force a re-run.
- `demo/` — synthetic demo matches; the **only** footage licence-clean for
  public demos/videos (regenerate: `scripts/build_demo.py`).
- `downloads/` — third-party datasets, re-fetchable.
