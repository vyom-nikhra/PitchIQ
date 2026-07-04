# Deployment

## Local (bare)

```bash
pip install -e .[app,ml]           # + torch CPU: pip install torch --index-url https://download.pytorch.org/whl/cpu
python scripts/build_demo.py       # one-time: bundled demo artifacts (~10 min)
streamlit run pitchiq/app/ui.py    # dashboard on :8501
uvicorn pitchiq.app.api:app --port 8000   # optional API service
```

## Local (Docker)

```bash
docker compose up --build          # UI :8501, API :8000, ./data mounted
# or the single all-in-one container:
docker build -t pitchiq . && docker run -p 7860:7860 --env-file .env pitchiq
```

The image bakes in `data/demo/` so the dashboard is explorable immediately.

## Public — Hugging Face Spaces (free tier, recommended)

1. Create a Space → SDK **Docker** → visibility public.
2. Push this repository to the Space (`git remote add space
   https://huggingface.co/spaces/<you>/pitchiq && git push space main`).
   The README's YAML front-matter sets `app_port: 7860`.
3. Space → Settings → *Variables and secrets* → add `GEMINI_API_KEY`
   (secret). Without it the app still works — reports fall back to the
   deterministic template.
4. Free CPU tier notes: precomputed demo matches are instant; fresh uploads
   run the full CV pipeline at CPU speed (minutes per video minute) — keep
   uploads short, or attach a GPU to the Space to lift this.

## Secrets in production

Only environment variables. Never bake keys into images (the Dockerfile
copies no `.env`; `.dockerignore` blocks it). The app reads them through
`pitchiq.core.env.get_secret` at runtime.

## Production hardening checklist (beyond the demo)

- Replace the in-process worker thread with a real queue (Celery/RQ + Redis)
  and object storage for artifacts.
- Put the FastAPI service behind auth if uploads are public.
- GPU inference for YOLO + keypoint model; batch video decoding.
- The Streamlit UI is deliberately thin over `ArtifactStore` + JSON artifacts
  — a React frontend can consume the FastAPI endpoints unchanged.
