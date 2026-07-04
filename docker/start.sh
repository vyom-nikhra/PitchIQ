#!/usr/bin/env bash
# Run the FastAPI backend alongside the Streamlit dashboard in one container.
set -e

uvicorn pitchiq.app.api:app --host 0.0.0.0 --port 8000 &

exec streamlit run pitchiq/app/ui.py \
    --server.port "${PORT:-7860}" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false \
    --server.maxUploadSize 500
