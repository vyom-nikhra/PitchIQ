#!/usr/bin/env bash
# Run the FastAPI backend alongside the Streamlit dashboard in one container.
set -e

uvicorn pitchiq.app.api:app --host 0.0.0.0 --port 8000 &

# XSRF/CORS protection must be disabled behind Hugging Face Spaces' reverse
# proxy: the XSRF-token cookie flow breaks through the proxy, which makes the
# file_uploader POST fail with a 403 (AxiosError). Safe here — this is a
# single-container app with no cross-origin surface.
exec streamlit run pitchiq/app/ui.py \
    --server.port "${PORT:-7860}" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false \
    --server.enableXsrfProtection false \
    --server.enableCORS false \
    --server.maxUploadSize 500
