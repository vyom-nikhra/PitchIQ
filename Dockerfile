# PitchIQ — single-container deployment (Streamlit UI + FastAPI backend).
# Suitable for Hugging Face Spaces (Docker SDK, port 7860) and local runs.
FROM python:3.13-slim

# opencv-headless needs glib even without GUI bits
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU torch first (small wheel index), then the package with app extras.
# constraints.txt pins the exact known-good versions so an upstream release
# can't silently break the deployed Space.
COPY pyproject.toml README.md constraints.txt ./
RUN pip install --no-cache-dir -c constraints.txt torch --index-url https://download.pytorch.org/whl/cpu
COPY pitchiq ./pitchiq
COPY configs ./configs
COPY scripts ./scripts
RUN pip install --no-cache-dir -c constraints.txt -e .[app,ml]

# bundled demo artifacts + (licence-clean, sim-trained) encoder weights
COPY data/demo ./data/demo
COPY weights ./weights

COPY .streamlit ./.streamlit
COPY docker/start.sh ./docker/start.sh
RUN chmod +x docker/start.sh \
    && mkdir -p data/jobs \
    && useradd -m appuser && chown -R appuser /app
USER appuser

# secrets (GEMINI_API_KEY etc.) are injected as env vars at runtime — never baked in
ENV PORT=7860 \
    PYTHONUNBUFFERED=1

EXPOSE 7860 8000
CMD ["./docker/start.sh"]
