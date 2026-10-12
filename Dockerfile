# ArboTTS Ultra-Lightweight Standalone CPU Container
FROM python:3.11-slim

# Install system audio and phonetics libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency specifications first for layer caching
COPY pyproject.toml README.md ./

# Copy package source code
COPY arbotts/ ./arbotts/
COPY configs/ ./configs/
COPY arbotts_v1.onnx.json ./

# Install ArboTTS with web server dependencies
RUN pip install --no-cache-dir .[web]

# Expose port for Web UI Studio & REST API
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Launch server
CMD ["python", "-m", "arbotts.server", "--host", "0.0.0.0", "--port", "8000"]
