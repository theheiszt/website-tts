# syntax=docker/dockerfile:1
# Cloud-Ready Multi-Stage Dockerfile for AIG Companion TTS Microservice

FROM python:3.11-slim as runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8880 \
    HOST=0.0.0.0 \
    MODEL_DIR=/app/models/tts/supertonic3

WORKDIR /app

# Install system audio and compilation libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications and install Python packages
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy engine code and model assets
COPY supertonic /app/supertonic
COPY tts_service.py /app/
COPY models/tts/supertonic3 /app/models/tts/supertonic3

# Create non-root user for cloud security hardening
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8880

# Cloud Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8880/health/live || exit 1

CMD ["python", "tts_service.py"]
