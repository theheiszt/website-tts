#!/usr/bin/env bash
# Start AIG Companion Speech Synthesis Microservice on Linux Cloud Server
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR"

export PORT="${PORT:-8880}"
export HOST="${HOST:-0.0.0.0}"
export MODEL_DIR="${MODEL_DIR:-$DIR/models/tts/supertonic3}"

echo "Starting AIG TTS Service on port $PORT..."
python3 tts_service.py
