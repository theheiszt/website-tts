# AIG Companion Speech Synthesis Engine (Supertonic 3)

Ultra-fast, private, CD-quality (44.1 kHz) neural text-to-speech microservice designed for multi-user live AI companion conversations.

---

## 🚀 Key Features

- **Multi-User Real-Time Latency**: <120 ms time-to-first-audio with ONNX Runtime multi-threaded C++ backend.
- **Audible Voice Customization**: Real pitch shifting (-3.5 to +3.5 semitones), parametric warmth EQ, gentle sibilance de-essing, breath-aware pause pacing, and soft-knee peak limiting.
- **Mathematical Voice Morphing / Blending**: Linear interpolation between high-dimensional style latents ($50 \times 256$ text-to-latent timbre and $8 \times 16$ duration predictor rhythm) allowing infinite unique vocal identities.
- **In-Memory LRU Utterance Cache**: Instant (<10 ms) replay for repeated companion prompts and greetings.
- **Cloud-Ready**: Headless Docker container with non-root security, healthcheck probes (`/health/live`, `/health/ready`), Prometheus-compatible metrics (`/v1/metrics`), and optional API key authentication (`X-API-Key`).

---

## 🎙️ Included Voices

| Voice ID | Display Name | Gender | Vocal Archetype | Base Style File |
|---|---|---|---|---|
| `warm-01` / `F1` | Elena | Female | Warm, relaxed, intimate companion | `F1.json` |
| `calm-01` / `F2` | Solène | Female | Calm, thoughtful, soothing & soft | `F2.json` |
| `bright-01` / `F3` | Aria | Female | Bright, playful, upbeat & energetic | `F3.json` |
| `velvet-01` / `F4` | Amara | Female | Deep, confident, composed & mature | `F4.json` |
| `artistic-01` / `F5` | Lucía | Female | Kind, gentle, affectionate & sweet | `F5.json` |
| `clone-jennie` | Jennie | Female | Zero-shot neural cloned voice | `Jennie_clone_clip.json` |
| `julian-01` / `M2` | Julian | Male | Deep, composed, documentary baritone | `M2.json` |
| `gabriel-01` / `M5` | Gabriel | Male | Warm, soft-spoken, reflective narrator | `M5.json` |
| `M1`, `M3`, `M4` | Marcus, Alex, Leo | Male | Upbeat, executive, and casual narrators | `M1.json`, `M3.json`, `M4.json` |

---

## 🛠️ Quick Start (Local Run)

### Requirements
- Python 3.10+ (tested up to Python 3.14)
- ONNX Runtime, FastAPI, uvicorn, scipy, soundfile, numpy

### Windows
Double-click `start_service.bat` or run:
```powershell
pip install -r requirements.txt
python tts_service.py
```

### Linux / macOS
```bash
chmod +x start_service.sh
./start_service.sh
```

The service will start at `http://127.0.0.1:8880`.

---

## ☁️ Cloud Deployment

### Method 1: Docker Compose (Recommended for Any VPS / EC2)

```bash
# Build and run in background
docker compose up -d

# Check logs
docker compose logs -f

# Verify readiness
curl http://localhost:8880/health/ready
```

### Method 2: Google Cloud Run (Serverless)

Build and push the container to Google Artifact Registry:
```bash
# 1. Build & tag
docker build -t gcr.io/<PROJECT_ID>/aig-tts:latest .

# 2. Push to Google Container Registry / Artifact Registry
docker push gcr.io/<PROJECT_ID>/aig-tts:latest

# 3. Deploy to Cloud Run (1 vCPU, 2GB RAM minimum recommended)
gcloud run deploy aig-tts \
  --image gcr.io/<PROJECT_ID>/aig-tts:latest \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --port 8880 \
  --allow-unauthenticated \
  --set-env-vars="DEFAULT_STEPS=12,CACHE_SIZE=1000"
```

### Method 3: Systemd Service on Linux VPS

Create `/etc/systemd/system/aig-tts.service`:
```ini
[Unit]
Description=AIG Companion TTS Microservice
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/aig-voice
ExecStart=/usr/bin/python3 tts_service.py
Restart=always
RestartSec=3
Environment=PORT=8880
Environment=HOST=0.0.0.0
Environment=MODEL_DIR=/opt/aig-voice/models/tts/supertonic3

[Install]
WantedBy=multi-user.target
```
Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now aig-tts
```

---

## 🌐 Connecting to the AIGirl.fans Website

In your website root (`F:\aigirl.fans` or production `.env`):
```env
# Set this to your local or deployed cloud TTS endpoint
SUPERTONIC_ONNX_URL="http://127.0.0.1:8880"
# Or in production:
# SUPERTONIC_ONNX_URL="https://tts.aigirl.fans"

# Optional shared API key
# TTS_API_KEY="your-secret-api-key"
```

The website server (`server.ts` & `src/server/ttsRegistry.ts`) automatically forwards all voice shaping sliders, vibe presets, and voice blending parameters to this service.

---

## 📡 API Reference

### `POST /v1/tts/synthesize`
Generates speech audio from text.

**Payload**:
```json
{
  "text": "Hey, it’s nice to hear from you. How has your day been?",
  "voice": "warm-01",
  "speed": 1.0,
  "pitch": 1.0,
  "warmth": 65,
  "softness": 55,
  "pauses": 50,
  "steps": 12,
  "style": "natural",
  "blend_weights": {
    "warm-01": 0.70,
    "artistic-01": 0.30
  },
  "bypass_cache": false,
  "return_base64": true
}
```

**Response**:
```json
{
  "status": "ok",
  "audioBase64": "UklGRi...",
  "duration": 3.76,
  "sampleRate": 44100,
  "cached": false
}
```
*(If `return_base64: false`, returns binary `audio/wav` streaming directly).*

### `GET /v1/tts/voices`
Lists all available voices, descriptions, and custom styles.

### `GET /v1/tts/voice-space`
Returns the 4 Principal Components capturing 76% of all voice variance across the 10 built-ins, along with coordinate maps and axis definitions.

### `POST /v1/tts/randomize`
Generates a Dirichlet-randomized multi-voice blend recipe and returns synthesized audio.

### `POST /v1/tts/save-custom`
Saves any blended or PCA-generated style vector to disk as a persistent custom voice JSON.

### `POST /v1/audio/speech`
OpenAI Audio Speech API compatible endpoint (`{"model": "supertonic-3", "input": "...", "voice": "Elena"}`).

### `GET /health/ready`
Returns `{"status": "ready", "model": "Supertonic 3", "sample_rate": 44100, "voices_loaded": 11}`.

### `GET /v1/metrics`
Returns total requests processed, cache hit ratio, and average synthesis latency in milliseconds.

---

## 🎭 Adding Custom Cloned Voices

To add a new voice:
1. Place the voice style JSON into `models/tts/supertonic3/voice_styles/custom/<your_voice_name>.json`.
2. The service automatically detects and loads the new voice upon startup or restart.
3. Access it by name via API or assign it to any companion in `F:\aigirl.fans`.
# website-tts
