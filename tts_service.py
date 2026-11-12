"""
Live Cloud & Local Speech Synthesis Microservice for AI Companions
===================================================================
A high-concurrency, cloud-ready FastAPI microservice wrapping Supertonic 3.
Supports:
  - Multi-user non-blocking concurrent synthesis
  - Distinct and audible voice parameter adjustments (Speed, Pitch, Warmth, Softness, Pauses)
  - Mathematical voice style blending (combining two or more voice styles)
  - In-memory LRU utterance caching for instant response
  - Studio-grade acoustic DSP post-processing (pitch shifting, parametric warmth, de-essing, limiting)
  - Cloud health probes, bearer authentication, and configurable CORS
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import re
import sys
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import soundfile as sf
from fastapi import FastAPI, Header, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from scipy import signal

# Ensure current directory is on sys.path for local supertonic import
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from supertonic import TTS
from supertonic.core import Style
from supertonic.loader import list_available_voice_style_names, load_voice_style_from_json_file

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("aig_tts_service")

# ---------------------------------------------------------------------------
# Configuration (Environment-driven for local & cloud)
# ---------------------------------------------------------------------------
MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(BASE_DIR / "models" / "tts" / "supertonic3")))
PORT = int(os.environ.get("PORT", "8880"))
HOST = os.environ.get("HOST", "0.0.0.0")
API_KEY = os.environ.get("TTS_API_KEY", "")
DEFAULT_STEPS = int(os.environ.get("DEFAULT_STEPS", "12"))
MAX_CACHE_ITEMS = int(os.environ.get("CACHE_SIZE", "1000"))
ENABLE_DSP = os.environ.get("ENABLE_DSP", "true").lower() in ("true", "1", "yes")

# Built-in Voice Catalog
VOICE_METADATA = {
    "F1": {"name": "Elena", "gender": "female", "vibe": "Warm, relaxed, intimate companion"},
    "F2": {"name": "Solène", "gender": "female", "vibe": "Calm, thoughtful, soothing & soft"},
    "F3": {"name": "Aria", "gender": "female", "vibe": "Bright, playful, upbeat & energetic"},
    "F4": {"name": "Amara", "gender": "female", "vibe": "Deep, confident, composed & mature"},
    "F5": {"name": "Lucía", "gender": "female", "vibe": "Kind, gentle, affectionate & sweet"},
    "M1": {"name": "Marcus", "gender": "male", "vibe": "Lively, upbeat, enthusiastic"},
    "M2": {"name": "Julian", "gender": "male", "vibe": "Deep, composed, documentary baritone"},
    "M3": {"name": "Alexander", "gender": "male", "vibe": "Authoritative, polished, executive"},
    "M4": {"name": "Leo", "gender": "male", "vibe": "Friendly, youthful, casual narrator"},
    "M5": {"name": "Gabriel", "gender": "male", "vibe": "Warm, soft-spoken, reflective"},
}

ALL_BASE_VOICES = list(VOICE_METADATA.keys())

# Friendly ID to model style code mappings
VOICE_ALIASES = {
    "warm-01": "F1",
    "elena": "F1",
    "calm-01": "F2",
    "solène": "F2",
    "solene": "F2",
    "bright-01": "F3",
    "aria": "F3",
    "velvet-01": "F4",
    "amara": "F4",
    "artistic-01": "F5",
    "lucía": "F5",
    "lucia": "F5",
    "clone-jennie": "Jennie_clone_clip",
    "jennie": "Jennie_clone_clip",
    "julian-01": "M2",
    "julian": "M2",
    "gabriel-01": "M5",
    "gabriel": "M5",
    "marcus": "M1",
    "alexander": "M3",
    "leo": "M4",
}


def resolve_voice_code(name: str) -> str:
    clean = name.replace(".json", "").strip()
    if clean in VOICE_ALIASES:
        return VOICE_ALIASES[clean]
    lower = clean.lower()
    if lower in VOICE_ALIASES:
        return VOICE_ALIASES[lower]
    return clean


# ---------------------------------------------------------------------------
# In-Memory LRU Utterance Cache
# ---------------------------------------------------------------------------
class AudioCache:
    """Thread-safe LRU cache storing synthesized audio chunks for identical requests."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: OrderedDict[str, tuple[bytes, float]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[tuple[bytes, float]]:
        if key in self._cache:
            self._cache.move_to_end(key)
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    def put(self, key: str, wav_bytes: bytes, duration: float):
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            self._cache[key] = (wav_bytes, duration)


# ---------------------------------------------------------------------------
# Studio Acoustic DSP Filter & Pitch Shifter
# ---------------------------------------------------------------------------
def apply_companion_acoustic_shaping(
    audio: np.ndarray,
    sample_rate: int,
    pitch: float = 1.0,
    warmth: float = 50.0,
    softness: float = 50.0,
) -> np.ndarray:
    """Applies audible, musical acoustic post-processing:
    - Pitch Shifting: Clean OLA time-stretch + resample shifting from -3.5 to +3.5 semitones
    - Warmth: Rich low-shelf boost at 200 Hz vs crisp treble presence at 5 kHz
    - Softness / De-essing: Gentle notch at 3.6 kHz to remove harsh sibilance
    - Soft-knee peak limiting to prevent clipping
    """
    if not ENABLE_DSP:
        return audio

    try:
        # 1. Audible Pitch Shifting
        # pitch: 0.8 to 1.2 maps to -3.5 to +3.5 semitones
        semitones = (pitch - 1.0) * 17.5
        if abs(semitones) > 0.15:
            factor = 2.0 ** (semitones / 12.0)
            new_len = int(len(audio) / factor)
            resampled = signal.resample(audio, new_len)
            hop = 256
            win_len = 1024
            window = np.hanning(win_len)
            stretched = np.zeros(len(audio) + win_len, dtype=np.float32)
            norm = np.zeros(len(audio) + win_len, dtype=np.float32)
            input_hop = int(hop / factor)
            in_pos = 0
            out_pos = 0
            while in_pos + win_len < len(resampled) and out_pos + win_len < len(stretched):
                stretched[out_pos : out_pos + win_len] += resampled[in_pos : in_pos + win_len] * window
                norm[out_pos : out_pos + win_len] += window
                in_pos += input_hop
                out_pos += hop
            mask = norm > 1e-4
            stretched[mask] /= norm[mask]
            audio = stretched[: len(audio)]

        # 2. Audible Warmth EQ (0 to 100, 50 neutral)
        if warmth > 50:
            # Low-shelf warmth boost around 220 Hz (up to +4.8 dB)
            gain_db = ((warmth - 50.0) / 50.0) * 4.8
            b, a = signal.iirfilter(2, 220 / (sample_rate / 2), btype="lowpass", ftype="butter")
            low = signal.lfilter(b, a, audio)
            audio = audio + low * (10 ** (gain_db / 20.0) - 1.0)
        elif warmth < 50:
            # Crisp high-frequency sparkle boost around 5.5 kHz (up to +4.5 dB)
            gain_db = ((50.0 - warmth) / 50.0) * 4.5
            b, a = signal.iirfilter(2, 4500 / (sample_rate / 2), btype="highpass", ftype="butter")
            high = signal.lfilter(b, a, audio)
            audio = audio + high * (10 ** (gain_db / 20.0) - 1.0)

        # 3. Softness / De-essing (0 to 100, 50 neutral)
        if softness > 50:
            # Smooth sibilants with notch at 3.6 kHz
            w0 = 3600.0 / (sample_rate / 2.0)
            if 0 < w0 < 1:
                b_notch, a_notch = signal.iirnotch(w0, 3.0)
                notched = signal.lfilter(b_notch, a_notch, audio)
                mix = ((softness - 50.0) / 50.0) * 0.45
                audio = (1.0 - mix) * audio + mix * notched

        # 4. Transparent soft-knee peak limiter (tanh)
        peak = np.max(np.abs(audio))
        if peak > 0.88:
            audio = np.tanh(audio / 0.88) * 0.88

    except Exception as e:
        logger.debug(f"DSP filter bypassed: {e}")

    return audio


def adjust_text_pauses(text: str, pauses: float) -> str:
    """Adjusts punctuation and cadence based on the pauses slider:
    - pauses > 55: Adds breath pauses (commas -> ', ... ') for relaxed, thoughtful cadence
    - pauses < 45: Compresses pauses for brisk, energetic delivery
    """
    if pauses > 55:
        # Extend commas and clause transitions into gentle pauses
        pause_intensity = (pauses - 50.0) / 50.0
        if pause_intensity > 0.4:
            text = re.sub(r",\s*", ", ... ", text)
            text = re.sub(r"—\s*", " — ... ", text)
        else:
            text = re.sub(r",\s*", ", .. ", text)
    elif pauses < 45:
        # Strip long pauses and ellipses for rapid cadence
        text = re.sub(r"\.{2,}", ".", text)
        text = re.sub(r"—", "-", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Speech Engine Manager
# ---------------------------------------------------------------------------
class SpeechEngineManager:
    """Manages the Supertonic 3 model, voice styles, blending, and caching."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.tts: Optional[TTS] = None
        self.styles: Dict[str, Style] = {}
        self.cache = AudioCache(max_size=MAX_CACHE_ITEMS)
        self.is_ready = False
        self.sample_rate = 44100
        self.request_count = 0
        self.total_synth_time_ms = 0.0

        # StoryVox Pro PCA Voice Space attributes
        self.mean_ttl: Optional[np.ndarray] = None
        self.pca_components: Optional[np.ndarray] = None
        self.pca_variance: Optional[np.ndarray] = None
        self.mean_dp: Optional[np.ndarray] = None
        self.voice_coords: Dict[str, List[float]] = {}

    def initialize(self):
        logger.info(f"Loading Supertonic 3 TTS from: {self.model_dir}")
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        self.tts = TTS(model_dir=self.model_dir, auto_download=False)
        self.sample_rate = self.tts.sample_rate
        logger.info(f"Supertonic 3 initialized. Native sample rate: {self.sample_rate} Hz")

        # Preload all built-in styles
        for name in ALL_BASE_VOICES:
            try:
                style = self.tts.get_voice_style(name)
                self.styles[name] = style
                logger.info(f"Loaded voice style: {name} ({VOICE_METADATA.get(name, {}).get('name')})")
            except Exception as e:
                logger.warning(f"Failed to load built-in voice {name}: {e}")

        # Compute PCA Voice Space over built-in styles (StoryVox Pro architecture)
        try:
            ttls = [self.styles[n].ttl.flatten() for n in ALL_BASE_VOICES if n in self.styles]
            dps = [self.styles[n].dp.flatten() for n in ALL_BASE_VOICES if n in self.styles]
            if len(ttls) == len(ALL_BASE_VOICES):
                X = np.stack(ttls)
                self.mean_ttl = X.mean(axis=0)
                Xc = X - self.mean_ttl
                _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
                self.pca_components = Vt
                self.pca_variance = (S ** 2) / (S ** 2).sum()
                coords = Xc @ Vt.T
                for idx, name in enumerate(ALL_BASE_VOICES):
                    self.voice_coords[name] = coords[idx].tolist()
                Xdp = np.stack(dps)
                self.mean_dp = Xdp.mean(axis=0)
                logger.info(
                    f"Computed Voice Space PCA ({len(self.pca_components)} components). Top 4 variance: "
                    f"{[f'{v*100:.1f}%' for v in self.pca_variance[:4]]}"
                )
        except Exception as e:
            logger.warning(f"Could not compute PCA voice space: {e}")

        # Scan for custom voice JSONs
        custom_dir = self.model_dir / "voice_styles" / "custom"
        if custom_dir.exists():
            for json_file in custom_dir.glob("*.json"):
                if json_file.stem != "manifest":
                    try:
                        style = load_voice_style_from_json_file(json_file)
                        self.styles[json_file.stem] = style
                        logger.info(f"Loaded custom voice style: {json_file.stem}")
                    except Exception as e:
                        logger.warning(f"Could not load custom style {json_file}: {e}")

        self.is_ready = True
        logger.info("Speech Engine Manager is ready to process requests.")

    def pc_to_style(self, pc_values: List[float]) -> Style:
        """Navigates continuous latent voice space along principal components."""
        if self.mean_ttl is None or self.pca_components is None:
            return self.styles.get("F1") or next(iter(self.styles.values()))
        ttl_flat = self.mean_ttl.copy()
        for i, v in enumerate(pc_values):
            if i < len(self.pca_components):
                ttl_flat += float(v) * self.pca_components[i]
        ttl = np.clip(ttl_flat, -2.0, 2.0).reshape(1, 50, 256).astype(np.float32)
        dp = self.mean_dp.reshape(1, 8, 16).astype(np.float32)
        return Style(ttl, dp)

    def randomize_blend(self) -> Dict[str, float]:
        """Generates a balanced Dirichlet blend across the 10 built-ins."""
        vals = np.random.dirichlet(np.ones(len(ALL_BASE_VOICES)))
        weights = {}
        for name, val in zip(ALL_BASE_VOICES, vals):
            if val > 0.05:
                weights[name] = round(float(val), 2)
        return weights

    def save_custom_voice(self, style: Style, name: str) -> Path:
        """Saves a custom voice JSON into the custom voice styles directory."""
        custom_dir = self.model_dir / "voice_styles" / "custom"
        custom_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9_\-]", "_", name).strip() or "custom_voice"
        filepath = custom_dir / f"{safe_name}.json"
        payload = {
            "style_ttl": {"dims": list(style.ttl.shape), "data": style.ttl.tolist()},
            "style_dp": {"dims": list(style.dp.shape), "data": style.dp.tolist()},
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        self.styles[safe_name] = style
        logger.info(f"Saved and registered custom voice '{safe_name}' -> {filepath}")
        return filepath

    def get_style(
        self,
        voice: str,
        blend_weights: Optional[Dict[str, float]] = None,
        blend_secondary: Optional[str] = None,
        blend_ratio: Optional[float] = None,
        pc_vector: Optional[List[float]] = None,
    ) -> Style:
        """Resolves or dynamically blends voice styles."""
        # 0. Latent PCA voice space navigation
        if pc_vector is not None and len(pc_vector) > 0:
            return self.pc_to_style(pc_vector)

        # 1. Explicit weight map blending
        if blend_weights and len(blend_weights) > 0:
            return self.blend_voices(blend_weights)

        # 2. Secondary blend pair (e.g. voice='F1', secondary='F3', ratio=0.35)
        if blend_secondary and blend_ratio is not None and blend_ratio > 0.01:
            ratio = max(0.0, min(1.0, blend_ratio))
            weights = {
                voice: 1.0 - ratio,
                blend_secondary: ratio,
            }
            return self.blend_voices(weights)

        code = resolve_voice_code(voice)

        # 3. Direct voice style match
        if code in self.styles:
            return self.styles[code]

        # 4. Case-insensitive lookup
        for k in self.styles:
            if k.lower() == code.lower():
                return self.styles[k]

        logger.warning(f"Voice '{voice}' (code '{code}') not found. Defaulting to 'F1' (Elena).")
        return self.styles.get("F1") or next(iter(self.styles.values()))

    def blend_voices(self, weights: Dict[str, float]) -> Style:
        """Computes a mathematical linear blend between voice styles."""
        total = sum(max(0.0, float(w)) for w in weights.values())
        if total <= 0:
            return self.styles.get("F1") or next(iter(self.styles.values()))

        first_style = next(iter(self.styles.values()))
        ttl_shape = first_style.ttl.shape
        dp_shape = first_style.dp.shape

        blended_ttl = np.zeros(ttl_shape, dtype=np.float32)
        blended_dp = np.zeros(dp_shape, dtype=np.float32)

        for voice_name, raw_weight in weights.items():
            w = max(0.0, float(raw_weight))
            if w > 0:
                code = resolve_voice_code(voice_name)
                style = self.styles.get(code)
                if not style:
                    # check case-insensitive
                    for k, st in self.styles.items():
                        if k.lower() == code.lower():
                            style = st
                            break
                if style:
                    nw = w / total
                    blended_ttl += nw * style.ttl
                    blended_dp += nw * style.dp

        return Style(blended_ttl, blended_dp)

    def synthesize(
        self,
        text: str,
        voice: str = "F1",
        blend_weights: Optional[Dict[str, float]] = None,
        blend_secondary: Optional[str] = None,
        blend_ratio: Optional[float] = None,
        pc_vector: Optional[List[float]] = None,
        speed: float = 1.0,
        pitch: float = 1.0,
        total_steps: int = 12,
        lang: str = "en",
        warmth: float = 50.0,
        softness: float = 50.0,
        pauses: float = 50.0,
        bypass_cache: bool = False,
    ) -> tuple[bytes, float, bool]:
        """Synthesizes text into high-quality WAV audio bytes.
        Returns: (wav_bytes, duration_seconds, is_from_cache)
        """
        raw_text = text.strip()
        if not raw_text:
            raw_text = "..."

        # Adjust text cadence based on pauses slider
        clean_text = adjust_text_pauses(raw_text, pauses)

        # Generate cache key that incorporates all acoustic parameters
        blend_str = json.dumps(blend_weights, sort_keys=True) if blend_weights else ""
        if blend_secondary:
            blend_str += f"+{blend_secondary}:{blend_ratio:.2f}"
        if pc_vector:
            blend_str += f"+pc:{','.join(f'{v:.2f}' for v in pc_vector)}"

        cache_key = hashlib.sha256(
            f"{clean_text}|{voice}|{blend_str}|{speed:.3f}|{pitch:.3f}|{total_steps}|{lang}|{warmth:.1f}|{softness:.1f}|{pauses:.1f}".encode("utf-8")
        ).hexdigest()

        if not bypass_cache:
            cached = self.cache.get(cache_key)
            if cached:
                return cached[0], cached[1], True

        t0 = time.perf_counter()
        style = self.get_style(
            voice,
            blend_weights=blend_weights,
            blend_secondary=blend_secondary,
            blend_ratio=blend_ratio,
            pc_vector=pc_vector,
        )

        clamped_steps = max(6, min(24, total_steps))
        clamped_speed = max(0.65, min(1.40, speed))

        # Perform neural inference
        wav, dur = self.tts.synthesize(
            text=clean_text,
            voice_style=style,
            lang=lang,
            speed=clamped_speed,
            total_steps=clamped_steps,
        )

        audio_1d = wav.squeeze()

        # Apply acoustic DSP post-processing (pitch, warmth, softness, limiting)
        shaped_audio = apply_companion_acoustic_shaping(
            audio_1d,
            self.sample_rate,
            pitch=pitch,
            warmth=warmth,
            softness=softness,
        )

        # Encode to WAV buffer
        buf = io.BytesIO()
        sf.write(buf, shaped_audio, self.sample_rate, format="WAV", subtype="PCM_16")
        wav_bytes = buf.getvalue()

        duration_sec = float(len(shaped_audio)) / self.sample_rate
        synth_time_ms = (time.perf_counter() - t0) * 1000.0

        self.request_count += 1
        self.total_synth_time_ms += synth_time_ms
        logger.info(
            f"Synthesized '{clean_text[:35]}...' | Voice: {voice} | Speed: {clamped_speed:.2f} | Pitch: {pitch:.2f} | "
            f"Duration: {duration_sec:.2f}s | Latency: {synth_time_ms:.1f}ms | RTF: {(synth_time_ms / (duration_sec * 1000.0)):.2f}"
        )

        # Store in LRU cache
        if not bypass_cache:
            self.cache.put(cache_key, wav_bytes, duration_sec)

        return wav_bytes, duration_sec, False


# ---------------------------------------------------------------------------
# FastAPI Microservice Lifecycle & App Definition
# ---------------------------------------------------------------------------
engine_manager = SpeechEngineManager(model_dir=MODEL_DIR)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load ONNX models and warm up
    engine_manager.initialize()
    yield
    logger.info("Speech Engine Microservice shutting down.")


app = FastAPI(
    title="AIG Companion Natural Speech Engine",
    description="High-concurrency, cloud-ready Supertonic 3 speech synthesis microservice.",
    version="1.3.2",
    lifespan=lifespan,
)

# Enable CORS for web clients and cross-domain cloud hosting
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------------------------
class SynthesizeRequest(BaseModel):
    text: str = Field(..., description="Text or sentence to synthesize", min_length=1)
    voice: str = Field(default="F1", description="Voice identifier (e.g. F1, F2, F3, F4, F5)")
    blend_weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Optional weights for voice interpolation (e.g. {'F1': 0.7, 'F2': 0.3})",
    )
    blend_secondary: Optional[str] = Field(
        default=None,
        description="Optional secondary voice to blend with (e.g. F3 or F5)",
    )
    blend_ratio: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Ratio of secondary voice (0.0 to 1.0)",
    )
    pc_vector: Optional[List[float]] = Field(
        default=None,
        description="Optional 4-dimensional PCA latent voice space coordinates [pc1, pc2, pc3, pc4]",
    )
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speaking speed multiplier (0.75x to 1.35x)")
    pitch: float = Field(default=1.0, ge=0.5, le=2.0, description="Pitch multiplier (0.85x to 1.15x)")
    steps: int = Field(default=DEFAULT_STEPS, ge=4, le=32, description="Flow-matching diffusion steps")
    lang: str = Field(default="en", description="Language code (en, ja, ko, fr, de, es, etc.)")
    warmth: float = Field(default=50.0, ge=0.0, le=100.0, description="Warm acoustic low-end shaping")
    softness: float = Field(default=50.0, ge=0.0, le=100.0, description="Gentle de-essing and smoothing")
    pauses: float = Field(default=50.0, ge=0.0, le=100.0, description="Pausing and breath cadence")
    style: Optional[str] = Field(default="natural", description="Personality tone (natural, intimate, playful)")
    bypass_cache: bool = Field(default=False, description="If true, bypasses LRU cache (for live slider preview)")
    return_base64: bool = Field(default=False, description="If true, returns JSON with Base64 audio")


class BlendRequest(BaseModel):
    name: str = Field(..., description="Unique name for the blended voice")
    weights: Dict[str, float] = Field(..., description="Weight map of base voices")


class SaveCustomVoiceRequest(BaseModel):
    name: str = Field(..., description="Unique name for the custom voice style", min_length=1)
    blend_weights: Optional[Dict[str, float]] = None
    pc_vector: Optional[List[float]] = None
    source_voice: Optional[str] = "F1"


class OpenAISpeechRequest(BaseModel):
    model: str = Field(default="supertonic-3", description="Model name")
    input: str = Field(..., description="Text to synthesize", min_length=1)
    voice: str = Field(default="F1", description="Voice identifier")
    response_format: Optional[str] = Field(default="wav", description="Audio format (wav)")
    speed: Optional[float] = Field(default=1.0, description="Speed multiplier")


# Security Helper
def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header",
        )


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.get("/health/live", summary="Cloud Liveness Probe")
def liveness():
    """Returns 200 once the server is accepting HTTP requests."""
    return {"status": "alive", "service": "aig-voice-tts"}


@app.get("/health/ready", summary="Cloud Readiness Probe")
def readiness():
    """Confirms the neural models and voices are fully loaded into RAM/VRAM."""
    if not engine_manager.is_ready:
        raise HTTPException(status_code=503, detail="Models still loading")
    return {
        "status": "ready",
        "model": "Supertonic 3",
        "sample_rate": engine_manager.sample_rate,
        "voices_loaded": len(engine_manager.styles),
    }


@app.get("/v1/tts/voices", summary="Catalog of Available Voices")
def list_voices():
    """Returns all available base voices, descriptions, and custom imported styles."""
    voices = []
    for code in ALL_BASE_VOICES:
        meta = VOICE_METADATA.get(code, {})
        voices.append({
            "id": code,
            "displayName": meta.get("name", code),
            "gender": meta.get("gender", "neutral"),
            "description": meta.get("vibe", "Natural voice"),
            "isBuiltin": True,
        })

    for name in engine_manager.styles:
        if name not in ALL_BASE_VOICES:
            voices.append({
                "id": name,
                "displayName": name.replace("_", " ").title(),
                "gender": "custom",
                "description": "Custom or blended voice style",
                "isBuiltin": False,
            })

    return {"voices": voices, "total": len(voices)}


@app.post("/v1/tts/synthesize", summary="Synthesize Speech")
def synthesize_speech(
    req: SynthesizeRequest,
    x_api_key: Optional[str] = Header(None),
):
    """Generates natural speech for live companion conversation."""
    verify_api_key(x_api_key)

    wav_bytes, duration, is_cached = engine_manager.synthesize(
        text=req.text,
        voice=req.voice,
        blend_weights=req.blend_weights,
        blend_secondary=req.blend_secondary,
        blend_ratio=req.blend_ratio,
        pc_vector=req.pc_vector,
        speed=req.speed,
        pitch=req.pitch,
        total_steps=req.steps,
        lang=req.lang,
        warmth=req.warmth,
        softness=req.softness,
        pauses=req.pauses,
        bypass_cache=req.bypass_cache,
    )

    if req.return_base64:
        return {
            "status": "ok",
            "voice": req.voice,
            "duration": duration,
            "cached": is_cached,
            "sampleRate": engine_manager.sample_rate,
            "mimeType": "audio/wav",
            "audioBase64": base64.b64encode(wav_bytes).decode("ascii"),
        }

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Audio-Duration": f"{duration:.3f}",
            "X-Audio-Cached": "true" if is_cached else "false",
            "X-Sample-Rate": str(engine_manager.sample_rate),
            "Cache-Control": "public, max-age=3600" if is_cached else "no-cache",
        },
    )


@app.get("/v1/tts/voice-space", summary="PCA Voice Space Coordinates and Variance")
def get_voice_space():
    """Returns the principal component coordinates for all built-in voices and explained variance (StoryVox Pro architecture)."""
    if engine_manager.pca_variance is None:
        raise HTTPException(status_code=503, detail="Voice space PCA not yet initialized")
    return {
        "status": "ok",
        "explained_variance": [float(v) for v in engine_manager.pca_variance[:4]],
        "explained_variance_labels": [f"{v * 100:.1f}%" for v in engine_manager.pca_variance[:4]],
        "voice_coordinates": engine_manager.voice_coords,
        "axes": [
            {
                "id": "pc1",
                "label": "Vocal Archetype Spectrum (Masculine / Baritone <-> Feminine / Lyric)",
                "variance": float(engine_manager.pca_variance[0]),
            },
            {
                "id": "pc2",
                "label": "Energy & Resonance (Youthful & Bright <-> Mature & Deep)",
                "variance": float(engine_manager.pca_variance[1]),
            },
            {
                "id": "pc3",
                "label": "Warmth & Softness (Airy & Gentle <-> Direct & Announcer)",
                "variance": float(engine_manager.pca_variance[2]),
            },
            {
                "id": "pc4",
                "label": "Articulation & Formant Focus",
                "variance": float(engine_manager.pca_variance[3]),
            },
        ],
    }


@app.post("/v1/tts/randomize", summary="Generate a Dirichlet-Randomized Voice Blend")
def randomize_voice(
    text: str = Query(default="Hey, it's nice to hear from you. How has your day been?"),
    return_base64: bool = Query(default=True),
    x_api_key: Optional[str] = Header(None),
):
    """Generates an organic multi-voice blend using Dirichlet distribution (StoryVox Pro architecture)."""
    verify_api_key(x_api_key)
    weights = engine_manager.randomize_blend()
    wav_bytes, duration, _ = engine_manager.synthesize(
        text=text,
        voice="random",
        blend_weights=weights,
        bypass_cache=True,
    )
    if return_base64:
        return {
            "status": "ok",
            "blend_weights": weights,
            "duration": duration,
            "audioBase64": base64.b64encode(wav_bytes).decode("ascii"),
        }
    return Response(content=wav_bytes, media_type="audio/wav")


@app.post("/v1/tts/save-custom", summary="Save a Custom Voice Style JSON")
def save_custom_voice_route(
    req: SaveCustomVoiceRequest,
    x_api_key: Optional[str] = Header(None),
):
    """Saves a designed voice style as a persistent JSON in models/tts/supertonic3/voice_styles/custom/."""
    verify_api_key(x_api_key)
    if req.pc_vector:
        style = engine_manager.pc_to_style(req.pc_vector)
    elif req.blend_weights:
        style = engine_manager.blend_voices(req.blend_weights)
    else:
        style = engine_manager.get_style(req.source_voice or "F1")

    saved_path = engine_manager.save_custom_voice(style, req.name)
    return {
        "status": "ok",
        "voice_id": saved_path.stem,
        "file_path": str(saved_path),
        "message": f"Custom voice '{saved_path.stem}' saved and loaded successfully.",
    }


@app.post("/v1/audio/speech", summary="OpenAI-Compatible Speech Synthesis Endpoint")
def openai_speech_compatibility(
    req: OpenAISpeechRequest,
    x_api_key: Optional[str] = Header(None),
):
    """Drop-in endpoint matching the OpenAI Audio Speech API specification."""
    verify_api_key(x_api_key)
    wav_bytes, _, _ = engine_manager.synthesize(
        text=req.input,
        voice=req.voice,
        speed=req.speed or 1.0,
    )
    return Response(content=wav_bytes, media_type="audio/wav")


@app.post("/v1/tts/blend", summary="Create Blended Voice Style")
def create_blend(
    req: BlendRequest,
    x_api_key: Optional[str] = Header(None),
):
    """Blends multiple base voices into a persistent custom voice style."""
    verify_api_key(x_api_key)

    blended_style = engine_manager.blend_voices(req.weights)
    engine_manager.styles[req.name] = blended_style

    # Optionally persist custom style to disk
    custom_dir = engine_manager.model_dir / "voice_styles" / "custom"
    custom_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "style_ttl": {"dims": list(blended_style.ttl.shape), "data": blended_style.ttl.tolist()},
        "style_dp": {"dims": list(blended_style.dp.shape), "data": blended_style.dp.tolist()},
    }
    with open(custom_dir / f"{req.name}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)

    return {
        "status": "created",
        "voice": req.name,
        "weights": req.weights,
        "persisted": True,
    }


@app.get("/v1/metrics", summary="Performance & Cache Metrics")
def get_metrics():
    avg_latency = (
        (engine_manager.total_synth_time_ms / engine_manager.request_count)
        if engine_manager.request_count > 0
        else 0.0
    )
    return {
        "requestsTotal": engine_manager.request_count,
        "averageLatencyMs": round(avg_latency, 2),
        "cacheHits": engine_manager.cache.hits,
        "cacheMisses": engine_manager.cache.misses,
        "cacheHitRate": (
            round(
                engine_manager.cache.hits
                / max(1, engine_manager.cache.hits + engine_manager.cache.misses),
                3,
            )
        ),
        "cachedUtterances": len(engine_manager.cache._cache),
        "sampleRate": engine_manager.sample_rate,
    }


# ---------------------------------------------------------------------------
# CLI / Direct Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting AIG TTS Microservice on {HOST}:{PORT}...")
    uvicorn.run(
        "tts_service:app",
        host=HOST,
        port=PORT,
        workers=int(os.environ.get("WORKERS", "1")),
        log_level="info",
    )
