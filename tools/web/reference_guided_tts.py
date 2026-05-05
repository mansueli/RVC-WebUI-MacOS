from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import pyworld
import soundfile as sf

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from TTS.api import TTS as CoquiTTS
except Exception as exc:  # pragma: no cover
    CoquiTTS = None
    _XTTS_IMPORT_ERROR = str(exc)
else:  # pragma: no cover
    _XTTS_IMPORT_ERROR = ""


DEFAULT_TTS_SR = 24000
DEFAULT_FRAME_PERIOD_MS = 5.0
DEFAULT_TTS_BACKEND = "xtts"
DEFAULT_XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

_XTTS_MODEL_CACHE = None
_XTTS_MODEL_DEVICE = None

BACKEND_PRESETS = {
    "xtts": {"label": "XTTS v2 (voice cloning)"},
    "system": {"label": "macOS System TTS"},
}

XTTS_LANGUAGE_PRESETS = {
    "pt": {"label": "Portuguese [pt]"},
    "en": {"label": "English [en]"},
    "es": {"label": "Spanish [es]"},
}

STYLE_PRESETS = {
    "default": {
        "label": "Default",
        "preferred_voices": [],
        "f0_scale": 1.0,
        "rate_delta": 0,
    },
    "pt_br_baritone_soft": {
        "label": "PT-BR Baritone Soft",
        "preferred_voices": [
            "Reed (Portuguese (Brazil))",
            "Rocko (Portuguese (Brazil))",
            "Grandpa (Portuguese (Brazil))",
            "Eddy (Portuguese (Brazil))",
            "Luciana",
        ],
        "f0_scale": 0.84,
        "rate_delta": -15,
    },
    "pt_br_baritone_deep": {
        "label": "PT-BR Baritone Deep",
        "preferred_voices": [
            "Grandpa (Portuguese (Brazil))",
            "Rocko (Portuguese (Brazil))",
            "Reed (Portuguese (Brazil))",
            "Eddy (Portuguese (Brazil))",
            "Luciana",
        ],
        "f0_scale": 0.76,
        "rate_delta": -30,
    },
}


def _parse_voice_line(line: str) -> dict[str, str] | None:
    if "#" not in line:
        return None
    left = line.split("#", 1)[0].rstrip()
    if not left:
        return None
    parts = left.split()
    if len(parts) < 2:
        return None
    lang = parts[-1]
    name = left[: left.rfind(lang)].strip()
    if not name:
        return None
    label = f"{name} [{lang}]"
    return {"name": name, "language": lang, "label": label}


def list_tts_voice_catalog() -> list[dict[str, str]]:
    if shutil.which("say") is None:
        return [{"name": "Default", "language": "system", "label": "Default [system]"}]
    try:
        result = subprocess.run(
            ["say", "-v", "?"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return [
            {"name": "Samantha", "language": "en_US", "label": "Samantha [en_US]"},
            {"name": "Luciana", "language": "pt_BR", "label": "Luciana [pt_BR]"},
            {"name": "Joana", "language": "pt_PT", "label": "Joana [pt_PT]"},
        ]

    catalog: list[dict[str, str]] = []
    seen: set[str] = set()
    for line in result.stdout.splitlines():
        parsed = _parse_voice_line(line)
        if not parsed:
            continue
        key = parsed["label"]
        if key in seen:
            continue
        seen.add(key)
        catalog.append(parsed)
    return catalog or [{"name": "Default", "language": "system", "label": "Default [system]"}]


def list_tts_voices() -> list[str]:
    return [item["name"] for item in list_tts_voice_catalog()]


def list_tts_backends() -> list[dict[str, str]]:
    return [{"key": key, "label": cfg["label"]} for key, cfg in BACKEND_PRESETS.items()]


def list_xtts_languages() -> list[dict[str, str]]:
    return [{"key": key, "label": cfg["label"]} for key, cfg in XTTS_LANGUAGE_PRESETS.items()]


def list_tts_style_presets() -> list[dict[str, str]]:
    return [{"key": key, "label": cfg["label"]} for key, cfg in STYLE_PRESETS.items()]


def _resolve_style(style_key: str) -> dict[str, object]:
    return STYLE_PRESETS.get(style_key, STYLE_PRESETS["default"])


def _resolve_backend(backend_key: str) -> str:
    return backend_key if backend_key in BACKEND_PRESETS else DEFAULT_TTS_BACKEND


def _resolve_xtts_language(language_key: str) -> str:
    return language_key if language_key in XTTS_LANGUAGE_PRESETS else "pt"


def _choose_xtts_device() -> str:
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_xtts_model():
    global _XTTS_MODEL_CACHE, _XTTS_MODEL_DEVICE
    if CoquiTTS is None:
        detail = f" Import error: {_XTTS_IMPORT_ERROR}" if _XTTS_IMPORT_ERROR else ""
        raise RuntimeError(
            "XTTS is unavailable because the Coqui TTS package is not installed." + detail
        )
    device = _choose_xtts_device()
    if _XTTS_MODEL_CACHE is not None and _XTTS_MODEL_DEVICE == device:
        return _XTTS_MODEL_CACHE

    model = CoquiTTS(model_name=DEFAULT_XTTS_MODEL, progress_bar=False, gpu=False)
    if hasattr(model, "to"):
        try:
            model = model.to(device)
        except Exception:
            device = "cpu"
    _XTTS_MODEL_CACHE = model
    _XTTS_MODEL_DEVICE = device
    return model


def _resolve_preferred_voice(selected_voice_name: str, style_key: str) -> str:
    catalog_names = set(list_tts_voices())
    if selected_voice_name in catalog_names:
        if style_key == "default":
            return selected_voice_name
    style = _resolve_style(style_key)
    for candidate in style["preferred_voices"]:
        if candidate in catalog_names:
            return candidate
    return selected_voice_name


def _load_audio(path: str | Path, target_sr: int) -> np.ndarray:
    wav, _ = librosa.load(str(path), sr=target_sr, mono=True)
    wav = np.asarray(wav, dtype=np.float32)
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 1.0:
        wav = wav / peak
    return wav


def _extract_f0(wav: np.ndarray, sample_rate: int, frame_period_ms: float = DEFAULT_FRAME_PERIOD_MS) -> np.ndarray:
    if wav.size == 0:
        return np.zeros((0,), dtype=np.float32)
    wav64 = wav.astype(np.float64)
    f0, t = pyworld.harvest(
        wav64,
        sample_rate,
        frame_period=frame_period_ms,
        f0_floor=50.0,
        f0_ceil=1100.0,
    )
    f0 = pyworld.stonemask(wav64, f0, t, sample_rate)
    return np.asarray(f0, dtype=np.float32)


def _interp_curve(values: np.ndarray, target_len: int) -> np.ndarray:
    if target_len <= 0:
        return np.zeros((0,), dtype=np.float32)
    if values.size == 0:
        return np.zeros((target_len,), dtype=np.float32)
    if values.size == 1:
        return np.full((target_len,), float(values[0]), dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, num=values.size)
    x_new = np.linspace(0.0, 1.0, num=target_len)
    return np.interp(x_new, x_old, values).astype(np.float32)


def _fill_unvoiced(f0: np.ndarray) -> np.ndarray:
    if f0.size == 0:
        return f0.astype(np.float32)
    out = f0.astype(np.float32).copy()
    voiced = np.where(out > 0)[0]
    if voiced.size == 0:
        return np.zeros_like(out, dtype=np.float32)
    if voiced.size == 1:
        out[:] = out[voiced[0]]
        return out
    unvoiced = np.where(out <= 0)[0]
    out[unvoiced] = np.interp(unvoiced, voiced, out[voiced])
    return out


def analyze_reference_audio(
    audio_path: str | Path,
    reference_text: str = "",
    target_sr: int = DEFAULT_TTS_SR,
) -> tuple[str, str]:
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Reference audio not found: {path}")

    wav = _load_audio(path, target_sr)
    f0 = _extract_f0(wav, target_sr)
    voiced = f0[f0 > 0]
    duration_sec = float(len(wav)) / float(target_sr) if wav.size else 0.0
    words = len(reference_text.split()) if reference_text.strip() else 0
    payload = {
        "source": str(path),
        "sample_rate": target_sr,
        "duration_seconds": round(duration_sec, 4),
        "sample_count": int(len(wav)),
        "f0_frame_count": int(f0.size),
        "voiced_frame_count": int(voiced.size),
        "median_f0_hz": round(float(np.median(voiced)), 3) if voiced.size else 0.0,
        "mean_f0_hz": round(float(np.mean(voiced)), 3) if voiced.size else 0.0,
        "min_f0_hz": round(float(np.min(voiced)), 3) if voiced.size else 0.0,
        "max_f0_hz": round(float(np.max(voiced)), 3) if voiced.size else 0.0,
        "reference_text": reference_text,
        "word_count": words,
        "words_per_second": round(words / duration_sec, 3) if words > 0 and duration_sec > 0 else 0.0,
    }
    info_lines = [
        f"Source: {path}",
        f"Duration: {payload['duration_seconds']:.3f}s",
        f"Samples @ {target_sr} Hz: {payload['sample_count']}",
        f"F0 frames: {payload['f0_frame_count']} (voiced: {payload['voiced_frame_count']})",
        f"Median F0: {payload['median_f0_hz']:.2f} Hz",
        f"Range F0: {payload['min_f0_hz']:.2f} Hz -> {payload['max_f0_hz']:.2f} Hz",
    ]
    if words > 0:
        info_lines.append(f"Reference text words: {words} ({payload['words_per_second']:.2f} words/s)")

    out_dir = Path(tempfile.mkdtemp(prefix="reference_tts_analysis_"))
    analysis_path = out_dir / "analysis.json"
    analysis_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return "\n".join(info_lines), str(analysis_path)


def _synthesize_macos_tts(text: str, voice_name: str, rate: int, out_dir: Path) -> Path:
    if shutil.which("say") is None:
        raise RuntimeError("macOS 'say' command is unavailable on this machine")
    text_path = out_dir / "input.txt"
    aiff_path = out_dir / "tts.aiff"
    text_path.write_text(text, encoding="utf-8")
    subprocess.run(
        ["say", "-v", voice_name, "-r", str(int(rate)), "-f", str(text_path), "-o", str(aiff_path)],
        check=True,
    )
    return aiff_path


def _synthesize_xtts(text: str, speaker_wav: Path, language_key: str, out_dir: Path) -> Path:
    model = _load_xtts_model()
    out_path = out_dir / "tts_xtts.wav"
    kwargs = {
        "text": text,
        "speaker_wav": str(speaker_wav),
        "language": _resolve_xtts_language(language_key),
        "file_path": str(out_path),
    }
    try:
        model.tts_to_file(**kwargs)
    except TypeError:
        kwargs["speaker_wav"] = [str(speaker_wav)]
        model.tts_to_file(**kwargs)
    return out_path


def _match_duration(wav: np.ndarray, target_samples: int) -> np.ndarray:
    if target_samples <= 0:
        return wav.astype(np.float32, copy=False)
    if wav.size == 0:
        return np.zeros((target_samples,), dtype=np.float32)
    if wav.size == target_samples:
        return wav.astype(np.float32, copy=False)

    # Prefer pitch-preserving stretch, but some local numpy/numba/librosa
    # combinations break inside phase_vocoder. Fall back to interpolation so
    # XTTS synthesis still completes.
    rate = max(0.25, min(4.0, float(wav.size) / float(target_samples)))
    try:
        stretched = librosa.effects.time_stretch(
            wav.astype(np.float32, copy=False),
            rate=rate,
        )
    except Exception:
        x_old = np.linspace(0.0, 1.0, num=wav.size, endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=target_samples, endpoint=False)
        stretched = np.interp(x_new, x_old, wav).astype(np.float32, copy=False)

    if stretched.size < target_samples:
        stretched = np.pad(stretched, (0, target_samples - stretched.size))
    return stretched[:target_samples].astype(np.float32, copy=False)


def _pitch_transfer(tts_wav: np.ndarray, ref_f0: np.ndarray, sample_rate: int, f0_scale: float = 1.0) -> np.ndarray:
    if tts_wav.size == 0:
        return tts_wav.astype(np.float32, copy=False)
    wav64 = tts_wav.astype(np.float64)
    f0_tts, sp, ap = pyworld.wav2world(
        wav64,
        sample_rate,
        frame_period=DEFAULT_FRAME_PERIOD_MS,
    )
    if f0_tts.size == 0:
        return tts_wav.astype(np.float32, copy=False)
    voiced_mask = f0_tts > 0
    if not np.any(voiced_mask):
        return tts_wav.astype(np.float32, copy=False)

    if ref_f0.size == 0 or np.count_nonzero(ref_f0 > 0) == 0:
        contour = f0_tts.astype(np.float32)
    else:
        ref_filled = _fill_unvoiced(ref_f0)
        contour = _interp_curve(ref_filled, f0_tts.size)
    contour = contour * float(f0_scale)

    new_f0 = np.zeros_like(f0_tts, dtype=np.float64)
    new_f0[voiced_mask] = np.clip(contour[voiced_mask], 50.0, 1100.0).astype(np.float64)
    rendered = pyworld.synthesize(new_f0, sp, ap, sample_rate, DEFAULT_FRAME_PERIOD_MS)
    return rendered.astype(np.float32, copy=False)


def generate_reference_guided_tts(
    reference_audio: str | Path,
    reference_text: str,
    target_text: str,
    voice_name: str,
    speaking_rate: int,
    style_key: str = "default",
    backend_key: str = DEFAULT_TTS_BACKEND,
    xtts_language: str = "pt",
    target_sr: int = DEFAULT_TTS_SR,
) -> tuple[str, str]:
    ref_path = Path(reference_audio)
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference audio not found: {ref_path}")

    synth_text = (target_text or reference_text or "").strip()
    if not synth_text:
        raise ValueError("Target text is required for synthesis")

    ref_wav = _load_audio(ref_path, target_sr)
    ref_f0 = _extract_f0(ref_wav, target_sr)
    out_dir = Path(tempfile.mkdtemp(prefix="reference_tts_render_"))

    style = _resolve_style(style_key)
    backend = _resolve_backend(backend_key)
    resolved_voice = _resolve_preferred_voice(voice_name, style_key)
    resolved_rate = max(100, int(speaking_rate) + int(style["rate_delta"]))

    if backend == "xtts":
        raw_tts_path = _synthesize_xtts(
            synth_text,
            ref_path,
            xtts_language,
            out_dir,
        )
    else:
        raw_tts_path = _synthesize_macos_tts(
            synth_text,
            resolved_voice,
            resolved_rate,
            out_dir,
        )
    tts_wav = _load_audio(raw_tts_path, target_sr)
    matched = _match_duration(tts_wav, len(ref_wav))
    pitched = _pitch_transfer(matched, ref_f0, target_sr, f0_scale=float(style["f0_scale"]))

    if pitched.size < len(ref_wav):
        pitched = np.pad(pitched, (0, len(ref_wav) - pitched.size))
    pitched = pitched[: len(ref_wav)]
    peak = float(np.max(np.abs(pitched))) if pitched.size else 0.0
    if peak > 0:
        pitched = pitched / max(1.0, peak)

    output_path = out_dir / "reference_guided_tts.wav"
    sf.write(str(output_path), pitched, target_sr)

    duration_sec = float(len(ref_wav)) / float(target_sr) if ref_wav.size else 0.0
    info_lines = [
        f"Backend: {BACKEND_PRESETS[backend]['label']}",
        f"Style preset: {style['label']}",
        f"Voice: {resolved_voice}" if backend == "system" else f"XTTS language: {_resolve_xtts_language(xtts_language)}",
        f"Speaking rate: {resolved_rate}" if backend == "system" else "Speaker clone source: reference audio",
        f"F0 scale: {float(style['f0_scale']):.2f}",
        f"Target sample rate: {target_sr}",
        f"Reference duration matched: {duration_sec:.3f}s",
        f"Synth text length: {len(synth_text)} chars",
        f"Output: {output_path}",
        "Next step: optionally run the generated wav through an RVC model in the Model Inference tab.",
    ]
    return "\n".join(info_lines), str(output_path)