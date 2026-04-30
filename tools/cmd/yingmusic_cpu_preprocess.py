#!/usr/bin/env python3
"""CPU fallback preprocessor inspired by YingMusic vocal isolation.

This is an approximation for environments without CUDA separators.
It applies HPSS and spectral gating to emphasize vocal components.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import soundfile as sf
except Exception:  # pragma: no cover
    sf = None

try:
    import librosa
except Exception:  # pragma: no cover
    librosa = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


_WINDOW_CACHE: dict[tuple[int, str], torch.Tensor] = {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CPU Ying-style vocal preprocessing")
    p.add_argument("--source-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--target-sr", type=int, default=48000)
    p.add_argument("--top-db", type=float, default=35.0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return p.parse_args()


def choose_device(name: str):
    if torch is None:
        return None
    if name == "cuda":
        return torch.device("cuda")
    if name == "mps":
        return torch.device("mps")
    if name == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _cache_key(device: torch.device) -> str:
    index = getattr(device, "index", None)
    return "%s:%s" % (device.type, index if index is not None else -1)


def get_hann_window(n_fft: int, device: torch.device) -> torch.Tensor:
    key = (n_fft, _cache_key(device))
    window = _WINDOW_CACHE.get(key)
    if window is None:
        window = torch.hann_window(n_fft, device=device)
        _WINDOW_CACHE[key] = window
    return window


def sync_device(device) -> None:
    if torch is None or device is None:
        return
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def process_file_torch(
    src: Path,
    dst: Path,
    target_sr: int,
    top_db: float,
    device: torch.device,
) -> dict[str, float]:
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    wav, sr = sf.read(str(src), always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    if sr != target_sr:
        if librosa is None:
            raise RuntimeError("librosa is required for resampling when sample rate differs")
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    timings["load"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    x = torch.from_numpy(wav).to(device=device, dtype=torch.float32)
    n_fft = 2048
    hop = 512
    window = get_hann_window(n_fft, device)
    sync_device(device)
    timings["transfer"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    spec = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        return_complex=True,
    )
    mag = spec.abs()

    # Approximate sustained vocal emphasis using temporal smoothing.
    harmonic_mag = torch.nn.functional.avg_pool1d(
        mag.transpose(0, 1), kernel_size=9, stride=1, padding=4
    ).transpose(0, 1)

    ref = harmonic_mag.amax(dim=-1, keepdim=True).clamp_min(1e-6)
    db = 20.0 * torch.log10(harmonic_mag / ref)
    mask = torch.sigmoid((db + top_db) / 6.0)
    filtered = spec * mask
    sync_device(device)
    timings["spectral"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = torch.istft(
        filtered,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        length=x.shape[0],
    )
    peak = out.abs().amax().clamp_min(1e-7)
    out = 0.95 * out / peak
    sync_device(device)
    timings["inverse"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), out.detach().cpu().numpy().astype(np.float32), target_sr)
    timings["save"] = time.perf_counter() - t0
    timings["total"] = sum(timings.values())
    return timings


def process_file_librosa(src: Path, dst: Path, target_sr: int, top_db: float) -> dict[str, float]:
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    y, sr = librosa.load(str(src), sr=target_sr, mono=True)
    timings["load"] = time.perf_counter() - t0

    # Harmonic-percussive separation keeps sustained components typical of vocals.
    t0 = time.perf_counter()
    harmonic, _ = librosa.effects.hpss(y)

    # Light pre-emphasis and denoising gate.
    harmonic = np.append(harmonic[0], harmonic[1:] - 0.97 * harmonic[:-1])
    intervals = librosa.effects.split(harmonic, top_db=top_db)
    masked = np.zeros_like(harmonic)
    for start, end in intervals:
        masked[start:end] = harmonic[start:end]
    timings["spectral"] = time.perf_counter() - t0

    # Loudness normalization.
    peak = np.max(np.abs(masked)) + 1e-7
    out = 0.95 * masked / peak

    t0 = time.perf_counter()
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), out.astype(np.float32), target_sr)
    timings["save"] = time.perf_counter() - t0
    timings["total"] = sum(timings.values())
    return timings


def main() -> int:
    args = parse_args()

    if np is None or sf is None:
        print("[error] Missing dependencies for CPU Ying preprocessing.")
        print("[hint] Install: numpy, soundfile")
        return 2

    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not source_dir.exists():
        print("[error] source-dir not found:", source_dir)
        return 1

    audio_exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
    files = [p for p in sorted(source_dir.iterdir()) if p.suffix.lower() in audio_exts]
    if not files:
        print("[error] no audio files found in source-dir:", source_dir)
        return 1

    device = choose_device(args.device)
    backend = "torch" if device is not None else "librosa"
    print("[timing] preprocess backend=%s device=%s files=%d" % (backend, device, len(files)))
    total_timings = {
        "load": 0.0,
        "transfer": 0.0,
        "spectral": 0.0,
        "inverse": 0.0,
        "save": 0.0,
        "total": 0.0,
    }

    for src in files:
        dst = output_dir / (src.stem + ".wav")
        if device is not None:
            timings = process_file_torch(src, dst, args.target_sr, args.top_db, device)
        elif librosa is not None:
            timings = process_file_librosa(src, dst, args.target_sr, args.top_db)
        else:
            print("[error] Need either torch or librosa for preprocessing.")
            return 2
        for key, value in timings.items():
            total_timings[key] = total_timings.get(key, 0.0) + value
        detail = ", ".join(
            "%s=%.3fs" % (key, timings[key])
            for key in ["load", "transfer", "spectral", "inverse", "save", "total"]
            if key in timings
        )
        print("[timing] %s: %s" % (src.name, detail))
        print("[done]", dst)

    total_detail = ", ".join(
        "%s=%.3fs" % (key, total_timings[key])
        for key in ["load", "transfer", "spectral", "inverse", "save", "total"]
        if total_timings.get(key, 0.0) > 0.0
    )
    print("[timing] total: %s" % total_detail)
    print("[done] CPU Ying-style preprocessing completed:", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
HQ-SVC