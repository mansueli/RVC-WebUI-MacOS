#!/usr/bin/env python3
"""CPU fallback preprocessor inspired by YingMusic vocal isolation.

This is an approximation for environments without CUDA separators.
It applies HPSS and spectral gating to emphasize vocal components.
"""

from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CPU Ying-style vocal preprocessing")
    p.add_argument("--source-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--target-sr", type=int, default=44100)
    p.add_argument("--top-db", type=float, default=35.0)
    return p.parse_args()


def process_file(src: Path, dst: Path, target_sr: int, top_db: float) -> None:
    y, sr = librosa.load(str(src), sr=target_sr, mono=True)

    # Harmonic-percussive separation keeps sustained components typical of vocals.
    harmonic, _ = librosa.effects.hpss(y)

    # Light pre-emphasis and denoising gate.
    harmonic = np.append(harmonic[0], harmonic[1:] - 0.97 * harmonic[:-1])
    intervals = librosa.effects.split(harmonic, top_db=top_db)
    masked = np.zeros_like(harmonic)
    for start, end in intervals:
        masked[start:end] = harmonic[start:end]

    # Loudness normalization.
    peak = np.max(np.abs(masked)) + 1e-7
    out = 0.95 * masked / peak

    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), out.astype(np.float32), target_sr)


def main() -> int:
    args = parse_args()

    if np is None or sf is None or librosa is None:
        print("[error] Missing dependencies for CPU Ying preprocessing.")
        print("[hint] Install: numpy, soundfile, librosa")
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

    for src in files:
        dst = output_dir / (src.stem + ".wav")
        process_file(src, dst, args.target_sr, args.top_db)
        print("[done]", dst)

    print("[done] CPU Ying-style preprocessing completed:", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
