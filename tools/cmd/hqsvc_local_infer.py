#!/usr/bin/env python3
"""Experimental in-repo HQ-SVC-style native inference (CPU/GPU)."""

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
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


try:
    import librosa
except Exception:  # pragma: no cover
    librosa = None


if nn is not None:
    class TinyHQSVC(nn.Module):
        def __init__(self, channels: int = 64):
            super().__init__()
            self.in_proj = nn.Conv1d(1, channels, kernel_size=7, padding=3)
            self.block1 = nn.Sequential(
                nn.Conv1d(channels, channels, 5, padding=2),
                nn.GELU(),
                nn.Conv1d(channels, channels, 3, padding=1),
                nn.GELU(),
            )
            self.block2 = nn.Sequential(
                nn.Conv1d(channels, channels, 5, padding=2),
                nn.GELU(),
                nn.Conv1d(channels, channels, 3, padding=1),
                nn.GELU(),
            )
            self.out_proj = nn.Conv1d(channels, 1, kernel_size=7, padding=3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.in_proj(x)
            h = h + self.block1(h)
            h = h + self.block2(h)
            y = torch.tanh(self.out_proj(h))
            return y
else:
    class TinyHQSVC:  # pragma: no cover
        pass


def choose_device(name: str) -> torch.device:
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experimental HQ-SVC-style local inference")
    p.add_argument("--source", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return p.parse_args()


def load_audio(path: Path, target_sr: int) -> np.ndarray:
    try:
        wav, sr = sf.read(str(path), always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = wav.astype(np.float32)
        if sr != target_sr:
            if librosa is None:
                raise RuntimeError(
                    "librosa is required for resampling when sample rate differs"
                )
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    except Exception as exc:
        if librosa is None:
            raise RuntimeError(
                "Failed to decode input audio with soundfile and librosa is unavailable"
            ) from exc
        wav, sr = librosa.load(str(path), sr=target_sr, mono=True)
        wav = wav.astype(np.float32)
    m = np.max(np.abs(wav))
    if m > 1.0:
        wav = wav / m
    return wav


def main() -> int:
    args = parse_args()
    if np is None or sf is None or torch is None or nn is None:
        print("[error] Missing dependencies for local experimental inference.")
        print("[hint] Install: numpy, torch, soundfile, librosa")
        return 2
    ckpt_path = Path(args.checkpoint).resolve()
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.exists():
        print("[error] checkpoint not found:", ckpt_path)
        return 1

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    sample_rate = int(ckpt.get("sample_rate", 44100))

    model = TinyHQSVC()
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)

    device = choose_device(args.device)
    model.to(device).eval()

    src = load_audio(Path(args.source).resolve(), sample_rate)

    # Chunked inference to control memory use on CPU.
    chunk = sample_rate * 8
    hop = sample_rate * 6
    out = np.zeros_like(src)
    wsum = np.zeros_like(src)

    pos = 0
    while pos < len(src):
        seg = src[pos : pos + chunk]
        if len(seg) < chunk:
            seg = np.pad(seg, (0, chunk - len(seg)))
        x = torch.from_numpy(seg).to(device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            y = model(x).squeeze(0).squeeze(0).detach().float().cpu().numpy()
        y = y[: min(chunk, len(src) - pos)]
        end = pos + len(y)
        out[pos:end] += y
        wsum[pos:end] += 1.0
        pos += hop

    wsum = np.maximum(wsum, 1e-6)
    out = out / wsum
    out = np.clip(out, -1.0, 1.0)

    sf.write(str(out_path), out, sample_rate)
    print("[done] local experimental inference output:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
