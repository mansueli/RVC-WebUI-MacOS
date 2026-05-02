#!/usr/bin/env python3
"""Paper-aligned V3 preprocessing wrapper.

This script prefers a neural separator backend when available and falls back to
the existing spectral approximation so the WebUI can default to a paper-mode
path without losing runnability on macOS.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import warnings
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
except Exception:  # pragma: no cover
    torch = None

try:
    import torchaudio
except Exception:  # pragma: no cover
    torchaudio = None


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-aligned YingMusic preprocessing")
    parser.add_argument("--source-dir", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--target-sr", type=int, default=48000)
    parser.add_argument(
        "--backend",
        default="full_paper_mode",
        choices=[
            "full_paper_mode",
            "paper_auto",
            "torchaudio_hdemucs",
            "cpu_fallback",
            "external_yingmusic",
        ],
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--harmony-alpha", type=float, default=0.0)
    parser.add_argument("--pitch-shift", type=float, default=0.0)
    return parser.parse_args()


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


def list_audio_files(source_dir: Path) -> list[Path]:
    audio_exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
    return [p for p in sorted(source_dir.iterdir()) if p.suffix.lower() in audio_exts]


def run_cpu_fallback(source_dir: Path, output_dir: Path, target_sr: int, device_name: str) -> int:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "cmd" / "yingmusic_cpu_preprocess.py"),
        "--source-dir",
        str(source_dir),
        "--output-dir",
        str(output_dir),
        "--target-sr",
        str(target_sr),
        "--device",
        device_name,
    ]
    print("[info] Falling back to spectral preprocess backend.")
    return subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode


def run_external_yingmusic(source_dir: Path, output_dir: Path) -> int:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "cmd" / "yingmusic_experiment.py"),
        "--source-dir",
        str(source_dir),
        "--output-dir",
        str(output_dir),
    ]
    print("[info] Delegating to external YingMusic wrapper.")
    return subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode


def _load_hdemucs_bundle():
    if torchaudio is None:
        return None
    pipelines = getattr(torchaudio, "pipelines", None)
    if pipelines is None:
        return None
    for name in ("HDEMUCS_HIGH_MUSDB_PLUS", "HDEMUCS_HIGH_MUSDB"):
        bundle = getattr(pipelines, name, None)
        if bundle is not None:
            return bundle
    return None


def _load_waveform_for_separator(path: Path):
    """Load audio without requiring torchcodec-backed torchaudio loaders.

    On some torchaudio builds, torchaudio.load hard-depends on torchcodec.
    Prefer soundfile when available, then fall back to torchaudio.
    """
    if torch is None:
        raise RuntimeError("torch is required for separator audio loading")

    if sf is not None and np is not None:
        wav, sr = sf.read(str(path), always_2d=True)
        wav = np.asarray(wav, dtype=np.float32)
        waveform = torch.from_numpy(wav.T.copy())
        return waveform, int(sr)

    if torchaudio is None:
        raise RuntimeError("neither soundfile nor torchaudio audio loading is available")
    return torchaudio.load(str(path))


def run_torchaudio_separator(
    source_dir: Path,
    output_dir: Path,
    target_sr: int,
    device,
    harmony_alpha: float,
    pitch_shift: float,
) -> int:
    if torch is None or torchaudio is None or sf is None or np is None:
        print("[warn] torchaudio neural separator dependencies are unavailable.")
        return 2

    bundle = _load_hdemucs_bundle()
    if bundle is None:
        print("[warn] No torchaudio HDemucs bundle is available in this environment.")
        return 2

    # PyTorch/torchaudio version combos on macOS can emit noisy internal
    # resize warnings from STFT/ISTFT scratch buffers; these are non-fatal.
    warnings.filterwarnings(
        "ignore",
        message=r"An output with one or more elements was resized since it had shape \[\]",
        category=UserWarning,
    )

    files = list_audio_files(source_dir)
    if not files:
        print("[error] no audio files found in source-dir:", source_dir)
        return 1

    model = bundle.get_model().to(device).eval()
    sources = list(getattr(bundle, "sources", []))
    source_to_idx = {name: idx for idx, name in enumerate(sources)}
    vocals_idx = source_to_idx.get("vocals", 0)
    other_idx = source_to_idx.get("other", None)

    for src in files:
        try:
            waveform, sr = _load_waveform_for_separator(src)
        except Exception as exc:
            print("[warn] failed to load %s for neural separator: %s" % (src, exc))
            return 2
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        if sr != bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, bundle.sample_rate)
        waveform = waveform.to(device)

        with torch.no_grad():
            separated = model(waveform.unsqueeze(0))[0]
        lead = separated[vocals_idx]
        if other_idx is not None and harmony_alpha > 0.0:
            lead = lead + (harmony_alpha * separated[other_idx])
        if pitch_shift != 0.0:
            lead = torchaudio.functional.pitch_shift(
                lead,
                sample_rate=bundle.sample_rate,
                n_steps=float(pitch_shift),
            )
        if bundle.sample_rate != target_sr:
            lead = torchaudio.functional.resample(lead, bundle.sample_rate, target_sr)

        lead = lead.mean(dim=0, keepdim=True)
        peak = lead.abs().amax().clamp_min(1e-7)
        lead = 0.95 * lead / peak

        out_path = output_dir / (src.stem + ".wav")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), lead.squeeze(0).detach().cpu().numpy().astype(np.float32), target_sr)
        print("[done]", out_path)

    print("[done] Paper-aligned neural preprocessing completed:", output_dir)
    return 0


def main() -> int:
    args = parse_args()
    device = choose_device(args.device)

    if args.setup_only:
        bundle = _load_hdemucs_bundle()
        print("[info] backend request:", args.backend)
        print("[info] torch device:", device)
        print("[info] torchaudio_hdemucs available:", bundle is not None)
        print("[info] cpu fallback available:", np is not None and sf is not None)
        print("[done] Paper preprocess environment check complete.")
        return 0

    if not args.source_dir or not args.output_dir:
        print("[error] --source-dir and --output-dir are required unless --setup-only is used.")
        return 2

    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not source_dir.exists():
        print("[error] source-dir not found:", source_dir)
        return 1

    backend = args.backend
    if backend == "full_paper_mode":
        backend = "paper_auto"
    if backend == "paper_auto":
        backend = "torchaudio_hdemucs" if _load_hdemucs_bundle() is not None else "cpu_fallback"

    print("[info] preprocess backend:", backend)
    print("[info] device:", device)

    if backend == "external_yingmusic":
        return run_external_yingmusic(source_dir, output_dir)
    if backend == "cpu_fallback":
        return run_cpu_fallback(source_dir, output_dir, args.target_sr, str(device or "cpu"))

    rc = run_torchaudio_separator(
        source_dir,
        output_dir,
        args.target_sr,
        device,
        args.harmony_alpha,
        args.pitch_shift,
    )
    if rc == 0:
        return 0
    return run_cpu_fallback(source_dir, output_dir, args.target_sr, str(device or "cpu"))


if __name__ == "__main__":
    raise SystemExit(main())
