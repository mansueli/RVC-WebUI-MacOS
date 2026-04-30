#!/usr/bin/env python3
"""Experimental in-repo HQ-SVC-style training (CPU/GPU).

This is a pragmatic, paper-inspired trainer to provide a runnable end-to-end
V3 path in this repository when upstream HQ-SVC training integration is not
available. It uses a lightweight waveform model and composite losses named to
mirror HQ-SVC concepts:
- L_ddsp: multi-resolution STFT magnitude loss
- L_diff: denoising consistency loss
- L_spk: timbre statistics loss over mel features
- L_f0: pitch contour loss (autocorrelation estimator)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None


try:
    import librosa
except Exception:  # pragma: no cover
    librosa = None

try:
    import soundfile as sf
except Exception:  # pragma: no cover
    sf = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experimental HQ-SVC-style local trainer")
    p.add_argument("--exp-dir", required=True, help="Experiment directory under logs/")
    p.add_argument("--dataset-dir", default="", help="Directory of target singer wav files")
    p.add_argument("--sample-rate", type=int, default=44100)
    p.add_argument("--encoder-sr", type=int, default=16000)
    p.add_argument("--hop-feature", type=int, default=512)
    p.add_argument("--hop-infer", type=int, default=256)
    p.add_argument("--mel-bins", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--segment-seconds", type=float, default=1.5)
    p.add_argument("--learning-rate", type=float, default=1.5e-4)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--author", default="")
    p.add_argument("--output-checkpoint", default="")
    return p.parse_args()


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


def load_wavs(dataset_dir: Path, sample_rate: int) -> list[np.ndarray]:
    wavs: list[np.ndarray] = []
    for wav_path in sorted(dataset_dir.rglob("*.wav")):
        wav, sr = sf.read(str(wav_path), always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = wav.astype(np.float32)
        if sr != sample_rate:
            if librosa is None:
                raise RuntimeError(
                    "librosa is required for resampling when dataset sample rate differs"
                )
            wav = librosa.resample(wav, orig_sr=sr, target_sr=sample_rate)
        if wav.size > sample_rate // 2:
            wavs.append(wav)
    return wavs


def random_crop(w: np.ndarray, n: int) -> np.ndarray:
    if len(w) <= n:
        out = np.zeros(n, dtype=np.float32)
        out[: len(w)] = w
        return out
    start = random.randint(0, len(w) - n)
    return w[start : start + n]


def stft_mag(x: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    win = torch.hann_window(n_fft, device=x.device)
    s = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=win,
        return_complex=True,
    )
    return torch.abs(s)


def f0_autocorr(x: torch.Tensor, sr: int, fmin: float = 60.0, fmax: float = 900.0) -> torch.Tensor:
    # x: [B, T]
    b, t = x.shape
    min_lag = max(1, int(sr / fmax))
    max_lag = max(min_lag + 1, int(sr / fmin))
    frame = 1024
    hop = 256
    if t < frame:
        x = F.pad(x, (0, frame - t))
        t = x.shape[1]
    frames = x.unfold(1, frame, hop)  # [B, N, F]
    frames = frames - frames.mean(dim=-1, keepdim=True)
    energy = (frames * frames).sum(dim=-1, keepdim=True) + 1e-7
    frames = frames / torch.sqrt(energy)
    corrs = []
    for lag in range(min_lag, max_lag):
        a = frames[..., :-lag]
        b2 = frames[..., lag:]
        corrs.append((a * b2).mean(dim=-1))
    corr = torch.stack(corrs, dim=-1)
    best = corr.argmax(dim=-1).float() + float(min_lag)
    f0 = float(sr) / best
    return f0


def mel_stats(x: np.ndarray, sr: int, mel_bins: int, hop: int) -> np.ndarray:
    if librosa is None:
        # Fallback: rough stats in waveform domain
        return np.array([float(np.mean(x)), float(np.std(x))], dtype=np.float32)
    mel = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=mel_bins, hop_length=hop)
    logm = np.log(np.maximum(mel, 1e-6))
    mu = logm.mean(axis=1)
    sigma = logm.std(axis=1)
    return np.concatenate([mu, sigma], axis=0).astype(np.float32)


def main() -> int:
    args = parse_args()

    if np is None or torch is None or nn is None or F is None or sf is None:
        print("[error] Missing dependencies for local experimental training.")
        print("[hint] Install: numpy, torch, soundfile, librosa")
        return 2

    repo_root = Path(__file__).resolve().parent.parent.parent
    exp_dir = repo_root / "logs" / args.exp_dir
    exp_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = Path(args.dataset_dir).resolve() if args.dataset_dir else exp_dir / "0_gt_wavs"
    if not dataset_dir.exists():
        print("[error] dataset dir not found: %s" % dataset_dir)
        return 1

    wavs = load_wavs(dataset_dir, args.sample_rate)
    if not wavs:
        print("[error] no wav files found in: %s" % dataset_dir)
        return 1

    device = choose_device(args.device)
    print("[info] device:", device)
    print("[info] wav files:", len(wavs))

    model = TinyHQSVC().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))

    seg_n = int(args.segment_seconds * args.sample_rate)
    global_step = 0
    model_dir = exp_dir / "hqsvc_local"
    model_dir.mkdir(parents=True, exist_ok=True)

    target_stats_np = np.stack(
        [mel_stats(random_crop(w, seg_n), args.sample_rate, args.mel_bins, args.hop_feature) for w in wavs[: min(64, len(wavs))]],
        axis=0,
    )
    target_stats = torch.from_numpy(target_stats_np).to(device).mean(dim=0, keepdim=True)

    t0 = time.time()
    while global_step < args.steps:
        batch = np.stack([random_crop(random.choice(wavs), seg_n) for _ in range(args.batch_size)], axis=0)
        x = torch.from_numpy(batch).to(device).unsqueeze(1)

        # Denoising-consistency input for L_diff surrogate.
        noise = torch.randn_like(x) * 0.01
        x_noisy = (x + noise).clamp(-1.0, 1.0)

        y = model(x)
        y_noisy = model(x_noisy)

        y1 = y.squeeze(1)
        x1 = x.squeeze(1)
        y2 = y_noisy.squeeze(1)

        # L_ddsp: multi-resolution STFT magnitude distance.
        l_ddsp = 0.0
        for n_fft, hop in ((512, 128), (1024, 256), (2048, 512)):
            m_y = stft_mag(y1, n_fft=n_fft, hop=hop)
            m_x = stft_mag(x1, n_fft=n_fft, hop=hop)
            l_ddsp = l_ddsp + F.l1_loss(torch.log(m_y + 1e-6), torch.log(m_x + 1e-6))

        # L_diff: consistency under noise perturbation.
        l_diff = F.l1_loss(y1, y2)

        # L_f0: rough pitch contour match.
        f0_y = f0_autocorr(y1, args.sample_rate)
        f0_x = f0_autocorr(x1, args.sample_rate)
        l_f0 = F.l1_loss(torch.log(f0_y + 1e-6), torch.log(f0_x + 1e-6))

        # L_spk: timbre-statistics match to dataset centroid.
        y_cpu = y1.detach().float().cpu().numpy()
        y_stats_np = np.stack(
            [mel_stats(v, args.sample_rate, args.mel_bins, args.hop_feature) for v in y_cpu], axis=0
        )
        y_stats = torch.from_numpy(y_stats_np).to(device)
        l_spk = F.l1_loss(y_stats.mean(dim=0, keepdim=True), target_stats)

        loss = 1.0 * l_ddsp + 0.2 * l_diff + 0.1 * l_spk + 0.2 * l_f0

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        global_step += 1
        if global_step % 20 == 0:
            elapsed = time.time() - t0
            print(
                "[step %d/%d] loss=%.4f ddsp=%.4f diff=%.4f spk=%.4f f0=%.4f time=%.1fs"
                % (
                    global_step,
                    args.steps,
                    float(loss.item()),
                    float(l_ddsp.item()),
                    float(l_diff.item()),
                    float(l_spk.item()),
                    float(l_f0.item()),
                    elapsed,
                )
            )

        if global_step % args.save_every == 0 or global_step == args.steps:
            ckpt = {
                "model": model.state_dict(),
                "sample_rate": args.sample_rate,
                "encoder_sr": args.encoder_sr,
                "mel_bins": args.mel_bins,
                "hop_feature": args.hop_feature,
                "hop_infer": args.hop_infer,
                "global_step": global_step,
                "author": args.author,
                "version": "v3-local-experimental",
            }
            path = model_dir / ("G_%d.pt" % global_step)
            torch.save(ckpt, str(path))
            print("[save]", path)

    final_path = Path(args.output_checkpoint).resolve() if args.output_checkpoint else (model_dir / "G_latest.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "sample_rate": args.sample_rate,
            "encoder_sr": args.encoder_sr,
            "mel_bins": args.mel_bins,
            "hop_feature": args.hop_feature,
            "hop_infer": args.hop_infer,
            "global_step": global_step,
            "author": args.author,
            "version": "v3-local-experimental",
        },
        str(final_path),
    )

    meta = {
        "experiment": args.exp_dir,
        "dataset_dir": str(dataset_dir),
        "checkpoint": str(final_path),
        "sample_rate": args.sample_rate,
        "encoder_sr": args.encoder_sr,
        "mel_bins": args.mel_bins,
        "hop_feature": args.hop_feature,
        "hop_infer": args.hop_infer,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "losses": ["L_ddsp", "L_diff", "L_spk", "L_f0"],
        "paper_alignment": "experimental_approximation",
    }
    (exp_dir / "hqsvc_local_training.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("[done] local experimental training complete")
    print("[done] checkpoint:", final_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
