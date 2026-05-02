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
import random
import time
from collections import deque
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


_STFT_WINDOW_CACHE: dict[tuple[int, str], torch.Tensor] = {}
_MEL_FILTER_CACHE: dict[tuple[int, int, int, str], torch.Tensor] = {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experimental HQ-SVC-style local trainer")
    p.add_argument("--exp-dir", required=True, help="Experiment directory under logs/")
    p.add_argument("--dataset-dir", default="", help="Directory of target singer wav files")
    p.add_argument("--sample-rate", type=int, default=48000)
    p.add_argument("--encoder-sr", type=int, default=16000)
    p.add_argument("--hop-feature", type=int, default=512)
    p.add_argument("--hop-infer", type=int, default=256)
    p.add_argument("--mel-bins", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--segment-seconds", type=float, default=2.2)
    p.add_argument("--learning-rate", type=float, default=1.0e-4)
    p.add_argument("--save-every", type=int, default=300)
    p.add_argument("--smart-save", default="on", choices=["on", "off"])
    p.add_argument("--smart-save-window", type=int, default=10)
    p.add_argument("--smart-save-min-improve", type=float, default=2.0)
    p.add_argument("--smart-save-max-mel", type=float, default=16.0)
    p.add_argument("--smart-save-cooldown", type=int, default=200)
    p.add_argument("--smart-save-min-step", type=int, default=200)
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


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


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


def _cache_key(device: torch.device) -> str:
    index = getattr(device, "index", None)
    return "%s:%s" % (device.type, index if index is not None else -1)


def get_hann_window(n_fft: int, device: torch.device) -> torch.Tensor:
    key = (n_fft, _cache_key(device))
    win = _STFT_WINDOW_CACHE.get(key)
    if win is None:
        win = torch.hann_window(n_fft, device=device)
        _STFT_WINDOW_CACHE[key] = win
    return win


def get_mel_filter(
    sample_rate: int,
    n_fft: int,
    mel_bins: int,
    device: torch.device,
) -> torch.Tensor:
    key = (sample_rate, n_fft, mel_bins, _cache_key(device))
    mel = _MEL_FILTER_CACHE.get(key)
    if mel is None:
        if librosa is not None:
            mel_np = librosa.filters.mel(
                sr=sample_rate,
                n_fft=n_fft,
                n_mels=mel_bins,
                dtype=np.float32,
            )
        else:
            mel_np = np.eye(mel_bins, (n_fft // 2) + 1, dtype=np.float32)
        mel = torch.from_numpy(mel_np).to(device=device, dtype=torch.float32)
        _MEL_FILTER_CACHE[key] = mel
    return mel


def stft_mag(x: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    win = get_hann_window(n_fft, x.device)
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
    _, t = x.shape
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
    fft_size = 1 << ((2 * frame - 1).bit_length())
    spec = torch.fft.rfft(frames, n=fft_size, dim=-1)
    corr = torch.fft.irfft(spec * torch.conj(spec), n=fft_size, dim=-1)[..., min_lag:max_lag]
    best = corr.argmax(dim=-1).float() + float(min_lag)
    f0 = float(sr) / best
    return f0


def mel_stats_tensor(
    x: torch.Tensor,
    sr: int,
    mel_bins: int,
    hop: int,
    n_fft: int = 1024,
) -> torch.Tensor:
    if x.dim() == 1:
        x = x.unsqueeze(0)

    if librosa is None:
        mu = x.mean(dim=-1, keepdim=True)
        sigma = x.std(dim=-1, keepdim=True)
        return torch.cat([mu, sigma], dim=-1)

    win = get_hann_window(n_fft, x.device)
    spec = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=win,
        return_complex=True,
    )
    power = spec.abs().pow(2.0)
    mel_filter = get_mel_filter(sr, n_fft, mel_bins, x.device)
    mel = torch.matmul(mel_filter.unsqueeze(0), power)
    logm = torch.log(torch.clamp(mel, min=1e-6))
    mu = logm.mean(dim=-1)
    sigma = logm.std(dim=-1)
    return torch.cat([mu, sigma], dim=-1)


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

    t0 = time.perf_counter()
    wavs = load_wavs(dataset_dir, args.sample_rate)
    dataset_load_time = time.perf_counter() - t0
    if not wavs:
        print("[error] no wav files found in: %s" % dataset_dir)
        return 1

    device = choose_device(args.device)
    print("[info] device:", device)
    print("[info] wav files:", len(wavs))
    print("[timing] dataset_load=%.3fs" % dataset_load_time)

    model = TinyHQSVC().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))

    seg_n = int(args.segment_seconds * args.sample_rate)
    global_step = 0
    model_dir = exp_dir / "hqsvc_local"
    model_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    target_batch = np.stack(
        [
            random_crop(w, seg_n)
            for w in wavs[: min(64, len(wavs))]
        ],
        axis=0,
    )
    target_batch_t = torch.from_numpy(target_batch).to(device=device, dtype=torch.float32)
    target_stats = mel_stats_tensor(
        target_batch_t,
        args.sample_rate,
        args.mel_bins,
        args.hop_feature,
    ).mean(dim=0, keepdim=True)
    sync_device(device)
    print("[timing] target_stats=%.3fs" % (time.perf_counter() - t0))

    t0 = time.time()
    phase_times = {
        "batch": 0.0,
        "forward": 0.0,
        "ddsp": 0.0,
        "f0": 0.0,
        "spk": 0.0,
        "backward": 0.0,
    }
    phase_steps = 0
    mel_history = deque(maxlen=max(1, int(args.smart_save_window)))
    smart_save_enabled = str(args.smart_save).lower() == "on"
    last_smart_save_step = -10**9
    while global_step < args.steps:
        phase_start = time.perf_counter()
        batch = np.stack([random_crop(random.choice(wavs), seg_n) for _ in range(args.batch_size)], axis=0)
        x = torch.from_numpy(batch).to(device).unsqueeze(1)
        sync_device(device)
        phase_times["batch"] += time.perf_counter() - phase_start

        # Denoising-consistency input for L_diff surrogate.
        noise = torch.randn_like(x) * 0.01
        x_noisy = (x + noise).clamp(-1.0, 1.0)

        phase_start = time.perf_counter()
        y = model(x)
        y_noisy = model(x_noisy)
        sync_device(device)
        phase_times["forward"] += time.perf_counter() - phase_start

        y1 = y.squeeze(1)
        x1 = x.squeeze(1)
        y2 = y_noisy.squeeze(1)

        # L_ddsp: multi-resolution STFT magnitude distance.
        phase_start = time.perf_counter()
        l_ddsp = 0.0
        for n_fft, hop in ((512, 128), (1024, 256), (2048, 512)):
            m_y = stft_mag(y1, n_fft=n_fft, hop=hop)
            m_x = stft_mag(x1, n_fft=n_fft, hop=hop)
            l_ddsp = l_ddsp + F.l1_loss(torch.log(m_y + 1e-6), torch.log(m_x + 1e-6))
        sync_device(device)
        phase_times["ddsp"] += time.perf_counter() - phase_start

        # L_diff: consistency under noise perturbation.
        l_diff = F.l1_loss(y1, y2)

        # L_f0: rough pitch contour match.
        phase_start = time.perf_counter()
        f0_y = f0_autocorr(y1, args.sample_rate)
        f0_x = f0_autocorr(x1, args.sample_rate)
        l_f0 = F.l1_loss(torch.log(f0_y + 1e-6), torch.log(f0_x + 1e-6))
        sync_device(device)
        phase_times["f0"] += time.perf_counter() - phase_start

        # L_spk: timbre-statistics match to dataset centroid, kept on-device.
        phase_start = time.perf_counter()
        y_stats = mel_stats_tensor(
            y1,
            args.sample_rate,
            args.mel_bins,
            args.hop_feature,
        )
        l_spk = F.l1_loss(y_stats.mean(dim=0, keepdim=True), target_stats)
        sync_device(device)
        phase_times["spk"] += time.perf_counter() - phase_start

        loss = 1.0 * l_ddsp + 0.2 * l_diff + 0.1 * l_spk + 0.2 * l_f0

        phase_start = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sync_device(device)
        phase_times["backward"] += time.perf_counter() - phase_start

        global_step += 1
        phase_steps += 1
        loss_mel = float(l_ddsp.item())

        saved_this_step = False
        if (
            smart_save_enabled
            and global_step >= int(args.smart_save_min_step)
            and len(mel_history) >= max(1, int(args.smart_save_window))
            and (global_step - last_smart_save_step) >= int(args.smart_save_cooldown)
        ):
            avg_prev = float(sum(mel_history) / len(mel_history))
            improve = avg_prev - loss_mel
            max_mel_ok = float(args.smart_save_max_mel) <= 0 or loss_mel <= float(args.smart_save_max_mel)
            if improve >= float(args.smart_save_min_improve) and max_mel_ok:
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
                smart_path = model_dir / ("G_smart_%d.pt" % global_step)
                torch.save(ckpt, str(smart_path))
                print(
                    "[smart-save] %s loss_mel=%.4f avg_prev_%d=%.4f improve=%.4f"
                    % (smart_path, loss_mel, len(mel_history), avg_prev, improve)
                )
                last_smart_save_step = global_step
                saved_this_step = True

        mel_history.append(loss_mel)

        if global_step % 20 == 0:
            elapsed = time.time() - t0
            avg_phase = ", ".join(
                "%s=%.1fms" % (key, (phase_times[key] / max(1, phase_steps)) * 1000.0)
                for key in ["batch", "forward", "ddsp", "f0", "spk", "backward"]
            )
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
            print("[timing] avg_per_step %s" % avg_phase)
            for key in phase_times:
                phase_times[key] = 0.0
            phase_steps = 0

        if (global_step % args.save_every == 0 or global_step == args.steps) and not saved_this_step:
            save_start = time.perf_counter()
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
            print("[timing] checkpoint_save=%.3fs" % (time.perf_counter() - save_start))

    final_path = Path(args.output_checkpoint).resolve() if args.output_checkpoint else (model_dir / "G_latest.pt")
    save_start = time.perf_counter()
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
    print("[timing] final_checkpoint_save=%.3fs" % (time.perf_counter() - save_start))

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
        "smart_save": {
            "enabled": smart_save_enabled,
            "window": int(args.smart_save_window),
            "min_improve": float(args.smart_save_min_improve),
            "max_mel": float(args.smart_save_max_mel),
            "cooldown": int(args.smart_save_cooldown),
            "min_step": int(args.smart_save_min_step),
        },
        "losses": ["L_ddsp", "L_diff", "L_spk", "L_f0"],
        "paper_alignment": "experimental_approximation",
    }
    (exp_dir / "hqsvc_local_training.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("[done] local experimental training complete")
    print("[done] checkpoint:", final_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
