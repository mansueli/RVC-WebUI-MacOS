#!/usr/bin/env python3
"""Paper-aligned HQ-SVC training scaffold.

This backend is a closer structural approximation of the HQ-SVC paper than the
local experimental waveform trainer. It keeps the implementation self-contained
and runnable on macOS/MPS by using lightweight proxy modules for FACodec/EVA/
DDSP/diffusion while matching the paper's data shapes and loss layout.
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

try:
    from rvc.f0.rmvpe import RMVPE
except Exception:  # pragma: no cover
    RMVPE = None

try:
    from tools.cmd.rvc_discriminator import (
        MultiPeriodDiscriminator,
        discriminator_loss,
        feature_loss,
        generator_loss,
    )
except Exception:  # pragma: no cover
    MultiPeriodDiscriminator = None
    discriminator_loss = None
    feature_loss = None
    generator_loss = None


_STFT_WINDOW_CACHE: dict[tuple[int, str], torch.Tensor] = {}
_MEL_FILTER_CACHE: dict[tuple[int, int, int, str], torch.Tensor] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-aligned HQ-SVC trainer")
    parser.add_argument("--exp-dir", required=True)
    parser.add_argument("--dataset-dir", default="")
    parser.add_argument("--stage", default="1", choices=["1", "2"])
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--encoder-sr", type=int, default=16000)
    parser.add_argument("--mel-bins", type=int, default=128)
    parser.add_argument("--hop-feature", type=int, default=512)
    parser.add_argument("--hop-infer", type=int, default=256)
    parser.add_argument("--min-clip-seconds", type=float, default=2.1)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--segment-seconds", type=float, default=2.8)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--save-every", type=int, default=400)
    parser.add_argument("--smart-save", default="on", choices=["on", "off"])
    parser.add_argument("--smart-save-window", type=int, default=10)
    parser.add_argument("--smart-save-min-improve", type=float, default=2.0)
    parser.add_argument("--smart-save-max-mel", type=float, default=16.0)
    parser.add_argument("--smart-save-cooldown", type=int, default=200)
    parser.add_argument("--smart-save-min-step", type=int, default=200)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--author", default="")
    parser.add_argument("--output-checkpoint", default="")
    parser.add_argument("--discriminator-checkpoint", default="assets/pretrained_v2/f0D48k.pth")
    parser.add_argument("--rmvpe", default="auto", choices=["auto", "on", "off"])
    return parser.parse_args()


def _load_checkpoint(path: Path, device: torch.device) -> dict | None:
    if not path.exists():
        return None
    return torch.load(str(path), map_location=device)


def _load_model_state(model: nn.Module, payload: dict, prefix: str = "model") -> tuple[list[str], list[str]]:
    missing: list[str] = []
    unexpected: list[str] = []
    state = payload.get(prefix)
    if state:
        incompatible = model.load_state_dict(state, strict=False)
        missing = list(getattr(incompatible, "missing_keys", []))
        unexpected = list(getattr(incompatible, "unexpected_keys", []))
    return missing, unexpected


def _resolve_init_checkpoint(exp_dir: Path, init_checkpoint: str, stage: str) -> Path | None:
    if init_checkpoint:
        return Path(init_checkpoint).resolve()
    if stage != "2":
        return None
    model_dir = exp_dir / "hqsvc_full"
    for candidate in [
        model_dir / "G_stage1_latest.pt",
        model_dir / "G_latest.pt",
        model_dir / "G_8000.pt",
    ]:
        if candidate.exists():
            return candidate
    matches = sorted(model_dir.glob("G_*.pt"))
    return matches[-1] if matches else None


def _checkpoint_payload(
    args: argparse.Namespace,
    model: nn.Module,
    global_step: int,
    speaker_count: int,
    stage: str,
    discriminator: nn.Module | None = None,
) -> dict:
    payload = {
        "model": model.state_dict(),
        "sample_rate": args.sample_rate,
        "encoder_sr": args.encoder_sr,
        "mel_bins": args.mel_bins,
        "hop_feature": args.hop_feature,
        "hop_infer": args.hop_infer,
        "global_step": global_step,
        "speaker_count": speaker_count,
        "author": args.author,
        "version": "v3-full-paper-rvc-ft" if stage == "2" else "v3-full-paper-experimental",
        "training_stage": stage,
    }
    if discriminator is not None:
        payload["discriminator"] = discriminator.state_dict()
    return payload


def _extract_state_dict(payload: object) -> dict | None:
    if isinstance(payload, dict):
        if payload and all(isinstance(k, str) for k in payload.keys()):
            first_val = next(iter(payload.values()))
            if torch.is_tensor(first_val):
                return payload
        for key in ["model", "weight", "discriminator", "state_dict"]:
            value = payload.get(key)
            if isinstance(value, dict):
                inner = _extract_state_dict(value)
                if inner:
                    return inner
    return None


def _load_discriminator_checkpoint(discriminator: nn.Module, ckpt_path: Path, device: torch.device) -> None:
    if not ckpt_path.exists():
        print("[warn] discriminator checkpoint not found:", ckpt_path)
        return
    try:
        raw_payload = torch.load(str(ckpt_path), map_location=device)
    except Exception as exc:
        print("[warn] failed to read discriminator checkpoint %s: %s" % (ckpt_path, exc))
        return
    state = _extract_state_dict(raw_payload)
    if not state:
        print("[warn] no discriminator-like state dict found in:", ckpt_path)
        return

    cleaned = {}
    for key, value in state.items():
        if not isinstance(key, str):
            continue
        new_key = key[7:] if key.startswith("module.") else key
        cleaned[new_key] = value
    incompatible = discriminator.load_state_dict(cleaned, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    print(
        "[info] discriminator init loaded from %s (missing=%d unexpected=%d)"
        % (ckpt_path, len(missing), len(unexpected))
    )


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


def _cache_key(device: torch.device) -> str:
    index = getattr(device, "index", None)
    return "%s:%s" % (device.type, index if index is not None else -1)


def get_hann_window(n_fft: int, device: torch.device) -> torch.Tensor:
    key = (n_fft, _cache_key(device))
    window = _STFT_WINDOW_CACHE.get(key)
    if window is None:
        window = torch.hann_window(n_fft, device=device)
        _STFT_WINDOW_CACHE[key] = window
    return window


def get_mel_filter(sample_rate: int, n_fft: int, mel_bins: int, device: torch.device) -> torch.Tensor:
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


def waveform_to_mel(x: torch.Tensor, sr: int, mel_bins: int, hop: int, n_fft: int = 2048) -> torch.Tensor:
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
    return torch.log(torch.clamp(mel, min=1e-6))


def f0_autocorr(x: torch.Tensor, sr: int, fmin: float = 60.0, fmax: float = 900.0) -> torch.Tensor:
    _, t = x.shape
    min_lag = max(1, int(sr / fmax))
    max_lag = max(min_lag + 1, int(sr / fmin))
    frame = 1024
    hop = 256
    if t < frame:
        x = F.pad(x, (0, frame - t))
    frames = x.unfold(1, frame, hop)
    frames = frames - frames.mean(dim=-1, keepdim=True)
    energy = (frames * frames).sum(dim=-1, keepdim=True) + 1e-7
    frames = frames / torch.sqrt(energy)
    fft_size = 1 << ((2 * frame - 1).bit_length())
    spec = torch.fft.rfft(frames, n=fft_size, dim=-1)
    corr = torch.fft.irfft(spec * torch.conj(spec), n=fft_size, dim=-1)[..., min_lag:max_lag]
    best = corr.argmax(dim=-1).float() + float(min_lag)
    return float(sr) / best


def rms_stats(x: torch.Tensor, frame: int = 1024, hop: int = 256) -> torch.Tensor:
    if x.shape[-1] < frame:
        x = F.pad(x, (0, frame - x.shape[-1]))
    frames = x.unfold(1, frame, hop)
    rms = torch.sqrt(torch.clamp(frames.pow(2.0).mean(dim=-1), min=1e-7))
    return torch.stack([rms.mean(dim=-1), rms.std(dim=-1)], dim=-1)


def phase_stats(x: torch.Tensor, n_fft: int = 1024, hop: int = 256) -> torch.Tensor:
    win = get_hann_window(n_fft, x.device)
    spec = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=win,
        return_complex=True,
    )
    phase = torch.angle(spec)
    return torch.stack([phase.mean(dim=(-2, -1)), phase.std(dim=(-2, -1))], dim=-1)


def build_speaker_table(dataset_dir: Path, wav_paths: list[Path]) -> dict[str, int]:
    speakers: dict[str, int] = {}
    for wav_path in wav_paths:
        rel_parent = wav_path.parent.relative_to(dataset_dir)
        speaker_name = str(rel_parent) if str(rel_parent) != "." else "speaker_0"
        if speaker_name not in speakers:
            speakers[speaker_name] = len(speakers)
    return speakers


def maybe_make_rmvpe(device: torch.device, mode: str):
    if mode == "off" or RMVPE is None:
        return None
    repo_root = Path(__file__).resolve().parent.parent.parent
    model_path = repo_root / "assets" / "rmvpe" / "rmvpe.pt"
    if not model_path.exists():
        return None
    try:
        return RMVPE(str(model_path), device=str(device), is_half=False)
    except Exception:
        return None


def extract_f0_stats_np(wav: np.ndarray, sample_rate: int, rmvpe_model) -> np.ndarray:
    if rmvpe_model is not None:
        f0 = rmvpe_model.compute_f0(wav, p_len=max(1, wav.shape[0] // 256), filter_radius=0.03)
        f0 = np.asarray(f0, dtype=np.float32)
        voiced = f0[f0 > 0]
        if voiced.size > 0:
            return np.array([float(voiced.mean()), float(voiced.std())], dtype=np.float32)
    x = torch.from_numpy(wav).float().unsqueeze(0)
    f0 = f0_autocorr(x, sample_rate).squeeze(0).cpu().numpy().astype(np.float32)
    return np.array([float(f0.mean()), float(f0.std())], dtype=np.float32)


def load_dataset(dataset_dir: Path, sample_rate: int, min_clip_seconds: float, rmvpe_model):
    min_samples = int(min_clip_seconds * sample_rate)
    wav_paths = sorted(dataset_dir.rglob("*.wav"))
    if not wav_paths:
        return [], {}
    speaker_table = build_speaker_table(dataset_dir, wav_paths)
    samples = []
    for wav_path in wav_paths:
        wav, sr = sf.read(str(wav_path), always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = wav.astype(np.float32)
        if sr != sample_rate:
            if librosa is None:
                continue
            wav = librosa.resample(wav, orig_sr=sr, target_sr=sample_rate)
        if wav.shape[0] < min_samples:
            continue
        rel_parent = wav_path.parent.relative_to(dataset_dir)
        speaker_name = str(rel_parent) if str(rel_parent) != "." else "speaker_0"
        samples.append(
            {
                "path": str(wav_path),
                "wav": wav,
                "speaker_id": speaker_table[speaker_name],
                "f0_stats": extract_f0_stats_np(wav, sample_rate, rmvpe_model),
            }
        )
    return samples, speaker_table


def random_crop(wav: np.ndarray, n: int) -> np.ndarray:
    if wav.shape[0] <= n:
        out = np.zeros(n, dtype=np.float32)
        out[: wav.shape[0]] = wav
        return out
    start = random.randint(0, wav.shape[0] - n)
    return wav[start : start + n]


class EVAModule(nn.Module):
    def __init__(self, hidden_dim: int, speaker_dim: int, cond_dim: int):
        super().__init__()
        self.film = nn.Sequential(
            nn.Linear(speaker_dim + cond_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
                activation="gelu",
            ),
            num_layers=2,
        )

    def forward(self, content: torch.Tensor, speaker: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(torch.cat([speaker, cond], dim=-1)).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        fused = content * (1.0 + torch.tanh(gamma)) + beta
        return self.encoder(fused.transpose(1, 2)).transpose(1, 2)


class DDSPHead(nn.Module):
    def __init__(self, hidden_dim: int, mel_bins: int):
        super().__init__()
        self.proj = nn.Conv1d(hidden_dim, mel_bins, kernel_size=1)

    def forward(self, fused: torch.Tensor, frames: int) -> torch.Tensor:
        pooled = F.adaptive_avg_pool1d(fused, frames)
        return self.proj(pooled)


class DiffusionRefiner(nn.Module):
    def __init__(self, mel_bins: int, cond_dim: int):
        super().__init__()
        self.cond = nn.Linear(cond_dim, mel_bins)
        self.net = nn.Sequential(
            nn.Conv1d(mel_bins, mel_bins * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(mel_bins * 2, mel_bins * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(mel_bins * 2, mel_bins, kernel_size=3, padding=1),
        )

    def forward(self, noisy_mel: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        cond_bias = self.cond(cond).unsqueeze(-1)
        return self.net(noisy_mel + cond_bias)


class LightweightVocoder(nn.Module):
    def __init__(self, mel_bins: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(mel_bins, mel_bins, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(mel_bins, mel_bins // 2, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(mel_bins // 2, 1, kernel_size=7, padding=3),
        )

    def forward(self, mel: torch.Tensor, target_length: int) -> torch.Tensor:
        x = F.interpolate(mel, size=target_length, mode="linear", align_corners=False)
        return torch.tanh(self.net(x)).squeeze(1)


class FullPaperHQSVC(nn.Module):
    def __init__(self, num_speakers: int, mel_bins: int, hidden_dim: int = 128, speaker_dim: int = 128):
        super().__init__()
        self.content_encoder = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.speaker_proj = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, speaker_dim),
        )
        self.eva = EVAModule(hidden_dim, speaker_dim, cond_dim=6)
        self.ddsp = DDSPHead(hidden_dim, mel_bins)
        self.diffusion = DiffusionRefiner(mel_bins, cond_dim=6)
        self.vocoder = LightweightVocoder(mel_bins)
        self.speaker_table = nn.Embedding(max(1, num_speakers), speaker_dim)
        nn.init.normal_(self.speaker_table.weight, mean=0.0, std=0.02)

    def forward(self, wav: torch.Tensor, cond: torch.Tensor, frames: int) -> dict[str, torch.Tensor]:
        content = self.content_encoder(wav.unsqueeze(1))
        speaker = F.normalize(self.speaker_proj(content), dim=-1)
        fused = self.eva(content, speaker, cond)
        mel_ddsp = self.ddsp(fused, frames)
        noise = torch.randn_like(mel_ddsp)
        noisy_mel = mel_ddsp + (0.1 * noise)
        noise_pred = self.diffusion(noisy_mel, cond)
        mel_refined = noisy_mel - (0.1 * noise_pred)
        audio = self.vocoder(mel_refined, wav.shape[-1])
        speaker_logits = torch.matmul(
            F.normalize(speaker, dim=-1),
            F.normalize(self.speaker_table.weight, dim=-1).transpose(0, 1),
        ) / 0.1
        return {
            "mel_ddsp": mel_ddsp,
            "mel_refined": mel_refined,
            "noise": noise,
            "noise_pred": noise_pred,
            "audio": audio,
            "speaker_logits": speaker_logits,
        }


def build_condition(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    f0 = f0_autocorr(audio, sample_rate)
    f0_stats = torch.stack([f0.mean(dim=-1), f0.std(dim=-1)], dim=-1)
    vol = rms_stats(audio)
    pha = phase_stats(audio)
    return torch.cat([f0_stats, vol, pha], dim=-1)


def main() -> int:
    args = parse_args()
    if np is None or torch is None or nn is None or F is None or sf is None:
        print("[error] Missing dependencies for paper-mode training.")
        print("[hint] Install: numpy, torch, soundfile, librosa")
        return 2
    if args.stage == "2" and (
        MultiPeriodDiscriminator is None
        or discriminator_loss is None
        or feature_loss is None
        or generator_loss is None
    ):
        print("[error] RVC discriminator components are unavailable for Stage 2 fine-tuning.")
        return 2

    repo_root = Path(__file__).resolve().parent.parent.parent
    exp_dir = repo_root / "logs" / args.exp_dir
    exp_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args.dataset_dir).resolve() if args.dataset_dir else exp_dir / "0_gt_wavs"
    if not dataset_dir.exists():
        print("[error] dataset dir not found:", dataset_dir)
        return 1

    device = choose_device(args.device)
    rmvpe_model = maybe_make_rmvpe(device, args.rmvpe)
    print("[info] device:", device)
    print("[info] rmvpe:", "enabled" if rmvpe_model is not None else "fallback")

    load_start = time.perf_counter()
    samples, speaker_table = load_dataset(dataset_dir, args.sample_rate, args.min_clip_seconds, rmvpe_model)
    print("[timing] dataset_load=%.3fs" % (time.perf_counter() - load_start))
    if not samples:
        print("[error] no usable wav clips found in:", dataset_dir)
        return 1

    model_dir = exp_dir / "hqsvc_full"
    model_dir.mkdir(parents=True, exist_ok=True)
    init_checkpoint = _resolve_init_checkpoint(exp_dir, args.init_checkpoint, args.stage)
    model = FullPaperHQSVC(len(speaker_table), args.mel_bins).to(device)
    discriminator = None
    if args.stage == "2":
        discriminator = MultiPeriodDiscriminator("v2").to(device)
        disc_ckpt = Path(args.discriminator_checkpoint)
        if not disc_ckpt.is_absolute():
            disc_ckpt = (repo_root / disc_ckpt).resolve()
        _load_discriminator_checkpoint(discriminator, disc_ckpt, device)

    if init_checkpoint is not None:
        payload = _load_checkpoint(init_checkpoint, device)
        if payload is None:
            print("[error] init checkpoint not found:", init_checkpoint)
            return 1
        missing, unexpected = _load_model_state(model, payload, "model")
        if discriminator is not None and payload.get("discriminator"):
            disc_missing, disc_unexpected = _load_model_state(discriminator, payload, "discriminator")
            if disc_missing or disc_unexpected:
                print("[warn] discriminator checkpoint mismatch missing=%d unexpected=%d" % (len(disc_missing), len(disc_unexpected)))
        print("[info] loaded checkpoint:", init_checkpoint)
        if missing or unexpected:
            print("[warn] model checkpoint mismatch missing=%d unexpected=%d" % (len(missing), len(unexpected)))
        global_step = int(payload.get("global_step", 0) or 0)
    else:
        global_step = 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    optimizer_d = None
    if discriminator is not None:
        optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    seg_n = int(args.segment_seconds * args.sample_rate)

    phase_times = {"batch": 0.0, "cond": 0.0, "forward": 0.0, "loss": 0.0, "backward": 0.0}
    phase_steps = 0
    mel_history = deque(maxlen=max(1, int(args.smart_save_window)))
    smart_save_enabled = str(args.smart_save).lower() == "on"
    last_smart_save_step = -10**9
    wall_start = time.time()
    while global_step < args.steps:
        step_start = time.perf_counter()
        chosen = [random.choice(samples) for _ in range(args.batch_size)]
        batch_np = np.stack([random_crop(s["wav"], seg_n) for s in chosen], axis=0)
        speaker_ids = torch.tensor([s["speaker_id"] for s in chosen], device=device, dtype=torch.long)
        target_f0_stats = torch.tensor(np.stack([s["f0_stats"] for s in chosen], axis=0), device=device)
        audio = torch.from_numpy(batch_np).to(device=device, dtype=torch.float32)
        sync_device(device)
        phase_times["batch"] += time.perf_counter() - step_start

        cond_start = time.perf_counter()
        cond = build_condition(audio, args.sample_rate)
        target_mel = waveform_to_mel(audio, args.sample_rate, args.mel_bins, args.hop_feature)
        sync_device(device)
        phase_times["cond"] += time.perf_counter() - cond_start

        forward_start = time.perf_counter()
        out = model(audio, cond, frames=target_mel.shape[-1])
        recon_mel = waveform_to_mel(out["audio"], args.sample_rate, args.mel_bins, args.hop_feature)
        pred_f0 = f0_autocorr(out["audio"], args.sample_rate)
        pred_f0_stats = torch.stack([pred_f0.mean(dim=-1), pred_f0.std(dim=-1)], dim=-1)
        sync_device(device)
        phase_times["forward"] += time.perf_counter() - forward_start

        loss_start = time.perf_counter()
        l_ddsp = F.mse_loss(out["mel_ddsp"], target_mel) + 0.5 * F.mse_loss(recon_mel, target_mel)
        l_diff = F.mse_loss(out["noise_pred"], out["noise"])
        l_spk = F.cross_entropy(out["speaker_logits"], speaker_ids)
        l_f0 = F.l1_loss(pred_f0_stats, target_f0_stats)
        base_loss = (1.0 * l_ddsp) + (1.0 * l_diff) + (0.1 * l_spk) + (0.2 * l_f0)
        l_adv = torch.zeros((), device=device)
        l_fm = torch.zeros((), device=device)
        l_disc = torch.zeros((), device=device)
        sync_device(device)
        phase_times["loss"] += time.perf_counter() - loss_start

        backward_start = time.perf_counter()
        if discriminator is not None and optimizer_d is not None:
            discriminator.train()
            real_audio = audio.unsqueeze(1)
            fake_audio = out["audio"].unsqueeze(1)
            optimizer_d.zero_grad(set_to_none=True)
            y_d_r, y_d_g, _, _ = discriminator(real_audio, fake_audio.detach())
            l_disc, _, _ = discriminator_loss(y_d_r, y_d_g)
            l_disc.backward()
            nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
            optimizer_d.step()

            optimizer.zero_grad(set_to_none=True)
            y_d_r, y_d_g, fmap_r, fmap_g = discriminator(real_audio, fake_audio)
            l_adv, _ = generator_loss(y_d_g)
            l_fm = feature_loss(fmap_r, fmap_g)
            loss = (0.75 * base_loss) + (0.05 * l_adv) + (0.5 * l_fm)
            loss.backward()
        else:
            optimizer.zero_grad(set_to_none=True)
            loss = base_loss
            loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        sync_device(device)
        phase_times["backward"] += time.perf_counter() - backward_start

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
                smart_path = model_dir / ("G_smart_%d.pt" % global_step)
                torch.save(
                    _checkpoint_payload(args, model, global_step, len(speaker_table), args.stage, discriminator),
                    str(smart_path),
                )
                print(
                    "[smart-save] %s loss_mel=%.4f avg_prev_%d=%.4f improve=%.4f"
                    % (smart_path, loss_mel, len(mel_history), avg_prev, improve)
                )
                last_smart_save_step = global_step
                saved_this_step = True

        mel_history.append(loss_mel)

        if global_step % 20 == 0:
            elapsed = time.time() - wall_start
            avg_phase = ", ".join(
                "%s=%.1fms" % (key, (phase_times[key] / max(1, phase_steps)) * 1000.0)
                for key in ["batch", "cond", "forward", "loss", "backward"]
            )
            print(
                "[step %d/%d][stage %s] loss=%.4f ddsp=%.4f diff=%.4f spk=%.4f f0=%.4f adv=%.4f fm=%.4f disc=%.4f time=%.1fs"
                % (
                    global_step,
                    args.steps,
                    args.stage,
                    float(loss.item()),
                    float(l_ddsp.item()),
                    float(l_diff.item()),
                    float(l_spk.item()),
                    float(l_f0.item()),
                    float(l_adv.item()),
                    float(l_fm.item()),
                    float(l_disc.item()),
                    elapsed,
                )
            )
            print("[timing] avg_per_step %s" % avg_phase)
            for key in phase_times:
                phase_times[key] = 0.0
            phase_steps = 0

        if (global_step % args.save_every == 0 or global_step == args.steps) and not saved_this_step:
            save_path = model_dir / ("G_%d.pt" % global_step)
            torch.save(_checkpoint_payload(args, model, global_step, len(speaker_table), args.stage, discriminator), str(save_path))
            print("[save]", save_path)

    default_name = "G_stage2_latest.pt" if args.stage == "2" else "G_stage1_latest.pt"
    final_path = Path(args.output_checkpoint).resolve() if args.output_checkpoint else (model_dir / default_name)
    final_payload = _checkpoint_payload(args, model, global_step, len(speaker_table), args.stage, discriminator)
    torch.save(final_payload, str(final_path))

    export_name = "%s_v3_stage%s.pth" % (args.exp_dir.replace("/", "_").replace("\\", "_"), args.stage)
    export_path = repo_root / "assets" / "weights" / export_name
    export_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(final_payload, str(export_path))

    meta = {
        "experiment": args.exp_dir,
        "dataset_dir": str(dataset_dir),
        "checkpoint": str(final_path),
        "exported_weight": str(export_path),
        "init_checkpoint": str(init_checkpoint) if init_checkpoint is not None else "",
        "discriminator_checkpoint": args.discriminator_checkpoint,
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
        "stage": args.stage,
        "losses": ["L_ddsp", "L_diff", "L_spk", "L_f0"] + (["L_adv", "L_fm", "L_disc"] if args.stage == "2" else []),
        "speaker_count": len(speaker_table),
        "paper_alignment": "full_paper_mode_scaffold",
        "rvc_fine_tuned": args.stage == "2",
    }
    (exp_dir / "hqsvc_full_training.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("[done] paper-mode training complete")
    print("[done] checkpoint:", final_path)
    print("[done] exported weight:", export_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
