#!/usr/bin/env python3
"""Simulate an HQ-SVC-style training plan for this repository.

This does NOT run HQ-SVC training directly. Instead, it builds a reproducible
training blueprint aligned with the HQ-SVC paper and maps it to this repo's
practical workflow.

Paper-aligned defaults captured here:
- sample_rate: 44.1 kHz
- encoder_sr: 16 kHz
- mel bins: 128
- hop sizes: 512 for feature extraction, 256 for vocoder/inference path
- optimizer: AdamW, betas (0.9, 0.999), lr 1.5e-4
- batch size: 64
- training steps: 250,000
- losses: L_ddsp + L_diff + L_spk + L_f0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create HQ-SVC simulation blueprint")
    parser.add_argument("--exp-name", default="hqsvc_sim_v3")
    parser.add_argument("--dataset-dir", required=True, help="Prepared training wav folder")
    parser.add_argument(
        "--isolated-vocals-dir",
        default="",
        help="Optional YingMusic-isolated vocals folder",
    )
    parser.add_argument("--steps", type=int, default=250000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument(
        "--webui-sample-rate",
        type=int,
        default=48000,
        choices=[32000, 40000, 48000],
        help="Sample rate used by this WebUI training pipeline.",
    )
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def count_wavs(path: Path) -> int:
    if not path.exists():
        return 0
    return len(list(path.rglob("*.wav")))


def main() -> int:
    args = parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    isolated_dir = Path(args.isolated_vocals_dir).resolve() if args.isolated_vocals_dir else None

    if not dataset_dir.exists():
        print(f"[error] dataset-dir does not exist: {dataset_dir}")
        return 2

    wav_count = count_wavs(dataset_dir)
    isolated_wav_count = count_wavs(isolated_dir) if isolated_dir else 0

    blueprint = {
        "experiment": args.exp_name,
        "goal": "Simulate HQ-SVC-style low-resource singing training in this repo",
        "paper_alignment": {
            "sample_rate": 44100,
            "encoder_sample_rate": 16000,
            "mel_bins": 128,
            "hop_size_feature": 512,
            "hop_size_infer": 256,
            "f0_extractor": "rmvpe",
            "optimizer": "AdamW",
            "optimizer_betas": [0.9, 0.999],
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "losses": ["L_ddsp", "L_diff", "L_spk", "L_f0"],
        },
        "dataset": {
            "dataset_dir": str(dataset_dir),
            "wav_count": wav_count,
            "isolated_vocals_dir": str(isolated_dir) if isolated_dir else "",
            "isolated_wav_count": isolated_wav_count,
        },
        "repo_mapping": {
            "preprocessing": {
                "script": "training_data/prepare_rvc_dataset.py",
                "paper_target_sample_rate": 44100,
                "webui_target_sample_rate": args.webui_sample_rate,
                "recommended_args": [
                    "--target-sample-rate", "44100",
                    "--resample-if-needed",
                    "--isolated-vocals-dir", str(isolated_dir) if isolated_dir else "<optional>",
                ],
            },
            "feature_extraction": {
                "source": "Existing RVC stack (HuBERT + RMVPE) as practical surrogate",
                "note": (
                    "HQ-SVC exact FACodec/EVA training cannot be fully replicated until "
                    "upstream training code path is fully published and integrated."
                ),
            },
            "training_strategy": {
                "stage_1": "Train/finetune current RVC model with cleaned singing-only data",
                "stage_2": "Use HQ-SVC pretrained inference outputs as teacher-style A/B target",
                "stage_3": "Promote to native HQ-SVC training path when upstream training path is stable",
            },
        },
        "execution_commands": {
            "prepare_paper_44k": (
                "python training_data/prepare_rvc_dataset.py "
                "--source-folders singing augumented_singing "
                + (
                    f"--isolated-vocals-dir '{isolated_dir}' " if isolated_dir else ""
                )
                + "--target-sample-rate 44100 --resample-if-needed"
            ),
            "prepare_webui_training_sr": (
                "python training_data/prepare_rvc_dataset.py "
                "--source-folders singing augumented_singing "
                + (
                    f"--isolated-vocals-dir '{isolated_dir}' " if isolated_dir else ""
                )
                + f"--target-sample-rate {args.webui_sample_rate} --resample-if-needed"
            ),
            "hq_pretrained_setup": "python tools/cmd/hqsvc_experiment.py --setup-only",
            "yingmusic_setup": "python tools/cmd/yingmusic_experiment.py --setup-only",
        },
    }

    output_json = (
        Path(args.output_json).resolve()
        if args.output_json
        else Path("logs") / args.exp_name / "hqsvc_training_simulation.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(blueprint, indent=2), encoding="utf-8")

    print(f"[done] Wrote simulation blueprint: {output_json}")
    print(f"[info] Dataset wav count: {wav_count}")
    if isolated_dir:
        print(f"[info] YingMusic isolated wav count: {isolated_wav_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
