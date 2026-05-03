#!/usr/bin/env python3
"""V3 training adapter — bridges WebUI training dispatch to HQ-SVC-oriented backend.

This adapter normalises the WebUI training contract (experiment directory,
hyperparameters, status file) for V3 experiments.

On invocation it will:
1. Validate all V3-specific prerequisites (48k sample rate, prepared dataset,
   feature files).
2. Write a JSON status file compatible with the existing WebUI training supervisor
   (same schema as train_supervisor.py).
3. If the HQ-SVC training environment is available (external/HQ-SVC/ with
   training code and a prepared Python environment), delegate to it.
4. Otherwise, emit actionable setup instructions and exit cleanly so the WebUI
   can surface them as a normal status update.

Exit codes:
  0 — setup-only mode completed or training started/dispatched
  1 — prerequisite failure (missing dataset, wrong sample rate, etc.)
  2 — training runtime error after prerequisites passed

Backends:
    external — use external/HQ-SVC training scripts
    full_paper_mode — use in-repo paper-aligned HQ-SVC scaffold
    local_experimental — use in-repo experimental trainer (CPU/GPU)
    auto — prefer external when available, otherwise full_paper_mode

Stages:
    1 — paper-mode training only
    2 — RVC discriminator fine-tuning only (full_paper_mode only)
    both — run stage 1 followed by stage 2 (full_paper_mode only)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# Repo root is three levels up from this script: tools/cmd/hqsvc_train_adapter.py
NOW_DIR = Path(__file__).resolve().parent.parent.parent


def _write_status(
    status_file: Path,
    state: str,
    message: str,
    running: bool = False,
    **extra,
) -> None:
    status = {
        "state": state,
        "running": running,
        "message": message,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "supervisor_pid": os.getpid(),
        "child_pid": 0,
        "attempt": 1,
        "max_retries": 0,
        "last_exit_code": None,
        "last_error_type": None,
        **extra,
    }
    status_file.parent.mkdir(parents=True, exist_ok=True)
    status_file.write_text(
        json.dumps(status, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )


def _validate_prerequisites(args: argparse.Namespace, exp_dir: Path) -> list[str]:
    errors: list[str] = []

    if args.sr != "48k":
        errors.append(
            "V3 requires 48k sample rate (got '%s'). Set SR to 48k and re-run." % args.sr
        )

    gt_wavs = exp_dir / "0_gt_wavs"
    if not gt_wavs.exists() or not any(gt_wavs.glob("*.wav")):
        errors.append(
            "V3 dataset wavs not found at '%s'. "
            "Run Step 1 (Process data) first." % gt_wavs
        )

    if args.backend == "external":
        feature_dir = exp_dir / "3_feature768"
        if not feature_dir.exists() or not any(feature_dir.glob("*.npy")):
            errors.append(
                "V3 feature files not found at '%s'. "
                "Run Step 2 (Feature extraction) with v2/v3 feature dim first." % feature_dir
            )

    return errors


def _check_hqsvc_env(repo_dir: Path) -> tuple[bool, str]:
    """Return (available, reason_if_not_available)."""
    if not repo_dir.exists():
        return False, (
            "HQ-SVC repository not cloned.\n"
            "Run: python tools/cmd/hqsvc_experiment.py --setup-only\n"
            "Then re-run V3 training."
        )

    train_candidates = [
        repo_dir / "train.py",
        repo_dir / "train_cli.py",
        repo_dir / "scripts" / "train.py",
    ]
    if not any(p.exists() for p in train_candidates):
        return False, (
            "HQ-SVC training code not found in '%s'.\n"
            "Expected one of: train.py, train_cli.py, scripts/train.py\n"
            "You can also pass --train-entry to point at a custom training script."
            % repo_dir
        )

    venv_python = repo_dir / "venv" / "bin" / "python"
    if not venv_python.exists():
        return False, (
            "HQ-SVC Python environment not found at '%s'.\n"
            "Run: python tools/cmd/hqsvc_experiment.py --setup-only\n"
            "Then re-run V3 training." % venv_python
        )

    return True, ""


def _stage_checkpoint_paths(exp_dir: Path) -> tuple[Path, Path]:
    model_dir = exp_dir / "hqsvc_full"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / "G_stage1_latest.pt", model_dir / "G_stage2_latest.pt"


def _resolve_stage2_init(exp_dir: Path, init_checkpoint: str) -> Path | None:
    if init_checkpoint:
        path = Path(init_checkpoint).resolve()
        return path if path.exists() else None
    stage1_path, _ = _stage_checkpoint_paths(exp_dir)
    if stage1_path.exists():
        return stage1_path
    fallback = exp_dir / "hqsvc_full" / "G_latest.pt"
    if fallback.exists():
        return fallback
    matches = sorted((exp_dir / "hqsvc_full").glob("G_*.pt"))
    return matches[-1] if matches else None


def _run_backend(cmd: list[str], cwd: Path, log_handle, status_file: Path, stage_label: str) -> int:
    _write_status(status_file, "running", "V3 %s started" % stage_label, running=True)
    return subprocess.run(cmd, cwd=str(cwd), stdout=log_handle, stderr=subprocess.STDOUT).returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V3 HQ-SVC training adapter for WebUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--exp-dir",
        required=True,
        help="Experiment name (subdirectory under logs/)",
    )
    parser.add_argument(
        "--sr",
        default="48k",
        choices=["48k"],
        help="Sample rate. V3 only supports 48k.",
    )
    parser.add_argument(
        "--f0",
        type=int,
        default=1,
        choices=[0, 1],
        help="Enable F0 pitch guidance (1=yes, 0=no).",
    )
    parser.add_argument("--total-epoch", type=int, default=600)
    parser.add_argument("--save-epoch", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--save-every-weights",
        type=int,
        default=1,
        choices=[0, 1],
        help="Save a small final model at each save point (1=yes).",
    )
    parser.add_argument("--author", default="")
    parser.add_argument(
        "--gpus",
        default="",
        help="GPU indices separated by '-', e.g., 0-1 (empty = CPU/MPS).",
    )
    parser.add_argument(
        "--repo-dir",
        default="external/HQ-SVC",
        help="Path to cloned HQ-SVC repository (absolute or relative to repo root).",
    )
    parser.add_argument(
        "--status-file",
        default="",
        help="Path to training_status.json. Defaults to logs/<exp-dir>/training_status.json.",
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Validate prerequisites and environment, then exit without training.",
    )
    parser.add_argument(
        "--train-entry",
        default="auto",
        help="HQ-SVC training entry script path (relative to repo-dir) or 'auto'.",
    )
    parser.add_argument(
        "--backend",
        default="full_paper_mode",
        choices=["auto", "external", "full_paper_mode", "local_experimental"],
        help="V3 training backend selection.",
    )
    parser.add_argument(
        "--dataset-dir",
        default="",
        help="Optional dataset wav directory for local_experimental backend.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=6000,
        help="Training steps for local_experimental/full_paper_mode backends.",
    )
    parser.add_argument(
        "--stage2-steps",
        type=int,
        default=0,
        help="Optional explicit step count for Stage 2 (full_paper_mode). 0 = auto from --steps.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Execution device for local_experimental backend.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0,
        help="Override learning rate for local_experimental/full_paper_mode backends. 0 = use script default.",
    )
    parser.add_argument(
        "--stage",
        default="1",
        choices=["1", "2", "both"],
        help="Training stage to run for V3 full-paper backend.",
    )
    parser.add_argument(
        "--init-checkpoint",
        default="",
        help="Optional checkpoint path used to initialize Stage 2 fine-tuning.",
    )
    parser.add_argument("--smart-save", default="on", choices=["on", "off"])
    parser.add_argument("--smart-save-window", type=int, default=10)
    parser.add_argument("--smart-save-min-improve", type=float, default=2.0)
    parser.add_argument("--smart-save-max-mel", type=float, default=16.0)
    parser.add_argument("--smart-save-cooldown", type=int, default=5)
    parser.add_argument("--smart-save-min-step", type=int, default=10)
    parser.add_argument(
        "--stage2-rmvpe-frame-loss",
        default="auto",
        choices=["off", "on", "auto"],
        help="Optional Stage 2 frame-level RMVPE pitch loss.",
    )
    parser.add_argument("--stage2-rmvpe-frame-loss-weight", type=float, default=0.05)
    parser.add_argument("--stage2-rmvpe-hop", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    exp_dir = NOW_DIR / "logs" / args.exp_dir
    exp_dir.mkdir(parents=True, exist_ok=True)

    status_file = (
        Path(args.status_file)
        if args.status_file
        else exp_dir / "training_status.json"
    )
    log_file = exp_dir / "train_webui.log"

    repo_dir_raw = Path(args.repo_dir)
    repo_dir = (
        repo_dir_raw if repo_dir_raw.is_absolute() else (NOW_DIR / repo_dir_raw).resolve()
    )

    _log = log_file.open("a", encoding="utf-8")

    def _print(msg: str) -> None:
        line = "%s | v3-adapter | %s" % (datetime.now().isoformat(timespec="seconds"), msg)
        print(line)
        _log.write(line + "\n")
        _log.flush()

    _print("Experiment: %s" % args.exp_dir)
    _print("Exp dir: %s" % exp_dir)
    _print("Sample rate: %s" % args.sr)
    _print("F0: %s" % args.f0)
    _print("Epochs: %s / save every %s" % (args.total_epoch, args.save_epoch))
    _print("Batch size: %s" % args.batch_size)
    _print("Backend: %s" % args.backend)
    _print("Stage: %s" % args.stage)

    # --- Prerequisite validation ---
    errors = _validate_prerequisites(args, exp_dir)
    if errors:
        msg = "V3 prerequisite check failed:\n" + "\n".join("  - " + e for e in errors)
        _print(msg)
        _write_status(status_file, "failed", msg)
        _log.close()
        return 1

    _print("Prerequisites OK")

    # --- HQ-SVC environment check ---
    available, reason = _check_hqsvc_env(repo_dir)
    if args.backend == "external" and not available:
        msg = "V3 backend is set to external, but external HQ-SVC is unavailable.\n\n" + reason
        _print(msg)
        _write_status(status_file, "failed", msg)
        _log.close()
        return 2

    selected_backend = args.backend
    if selected_backend == "auto":
        selected_backend = "external" if available else "full_paper_mode"

    if selected_backend == "external" and not available:
        msg = (
            "V3 training environment not yet fully available.\n\n"
            + reason
            + "\n\n"
            "V3 dataset preparation and feature extraction are fully functional.\n"
            "Once HQ-SVC training code is available, re-run V3 training to start fine-tuning."
        )
        _print(msg)
        _write_status(status_file, "setup_required", msg)
        _log.close()
        return 0  # Not a hard error — informational setup-only exit

    if args.stage in {"2", "both"} and selected_backend != "full_paper_mode":
        msg = "RVC discriminator Stage 2 is only supported with backend 'full_paper_mode'."
        _print(msg)
        _write_status(status_file, "failed", msg)
        _log.close()
        return 2

    if args.setup_only:
        _print("V3 environment validated. Ready to train with backend: %s" % selected_backend)
        _write_status(
            status_file,
            "setup_complete",
            "V3 environment validated. Ready to train with backend: %s" % selected_backend,
        )
        _log.close()
        return 0

    # --- Invoke selected V3 training backend ---
    stage1_path, stage2_path = _stage_checkpoint_paths(exp_dir)
    if selected_backend == "external":
        venv_python = repo_dir / "venv" / "bin" / "python"
        train_launcher = NOW_DIR / "tools" / "cmd" / "hqsvc_native_train.py"
        cmd = [
            str(venv_python),
            str(train_launcher),
            "--repo-dir",
            str(repo_dir),
            "--exp-dir",
            str(exp_dir),
            "--sr",
            "48000",
            "--f0",
            str(args.f0),
            "--total-epoch",
            str(args.total_epoch),
            "--save-epoch",
            str(args.save_epoch),
            "--batch-size",
            str(args.batch_size),
            "--save-every-weights",
            str(args.save_every_weights),
            "--train-entry",
            str(args.train_entry),
        ]
        if args.author:
            cmd += ["--author", args.author]
        if args.gpus:
            cmd += ["--gpus", args.gpus]
        run_plan = [("external training", cmd, repo_dir)]
    elif selected_backend == "local_experimental":
        train_launcher = NOW_DIR / "tools" / "cmd" / "hqsvc_local_train.py"
        cmd = [
            str(sys.executable),
            str(train_launcher),
            "--exp-dir",
            args.exp_dir,
            "--sample-rate",
            "48000",
            "--batch-size",
            str(max(1, int(args.batch_size))),
            "--steps",
            str(max(50, int(args.steps))),
            "--author",
            str(args.author or ""),
            "--device",
            args.device,
        ]
        if args.dataset_dir:
            cmd += ["--dataset-dir", args.dataset_dir]
        if args.learning_rate > 0:
            cmd += ["--learning-rate", str(args.learning_rate)]
        cmd += [
            "--smart-save",
            args.smart_save,
            "--smart-save-window",
            str(max(1, int(args.smart_save_window))),
            "--smart-save-min-improve",
            str(float(args.smart_save_min_improve)),
            "--smart-save-max-mel",
            str(float(args.smart_save_max_mel)),
            "--smart-save-cooldown",
            str(max(0, int(args.smart_save_cooldown))),
            "--smart-save-min-step",
            str(max(1, int(args.smart_save_min_step))),
        ]
        run_plan = [("local experimental training", cmd, NOW_DIR)]
    else:
        train_launcher = NOW_DIR / "tools" / "cmd" / "hqsvc_full_train.py"
        stage1_steps = max(50, int(args.steps))
        if int(args.stage2_steps) > 0:
            stage2_steps = max(50, int(args.stage2_steps))
        else:
            stage2_steps = max(800, min(6000, max(1, int(args.steps)) // 3))
        stage2_lr = args.learning_rate if args.learning_rate > 0 else 5e-5
        stage2_lr = min(stage2_lr, 5e-5)
        stage1_cmd = [
            str(sys.executable),
            str(train_launcher),
            "--exp-dir",
            args.exp_dir,
            "--stage",
            "1",
            "--sample-rate",
            "48000",
            "--batch-size",
            str(max(1, int(args.batch_size))),
            "--steps",
            str(stage1_steps),
            "--author",
            str(args.author or ""),
            "--device",
            args.device,
            "--output-checkpoint",
            str(stage1_path),
        ]
        if args.dataset_dir:
            stage1_cmd += ["--dataset-dir", args.dataset_dir]
        if args.learning_rate > 0:
            stage1_cmd += ["--learning-rate", str(args.learning_rate)]
        stage1_cmd += [
            "--smart-save",
            args.smart_save,
            "--smart-save-window",
            str(max(1, int(args.smart_save_window))),
            "--smart-save-min-improve",
            str(float(args.smart_save_min_improve)),
            "--smart-save-max-mel",
            str(float(args.smart_save_max_mel)),
            "--smart-save-cooldown",
            str(max(0, int(args.smart_save_cooldown))),
            "--smart-save-min-step",
            str(max(1, int(args.smart_save_min_step))),
        ]

        stage2_init = _resolve_stage2_init(exp_dir, args.init_checkpoint)
        stage2_cmd = [
            str(sys.executable),
            str(train_launcher),
            "--exp-dir",
            args.exp_dir,
            "--stage",
            "2",
            "--sample-rate",
            "48000",
            "--batch-size",
            str(max(1, int(args.batch_size))),
            "--steps",
            str(stage2_steps),
            "--author",
            str(args.author or ""),
            "--device",
            args.device,
            "--output-checkpoint",
            str(stage2_path),
        ]
        if args.dataset_dir:
            stage2_cmd += ["--dataset-dir", args.dataset_dir]
        stage2_cmd += ["--learning-rate", str(stage2_lr)]
        stage2_cmd += [
            "--smart-save",
            args.smart_save,
            "--smart-save-window",
            str(max(1, int(args.smart_save_window))),
            "--smart-save-min-improve",
            str(float(args.smart_save_min_improve)),
            "--smart-save-max-mel",
            str(float(args.smart_save_max_mel)),
            "--smart-save-cooldown",
            str(max(0, int(args.smart_save_cooldown))),
            "--smart-save-min-step",
            str(max(1, int(args.smart_save_min_step))),
            "--stage2-rmvpe-frame-loss",
            args.stage2_rmvpe_frame_loss,
            "--stage2-rmvpe-frame-loss-weight",
            str(float(args.stage2_rmvpe_frame_loss_weight)),
            "--stage2-rmvpe-hop",
            str(max(64, int(args.stage2_rmvpe_hop))),
        ]
        if stage2_init is not None:
            stage2_cmd += ["--init-checkpoint", str(stage2_init)]

        if args.stage == "1":
            run_plan = [("paper Stage 1", stage1_cmd, NOW_DIR)]
        elif args.stage == "2":
            if stage2_init is None:
                msg = "Stage 2 requested, but no Stage 1 checkpoint was found. Run Stage 1 first or provide --init-checkpoint."
                _print(msg)
                _write_status(status_file, "failed", msg)
                _log.close()
                return 2
            run_plan = [("paper Stage 2", stage2_cmd, NOW_DIR)]
        else:
            run_plan = [("paper Stage 1", stage1_cmd, NOW_DIR), ("paper Stage 2", stage2_cmd, NOW_DIR)]

    for stage_label, cmd, cwd in run_plan:
        _print("Launching V3 %s (%s): %s" % (stage_label, selected_backend, " ".join(cmd)))
        rc = _run_backend(cmd, cwd, _log, status_file, stage_label)
        if rc != 0:
            _log.close()
            _write_status(
                status_file,
                "failed",
                "V3 %s exited with code %d. Check %s for details." % (stage_label, rc, log_file),
                last_exit_code=rc,
            )
            return 2

    _log.close()
    if args.stage == "both":
        msg = "V3 Stage 1 + Stage 2 completed successfully"
    elif args.stage == "2":
        msg = "V3 Stage 2 completed successfully"
    else:
        msg = "V3 Stage 1 completed successfully"
    _write_status(status_file, "complete", msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
