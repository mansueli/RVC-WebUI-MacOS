#!/usr/bin/env python3
"""Experimental end-to-end V3 happy path orchestrator.

Flow:
1) Optional YingMusic setup / isolation
2) HQ-SVC native training launcher
3) HQ-SVC native inference launcher
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def run(cmd: list[str]) -> int:
    print("[cmd] %s" % " ".join(cmd))
    return subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experimental V3 happy path")
    parser.add_argument("--exp-dir", required=True)
    parser.add_argument("--repo-dir", default="external/HQ-SVC")

    parser.add_argument("--source-dir", default="")
    parser.add_argument("--isolated-output-dir", default="")
    parser.add_argument("--skip-preprocess", action="store_true")

    parser.add_argument("--sr", default="48k", choices=["48k"])
    parser.add_argument("--f0", type=int, default=1, choices=[0, 1])
    parser.add_argument("--total-epoch", type=int, default=20)
    parser.add_argument("--save-epoch", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--save-every-weights", type=int, default=1, choices=[0, 1])
    parser.add_argument("--author", default="")
    parser.add_argument("--gpus", default="")

    parser.add_argument("--infer-source", default="")
    parser.add_argument("--infer-checkpoint", default="")
    parser.add_argument("--infer-output", default="")
    parser.add_argument("--infer-target", default="")
    parser.add_argument("--skip-infer", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    python_cmd = sys.executable

    if not args.skip_preprocess:
        preprocess_cmd = [
            python_cmd,
            "tools/cmd/yingmusic_experiment.py",
        ]
        if args.source_dir:
            preprocess_cmd.extend(["--source-dir", args.source_dir])
            if args.isolated_output_dir:
                preprocess_cmd.extend(["--output-dir", args.isolated_output_dir])
        else:
            preprocess_cmd.append("--setup-only")

        rc = run(preprocess_cmd)
        if rc != 0:
            print("[error] Preprocess step failed.")
            return rc

    train_cmd = [
        python_cmd,
        "tools/cmd/hqsvc_train_adapter.py",
        "--exp-dir",
        args.exp_dir,
        "--sr",
        args.sr,
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
        "--repo-dir",
        args.repo_dir,
    ]
    if args.author:
        train_cmd += ["--author", args.author]
    if args.gpus:
        train_cmd += ["--gpus", args.gpus]

    rc = run(train_cmd)
    if rc != 0:
        print("[error] Training step failed.")
        return rc

    if args.skip_infer:
        print("[done] Preprocess + train completed. Inference skipped by request.")
        return 0

    if not args.infer_source:
        print("[error] --infer-source is required unless --skip-infer is set.")
        return 2
    if not args.infer_checkpoint:
        print("[error] --infer-checkpoint is required unless --skip-infer is set.")
        return 2

    infer_output = args.infer_output or (str(REPO_ROOT / "logs" / args.exp_dir / "hqsvc_native_infer.wav"))
    infer_cmd = [
        python_cmd,
        "tools/cmd/hqsvc_native_infer.py",
        "--repo-dir",
        args.repo_dir,
        "--source",
        args.infer_source,
        "--checkpoint",
        args.infer_checkpoint,
        "--output",
        infer_output,
    ]
    if args.infer_target:
        infer_cmd += ["--target", args.infer_target]

    rc = run(infer_cmd)
    if rc != 0:
        print("[error] Inference step failed.")
        return rc

    print("[done] Experimental V3 happy path completed.")
    print("[info] Inference output: %s" % infer_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
