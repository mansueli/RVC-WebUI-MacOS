#!/usr/bin/env python3
"""Experimental HQ-SVC native training launcher.

This wrapper tries to run upstream HQ-SVC training code with a tolerant set of
entrypoint and argument conventions so this repo can execute an end-to-end
experimental V3 workflow.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


TRAIN_SCRIPT_CANDIDATES = [
    "train.py",
    "train_cli.py",
    "scripts/train.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch HQ-SVC native training")
    parser.add_argument("--repo-dir", default="external/HQ-SVC")
    parser.add_argument("--python-bin", default="")
    parser.add_argument("--exp-dir", required=True)
    parser.add_argument("--sr", default="48000")
    parser.add_argument("--f0", default="1")
    parser.add_argument("--total-epoch", default="600")
    parser.add_argument("--save-epoch", default="20")
    parser.add_argument("--batch-size", default="4")
    parser.add_argument("--save-every-weights", default="1")
    parser.add_argument("--author", default="")
    parser.add_argument("--gpus", default="")
    parser.add_argument("--train-entry", default="auto")
    return parser.parse_args()


def resolve_train_script(repo_dir: Path, train_entry: str) -> Path | None:
    if train_entry != "auto":
        path = Path(train_entry)
        if not path.is_absolute():
            path = (repo_dir / path).resolve()
        return path if path.exists() else None

    for candidate in TRAIN_SCRIPT_CANDIDATES:
        path = repo_dir / candidate
        if path.exists():
            return path
    return None


def resolve_python(repo_dir: Path, explicit_python: str) -> Path:
    if explicit_python:
        return Path(explicit_python)
    venv_python = repo_dir / "venv" / "bin" / "python"
    if venv_python.exists():
        return venv_python
    return Path(sys.executable)


def build_candidate_commands(python_bin: Path, train_script: Path, args: argparse.Namespace) -> list[list[str]]:
    base = [str(python_bin), str(train_script)]

    cmd1 = base + [
        "--exp-dir",
        args.exp_dir,
        "--sr",
        str(args.sr),
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
    ]
    if args.author:
        cmd1 += ["--author", args.author]
    if args.gpus:
        cmd1 += ["--gpus", args.gpus]

    cmd2 = base + [
        "--exp_name",
        Path(args.exp_dir).name,
        "--sample_rate",
        str(args.sr),
        "--f0",
        str(args.f0),
        "--epochs",
        str(args.total_epoch),
        "--save_interval",
        str(args.save_epoch),
        "--batch_size",
        str(args.batch_size),
    ]

    cmd3 = base + [
        "--exp-dir",
        args.exp_dir,
    ]

    cmd4 = base + [
        "--exp_name",
        Path(args.exp_dir).name,
    ]

    return [cmd1, cmd2, cmd3, cmd4]


def main() -> int:
    args = parse_args()
    repo_dir = Path(args.repo_dir).resolve()

    if not repo_dir.exists():
        print("[error] HQ-SVC repo not found: %s" % repo_dir)
        return 2

    train_script = resolve_train_script(repo_dir, args.train_entry)
    if train_script is None:
        print("[error] No HQ-SVC training entrypoint found in %s" % repo_dir)
        print("[hint] Looked for: %s" % ", ".join(TRAIN_SCRIPT_CANDIDATES))
        print("[hint] Or pass --train-entry <relative/path/to/train_script.py>")
        return 2

    python_bin = resolve_python(repo_dir, args.python_bin)
    if not python_bin.exists():
        print("[error] Python executable not found: %s" % python_bin)
        return 2

    print("[info] HQ-SVC repo: %s" % repo_dir)
    print("[info] Training entry: %s" % train_script)
    print("[info] Python: %s" % python_bin)

    failures: list[tuple[list[str], int]] = []
    for cmd in build_candidate_commands(python_bin, train_script, args):
        print("[cmd] %s" % " ".join(cmd))
        rc = subprocess.run(cmd, cwd=str(repo_dir)).returncode
        if rc == 0:
            return 0
        failures.append((cmd, rc))

    print("[error] HQ-SVC training launcher failed for all command variants.")
    for cmd, rc in failures:
        print("[fail] rc=%s cmd=%s" % (rc, " ".join(cmd)))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
