#!/usr/bin/env python3
"""Experimental HQ-SVC native inference launcher.

Runs HQ-SVC native inference scripts with tolerant argument patterns and writes
an output wav path for WebUI integration.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path


INFER_SCRIPT_CANDIDATES = [
    "my_inference.py",
    "inference.py",
    "infer.py",
    "scripts/inference.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch HQ-SVC native inference")
    parser.add_argument("--repo-dir", default="external/HQ-SVC")
    parser.add_argument("--python-bin", default="")
    parser.add_argument("--infer-entry", default="auto")
    parser.add_argument("--source", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", default="")
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--fp16", default="True")
    parser.add_argument("--config", default="")
    parser.add_argument("--expname", default="hqsvc_native_v3")
    return parser.parse_args()


def resolve_script(repo_dir: Path, entry: str, candidates: list[str]) -> Path | None:
    if entry != "auto":
        p = Path(entry)
        if not p.is_absolute():
            p = (repo_dir / p).resolve()
        return p if p.exists() else None
    for candidate in candidates:
        p = repo_dir / candidate
        if p.exists():
            return p
    return None


def resolve_python(repo_dir: Path, explicit_python: str) -> Path:
    if explicit_python:
        return Path(explicit_python)
    venv_python = repo_dir / "venv" / "bin" / "python"
    if venv_python.exists():
        return venv_python
    return Path(sys.executable)


def find_latest_wav(repo_dir: Path, start_ts: float) -> Path | None:
    outputs = repo_dir / "outputs"
    if not outputs.exists():
        return None
    newest: tuple[float, Path] | None = None
    for wav in outputs.rglob("*.wav"):
        mtime = wav.stat().st_mtime
        if mtime >= start_ts and (newest is None or mtime > newest[0]):
            newest = (mtime, wav)
    return newest[1] if newest else None


def candidate_cmds(py: Path, script: Path, args: argparse.Namespace) -> list[list[str]]:
    source = str(Path(args.source).resolve())
    ckpt = str(Path(args.checkpoint).resolve())
    output = str(Path(args.output).resolve())
    target = str(Path(args.target).resolve()) if args.target else ""

    commands = [
        [str(py), str(script), "--source", source, "--checkpoint", ckpt, "--output", output],
        [str(py), str(script), "--input", source, "--checkpoint", ckpt, "--output", output],
        [str(py), str(script), "--source", source, "--model", ckpt, "--output", output],
        [str(py), str(script), "--input", source, "--model", ckpt, "--output", output],
    ]

    if target:
        commands.insert(
            0,
            [str(py), str(script), "--source", source, "--target", target, "--checkpoint", ckpt, "--output", output],
        )

    if args.config:
        for cmd in commands:
            cmd.extend(["--config", str(Path(args.config).resolve())])
    for cmd in commands:
        cmd.extend(["--cuda", args.cuda, "--fp16", args.fp16, "--expname", args.expname])

    return commands


def main() -> int:
    args = parse_args()
    repo_dir = Path(args.repo_dir).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if not repo_dir.exists():
        print("[error] HQ-SVC repo not found: %s" % repo_dir)
        return 2

    script = resolve_script(repo_dir, args.infer_entry, INFER_SCRIPT_CANDIDATES)
    if script is None:
        print("[error] No HQ-SVC inference entrypoint found in %s" % repo_dir)
        print("[hint] Looked for: %s" % ", ".join(INFER_SCRIPT_CANDIDATES))
        print("[hint] Or pass --infer-entry <relative/path/to/inference_script.py>")
        return 2

    python_bin = resolve_python(repo_dir, args.python_bin)
    if not python_bin.exists():
        print("[error] Python executable not found: %s" % python_bin)
        return 2

    start_ts = time.time()
    failures: list[tuple[list[str], int]] = []
    for cmd in candidate_cmds(python_bin, script, args):
        print("[cmd] %s" % " ".join(cmd))
        rc = subprocess.run(cmd, cwd=str(repo_dir)).returncode
        if rc == 0:
            if output.exists() and output.stat().st_size > 0:
                print("[done] HQ-SVC inference output: %s" % output)
                return 0
            break
        failures.append((cmd, rc))

    newest = find_latest_wav(repo_dir, start_ts)
    if newest is not None and newest.exists() and newest.stat().st_size > 0:
        shutil.copyfile(newest, output)
        print("[done] HQ-SVC inference output (detected): %s" % output)
        print("[info] Source generated file: %s" % newest)
        return 0

    print("[error] HQ-SVC inference failed or output file was not produced.")
    for cmd, rc in failures:
        print("[fail] rc=%s cmd=%s" % (rc, " ".join(cmd)))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
