#!/usr/bin/env python3
"""Set up and run a pretrained HQ-SVC experiment.

This script is inference-only because upstream HQ-SVC training code is not
publicly released yet. It keeps HQ-SVC isolated from this repository runtime.

Workflow:
1) Clone HQ-SVC into external/
2) Download published environment tarball
3) Extract environment under external/HQ-SVC/venv
4) Optionally run Gradio inference app (Linux + CUDA only)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


DEFAULT_REPO_URL = "https://github.com/ShawnPi233/HQ-SVC.git"
DEFAULT_ENV_URL = "https://huggingface.co/shawnpi/HQ-SVC/resolve/main/environment.tar.gz"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def ensure_repo(repo_dir: Path, repo_url: str, ref: str) -> None:
    if (repo_dir / ".git").exists():
        print(f"[info] Using existing repo at {repo_dir}")
        run(["git", "fetch", "--all", "--tags"], cwd=repo_dir)
        run(["git", "checkout", ref], cwd=repo_dir)
        run(["git", "pull", "--ff-only"], cwd=repo_dir)
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", repo_url, str(repo_dir)])
    run(["git", "checkout", ref], cwd=repo_dir)


def has_cuda() -> bool:
    return shutil.which("nvidia-smi") is not None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HQ-SVC pretrained experiment")
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--repo-ref", default="main")
    parser.add_argument("--repo-dir", default="external/HQ-SVC")
    parser.add_argument("--env-url", default=DEFAULT_ENV_URL)
    parser.add_argument("--env-tar", default="external/HQ-SVC/environment.tar.gz")
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    env_tar = Path(args.env_tar).resolve()
    venv_dir = repo_dir / "venv"

    ensure_repo(repo_dir, args.repo_url, args.repo_ref)

    if not args.skip_download:
        env_tar.parent.mkdir(parents=True, exist_ok=True)
        if env_tar.exists() and env_tar.stat().st_size > 0:
            print(f"[info] Found {env_tar}")
        else:
            run(["curl", "-L", args.env_url, "-o", str(env_tar)])

    if not args.skip_extract:
        venv_dir.mkdir(parents=True, exist_ok=True)
        run(["tar", "-xzf", str(env_tar), "-C", str(venv_dir)])

    print("[info] HQ-SVC setup finished (inference-only upstream at this time).")

    if args.setup_only:
        print("[done] Setup-only completed.")
        return 0

    if not has_cuda():
        print("[error] CUDA runtime not detected. HQ-SVC upstream is tested on Linux + CUDA >= 11.8.")
        print("[hint] Run with --setup-only here, then execute on a CUDA Linux host.")
        return 2

    activate = venv_dir / "bin" / "activate"
    if not activate.exists():
        print(f"[error] Missing activate script: {activate}")
        return 2

    cmd = f"source '{activate}' && unset LD_LIBRARY_PATH && python gradio_app.py"
    run(["bash", "-lc", cmd], cwd=repo_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
