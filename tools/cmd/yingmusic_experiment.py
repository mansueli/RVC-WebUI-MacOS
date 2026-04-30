#!/usr/bin/env python3
"""Set up and run a pretrained YingMusic-SVC experiment.

This wrapper keeps YingMusic isolated from the main repo environment:
- clones upstream repo into external/
- creates a dedicated virtualenv
- installs upstream dependencies
- downloads official pretrained checkpoint
- runs official my_inference.py with user-provided source/target

Notes:
- Upstream CLI currently hardcodes CUDA device creation, so inference requires
  an NVIDIA CUDA environment.
- On macOS, use --setup-only here and run inference on a CUDA machine.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


DEFAULT_REPO_URL = "https://github.com/GiantAILab/YingMusic-SVC.git"
DEFAULT_CHECKPOINT_URL = (
    "https://huggingface.co/GiantAILab/YingMusic-SVC/resolve/main/YingMusic-SVC-full.pt"
)
DEFAULT_SEPARATOR_URL = (
    "https://huggingface.co/GiantAILab/YingMusic-SVC/resolve/main/bs_roformer.ckpt"
)


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def find_python(preferred: str) -> str:
    if shutil.which(preferred):
        return preferred
    if shutil.which("python3.10"):
        return "python3.10"
    if shutil.which("python3"):
        return "python3"
    return sys.executable


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


def ensure_venv(venv_dir: Path, python_cmd: str) -> Path:
    py = venv_dir / "bin" / "python"
    if py.exists():
        print(f"[info] Using existing venv at {venv_dir}")
        return py
    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    run([python_cmd, "-m", "venv", str(venv_dir)])
    return py


def install_deps(venv_python: Path, repo_dir: Path) -> None:
    run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    run([str(venv_python), "-m", "pip", "install", "-r", "requirements.txt"], cwd=repo_dir)


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[info] Found {dest}")
        return
    print(f"[download] {url} -> {dest}")
    urllib.request.urlretrieve(url, str(dest))


def cuda_available() -> bool:
    return shutil.which("nvidia-smi") is not None


def check_external_binaries() -> None:
    missing = [name for name in ("ffmpeg", "sox") if shutil.which(name) is None]
    if missing:
        print("[warn] Missing external tools:", ", ".join(missing))
        print("[warn] YingMusic recommends both ffmpeg and sox.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pretrained YingMusic-SVC experiment")
    parser.add_argument("--source", type=str, help="Path to source vocal wav")
    parser.add_argument("--target", type=str, help="Path to target timbre wav")
    parser.add_argument("--accompany", type=str, default="", help="Optional accompany wav path")

    parser.add_argument("--repo-url", type=str, default=DEFAULT_REPO_URL)
    parser.add_argument("--repo-ref", type=str, default="main")
    parser.add_argument("--repo-dir", type=str, default="external/YingMusic-SVC")
    parser.add_argument("--venv-dir", type=str, default="external/.venv-yingmusic")
    parser.add_argument("--python", type=str, default="python3.10")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="external/YingMusic-SVC/checkpoints/YingMusic-SVC-full.pt",
    )
    parser.add_argument("--checkpoint-url", type=str, default=DEFAULT_CHECKPOINT_URL)
    parser.add_argument(
        "--separator-ckpt",
        type=str,
        default="external/YingMusic-SVC/checkpoints/bs_roformer.ckpt",
    )
    parser.add_argument("--separator-url", type=str, default=DEFAULT_SEPARATOR_URL)

    parser.add_argument(
        "--config",
        type=str,
        default="external/YingMusic-SVC/configs/YingMusic-SVC.yml",
    )
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--fp16", type=str, default="True", choices=["True", "False"])
    parser.add_argument("--expname", type=str, default="rvc_yingmusic_pretrained")

    parser.add_argument("--setup-only", action="store_true", help="Only setup repo/env/models")
    parser.add_argument("--skip-install", action="store_true", help="Skip pip install")

    # WebUI batch isolation mode
    parser.add_argument(
        "--source-dir",
        type=str,
        default="",
        help="Directory of source audio files to isolate vocals from (batch mode).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory where isolated vocals will be written (batch mode).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    venv_dir = Path(args.venv_dir).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    separator_ckpt = Path(args.separator_ckpt).resolve()
    config = Path(args.config).resolve()

    python_cmd = find_python(args.python)
    print(f"[info] Python command: {python_cmd}")

    ensure_repo(repo_dir, args.repo_url, args.repo_ref)
    venv_python = ensure_venv(venv_dir, python_cmd)

    if not args.skip_install:
        install_deps(venv_python, repo_dir)

    download_file(args.checkpoint_url, checkpoint)
    download_file(args.separator_url, separator_ckpt)
    check_external_binaries()

    if args.setup_only:
        if args.source_dir:
            print("[info] Source directory: %s" % args.source_dir)
        if args.output_dir:
            print("[info] Isolated vocals output directory: %s" % args.output_dir)
        print("[done] Setup completed. Use a CUDA machine to run inference.")
        return 0

    # Batch isolation mode: source-dir -> output-dir (requires CUDA)
    if args.source_dir:
        if not cuda_available():
            print("[error] Batch isolation requires CUDA (nvidia-smi not found).")
            print("[hint] Run on a CUDA host, or use --setup-only on macOS.")
            print("[info] Source dir: %s" % args.source_dir)
            if args.output_dir:
                print("[info] Intended output dir: %s" % args.output_dir)
            return 2
        output_dir = Path(args.output_dir).resolve() if args.output_dir else None
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        source_dir = Path(args.source_dir).resolve()
        audio_exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
        audio_files = [f for f in source_dir.iterdir() if f.suffix.lower() in audio_exts]
        if not audio_files:
            print("[error] No audio files found in source-dir: %s" % source_dir)
            return 1
        print("[info] Found %d audio files to isolate." % len(audio_files))
        for af in sorted(audio_files):
            out_path = (output_dir / af.name) if output_dir else None
            print("[isolate] %s" % af.name)
            iso_cmd = [
                str(venv_python),
                "my_inference.py",
                "--source", str(af),
                "--diffusion-steps", str(args.diffusion_steps),
                "--checkpoint", str(checkpoint),
                "--expname", args.expname,
                "--cuda", args.cuda,
                "--fp16", args.fp16,
                "--config", str(config),
            ]
            if out_path:
                iso_cmd += ["--output", str(out_path)]
            run(iso_cmd, cwd=repo_dir)
        if output_dir:
            print("[done] Isolated vocals written to: %s" % output_dir)
        return 0

    if not args.source or not args.target:
        print("[error] --source and --target (or --source-dir) are required unless --setup-only is used.")
        return 2

    if not cuda_available():
        print("[error] CUDA runtime not detected (nvidia-smi missing).")
        print("[hint] Run with --setup-only on macOS, then execute inference on a CUDA host.")
        return 2

    source = str(Path(args.source).resolve())
    target = str(Path(args.target).resolve())
    accompany = str(Path(args.accompany).resolve()) if args.accompany else ""

    cmd = [
        str(venv_python),
        "my_inference.py",
        "--source",
        source,
        "--target",
        target,
        "--diffusion-steps",
        str(args.diffusion_steps),
        "--checkpoint",
        str(checkpoint),
        "--expname",
        args.expname,
        "--cuda",
        args.cuda,
        "--fp16",
        args.fp16,
        "--config",
        str(config),
    ]
    if accompany:
        cmd.extend(["--accompany", accompany])

    run(cmd, cwd=repo_dir)

    out_dir = repo_dir / "outputs" / args.expname
    print(f"[done] YingMusic output directory: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
