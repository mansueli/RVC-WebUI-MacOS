#!/usr/bin/env python3
"""Check and download required RVC assets for first-time setup.

Downloads directly from Hugging Face using requests — no binary downloader
needed.  This replaces the fragile rvcmd approach that required fetching a
Go binary from GitHub releases at runtime.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Hugging Face direct download URLs
# ---------------------------------------------------------------------------
HF_BASE = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main"

ASSETS: list[tuple[str, str]] = [
    # (local path relative to repo root,  HF path)
    ("assets/hubert/hubert_base.pt", "hubert_base.pt"),
    ("assets/rmvpe/rmvpe.pt", "rmvpe.pt"),
    # pretrained v1
    ("assets/pretrained/D32k.pth", "pretrained/D32k.pth"),
    ("assets/pretrained/D40k.pth", "pretrained/D40k.pth"),
    ("assets/pretrained/D48k.pth", "pretrained/D48k.pth"),
    ("assets/pretrained/G32k.pth", "pretrained/G32k.pth"),
    ("assets/pretrained/G40k.pth", "pretrained/G40k.pth"),
    ("assets/pretrained/G48k.pth", "pretrained/G48k.pth"),
    ("assets/pretrained/f0D32k.pth", "pretrained/f0D32k.pth"),
    ("assets/pretrained/f0D40k.pth", "pretrained/f0D40k.pth"),
    ("assets/pretrained/f0D48k.pth", "pretrained/f0D48k.pth"),
    ("assets/pretrained/f0G32k.pth", "pretrained/f0G32k.pth"),
    ("assets/pretrained/f0G40k.pth", "pretrained/f0G40k.pth"),
    ("assets/pretrained/f0G48k.pth", "pretrained/f0G48k.pth"),
    # pretrained v2
    ("assets/pretrained_v2/D32k.pth", "pretrained_v2/D32k.pth"),
    ("assets/pretrained_v2/D40k.pth", "pretrained_v2/D40k.pth"),
    ("assets/pretrained_v2/D48k.pth", "pretrained_v2/D48k.pth"),
    ("assets/pretrained_v2/G32k.pth", "pretrained_v2/G32k.pth"),
    ("assets/pretrained_v2/G40k.pth", "pretrained_v2/G40k.pth"),
    ("assets/pretrained_v2/G48k.pth", "pretrained_v2/G48k.pth"),
    ("assets/pretrained_v2/f0D32k.pth", "pretrained_v2/f0D32k.pth"),
    ("assets/pretrained_v2/f0D40k.pth", "pretrained_v2/f0D40k.pth"),
    ("assets/pretrained_v2/f0D48k.pth", "pretrained_v2/f0D48k.pth"),
    ("assets/pretrained_v2/f0G32k.pth", "pretrained_v2/f0G32k.pth"),
    ("assets/pretrained_v2/f0G40k.pth", "pretrained_v2/f0G40k.pth"),
    ("assets/pretrained_v2/f0G48k.pth", "pretrained_v2/f0G48k.pth"),
    # uvr5
    (
        "assets/uvr5_weights/HP2-人声vocals+非人声instrumentals.pth",
        "uvr5_weights/HP2-人声vocals+非人声instrumentals.pth",
    ),
    ("assets/uvr5_weights/HP2_all_vocals.pth", "uvr5_weights/HP2_all_vocals.pth"),
    ("assets/uvr5_weights/HP3_all_vocals.pth", "uvr5_weights/HP3_all_vocals.pth"),
    (
        "assets/uvr5_weights/HP5-主旋律人声vocals+其他instrumentals.pth",
        "uvr5_weights/HP5-主旋律人声vocals+其他instrumentals.pth",
    ),
    (
        "assets/uvr5_weights/HP5_only_main_vocal.pth",
        "uvr5_weights/HP5_only_main_vocal.pth",
    ),
    (
        "assets/uvr5_weights/VR-DeEchoAggressive.pth",
        "uvr5_weights/VR-DeEchoAggressive.pth",
    ),
    ("assets/uvr5_weights/VR-DeEchoDeReverb.pth", "uvr5_weights/VR-DeEchoDeReverb.pth"),
    ("assets/uvr5_weights/VR-DeEchoNormal.pth", "uvr5_weights/VR-DeEchoNormal.pth"),
    (
        "assets/uvr5_weights/onnx_dereverb_By_FoxJoy/vocals.onnx",
        "uvr5_weights/onnx_dereverb_By_FoxJoy/vocals.onnx",
    ),
]

# rmvpe.onnx is optional (Windows/Linux CPU only); skip it on macOS.
OPTIONAL_ASSETS: list[tuple[str, str]] = [
    ("assets/rmvpe/rmvpe.onnx", "rmvpe.onnx"),
]


# Characters that requests does not percent-encode by default in URL paths
# (notably '+' which HuggingFace treats literally, not as space).
def _hf_url(hf_path: str) -> str:
    from urllib.parse import quote

    return f"{HF_BASE}/{quote(hf_path, safe='/')}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download all required models/files if missing.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only validate required files; do not download.",
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Also download optional assets (e.g. rmvpe.onnx).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logs.",
    )
    return parser.parse_args()


def download_file(url: str, dest: Path, logger: logging.Logger) -> bool:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s", dest)
    try:
        with requests.get(url, stream=True, timeout=(10, 60)) as r:
            r.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                    f.write(chunk)
            tmp.rename(dest)
        return True
    except Exception as exc:
        logger.error("Failed to download %s: %s", url, exc)
        if dest.with_suffix(dest.suffix + ".tmp").exists():
            dest.with_suffix(dest.suffix + ".tmp").unlink()
        return False


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("download_models")

    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    load_dotenv(repo_root / ".env")
    load_dotenv(repo_root / "sha256.env")

    assets_to_process = list(ASSETS)
    if args.include_optional:
        assets_to_process += OPTIONAL_ASSETS

    missing = [
        (local, hf) for local, hf in assets_to_process if not Path(local).exists()
    ]

    if not missing:
        logger.info("All required assets are already present.")
        return 0

    logger.info("%d asset(s) missing.", len(missing))
    for local, _ in missing:
        logger.info("  missing: %s", local)

    if args.check_only:
        logger.error("Missing assets found. Re-run without --check-only to download.")
        return 1

    failed = []
    for local, hf_path in missing:
        url = _hf_url(hf_path)
        if not download_file(url, Path(local), logger):
            failed.append(local)

    if failed:
        logger.error("Failed to download %d file(s):", len(failed))
        for f in failed:
            logger.error("  %s", f)
        return 1

    logger.info("All required assets are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
