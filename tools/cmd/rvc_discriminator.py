#!/usr/bin/env python3
"""Local bridge for RVC discriminator components used by V3 Stage 2 training."""

from __future__ import annotations

import importlib.util
from pathlib import Path

try:
    from infer.lib.train.losses import discriminator_loss, feature_loss, generator_loss
except Exception:
    _repo_root = Path(__file__).resolve().parent.parent.parent
    _losses_path = _repo_root / "infer" / "lib" / "train" / "losses.py"
    _spec = importlib.util.spec_from_file_location("_rvc_train_losses", str(_losses_path))
    if _spec is None or _spec.loader is None:
        raise ImportError("Unable to load loss functions from %s" % _losses_path)
    _losses = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_losses)
    discriminator_loss = _losses.discriminator_loss
    feature_loss = _losses.feature_loss
    generator_loss = _losses.generator_loss

from rvc.layers.discriminators import MultiPeriodDiscriminator

__all__ = [
    "MultiPeriodDiscriminator",
    "discriminator_loss",
    "feature_loss",
    "generator_loss",
]
