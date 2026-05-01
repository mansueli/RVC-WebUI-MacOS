#!/usr/bin/env python3
"""Local bridge for RVC discriminator components used by V3 Stage 2 training."""

from __future__ import annotations

from infer.lib.train.losses import discriminator_loss, feature_loss, generator_loss
from rvc.layers.discriminators import MultiPeriodDiscriminator

__all__ = [
    "MultiPeriodDiscriminator",
    "discriminator_loss",
    "feature_loss",
    "generator_loss",
]
