from typing import Dict, Mapping

import torch
import torch.nn as nn

from .e2e import E2E


def _strip_common_prefixes(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    prefixes = ("module.", "model.")
    for prefix in prefixes:
        if any(k.startswith(prefix) for k in state_dict):
            state_dict = {
                (k[len(prefix) :] if k.startswith(prefix) else k): v
                for k, v in state_dict.items()
            }
    return state_dict


def _extract_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        candidate_keys = (
            "state_dict",
            "model",
            "weight",
            "weights",
            "model_state_dict",
        )
        for key in candidate_keys:
            candidate = checkpoint.get(key)
            if isinstance(candidate, Mapping) and any(
                torch.is_tensor(v) for v in candidate.values()
            ):
                return _strip_common_prefixes(dict(candidate))

        if any(torch.is_tensor(v) for v in checkpoint.values()):
            return _strip_common_prefixes(
                {k: v for k, v in checkpoint.items() if torch.is_tensor(v)}
            )

    raise RuntimeError("Unsupported RMVPE checkpoint format")


def get_rmvpe(model_path, device, is_half=True):
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = _extract_state_dict(checkpoint)

        model = E2E(4, 1, (2, 2))
        load_result = model.load_state_dict(state_dict, strict=False)

        loaded_keys = set(state_dict.keys()) - set(load_result.unexpected_keys)
        if len(loaded_keys) == 0:
            # Backward compatibility for legacy, lightweight custom checkpoints.
            legacy_model = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 1, kernel_size=3, padding=1),
            )
            legacy_model.load_state_dict(state_dict)
            model = legacy_model

        model.eval()
        if is_half:
            model = model.half()
        model = model.to(device)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load RMVPE model: {str(e)}")
