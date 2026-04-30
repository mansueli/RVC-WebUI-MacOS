import os, pathlib
import torch as _torch

# PyTorch 2.6 changed weights_only default to True, which breaks fairseq's
# checkpoint loading (it uses arbitrary Python objects like Dictionary).
# Patch torch.load to pass weights_only=False for .pt/.pth files.
_orig_torch_load = _torch.load
def _patched_torch_load(f, *args, **kwargs):
    if "weights_only" not in kwargs:
        path = f if isinstance(f, str) else getattr(f, "name", "")
        if isinstance(path, str) and path.endswith((".pt", ".pth")):
            kwargs["weights_only"] = False
    return _orig_torch_load(f, *args, **kwargs)
_torch.load = _patched_torch_load

from fairseq import checkpoint_utils


def get_index_path_from_model(sid):
    return next(
        (
            f
            for f in [
                str(pathlib.Path(root, name))
                for path in [os.getenv("outside_index_root"), os.getenv("index_root")]
                for root, _, files in os.walk(path, topdown=False)
                for name in files
                if name.endswith(".index") and "trained" not in name
            ]
            if sid.split(".")[0] in f
        ),
        "",
    )


def load_hubert(device, is_half):
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["assets/hubert/hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(device)
    if is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()
