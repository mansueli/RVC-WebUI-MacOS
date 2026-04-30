import os
import sys
import traceback
import logging

now_dir = os.getcwd()
sys.path.append(now_dir)

from infer.lib.audio import load_audio

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.pyplot").setLevel(logging.WARNING)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

device = sys.argv[1]
n_part = int(sys.argv[2])
i_part = int(sys.argv[3])
if len(sys.argv) == 7:
    exp_dir = sys.argv[4]
    version = sys.argv[5]
    is_half = sys.argv[6].lower() == "true"
else:
    i_gpu = sys.argv[4]
    exp_dir = sys.argv[5]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(i_gpu)
    version = sys.argv[6]
    is_half = sys.argv[7].lower() == "true"
import fairseq
import numpy as np
import torch
import torch.nn.functional as F

# PyTorch >=2.6 changed torch.load default to weights_only=True which breaks
# fairseq checkpoint loading (it uses custom classes like Dictionary).
# Patch torch.load to allow weights_only=False for trusted local checkpoint
# files only; all other calls fall through to the default.
_torch_load_orig = torch.load

def _torch_load_compat(f, map_location=None, pickle_module=None, weights_only=None, **kw):
    _path = str(f) if not hasattr(f, 'read') else getattr(f, 'name', '')
    if weights_only is None and _path.endswith('.pt'):
        weights_only = False
    if weights_only is not None:
        return _torch_load_orig(f, map_location=map_location, weights_only=weights_only, **kw)
    return _torch_load_orig(f, map_location=map_location, **kw)

torch.load = _torch_load_compat

if "privateuseone" not in device:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
else:
    import torch_directml

    device = torch_directml.device(torch_directml.default_device())

    def forward_dml(ctx, x, scale):
        ctx.scale = scale
        res = x.clone().detach()
        return res

    fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml

f = open("%s/extract_f0_feature.log" % exp_dir, "a+")


def printt(strr):
    print(strr)
    f.write("%s\n" % strr)
    f.flush()


printt(" ".join(sys.argv))
model_path = "assets/hubert/hubert_base.pt"

printt("exp_dir: " + exp_dir)
wavPath = "%s/1_16k_wavs" % exp_dir
outPath = (
    "%s/3_feature256" % exp_dir if version == "v1" else "%s/3_feature768" % exp_dir
)
os.makedirs(outPath, exist_ok=True)


# wave must be 16k, hop_size=320
def readwave(wav_path, normalize=False):
    wav, sr = load_audio(wav_path)
    assert sr == 16000
    feats = torch.from_numpy(wav).float()
    assert feats.dim() == 1, feats.dim()
    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    feats = feats.view(1, -1)
    return feats


# HuBERT model
printt("load model(s) from {}".format(model_path))
# if hubert model is exist
if os.access(model_path, os.F_OK) == False:
    printt(
        "Error: Extracting is shut down because %s does not exist, you may download it from https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main"
        % model_path
    )
    exit(0)
models, saved_cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
    [model_path],
    suffix="",
)
model = models[0]
try:
    model = model.to(device)
    printt("move model to %s" % device)
except Exception:
    printt(traceback.format_exc())
    if str(device) != "cpu":
        printt("MPS/CUDA HuBERT move failed, fallback to CPU for feature extraction")
        device = "cpu"
        model = model.to(device)
        printt("move model to %s" % device)
    else:
        raise
if is_half:
    if device not in ["mps", "cpu"]:
        model = model.half()
model.eval()

todo = sorted(list(os.listdir(wavPath)))[i_part::n_part]
n = max(1, len(todo) // 10)  # 最多打印十条
if len(todo) == 0:
    printt("no-feature-todo")
else:
    printt("all-feature-%s" % len(todo))
    for idx, file in enumerate(todo):
        try:
            if file.endswith(".wav"):
                wav_path = "%s/%s" % (wavPath, file)
                out_path = "%s/%s" % (outPath, file.replace("wav", "npy"))  # intentional: replaces all occurrences to match filelist logic

                if os.path.exists(out_path):
                    continue

                feats = readwave(wav_path, normalize=saved_cfg.task.normalize)
                padding_mask = torch.BoolTensor(feats.shape).fill_(False)
                inputs = {
                    "source": (
                        feats.half().to(device)
                        if is_half and device not in ["mps", "cpu"]
                        else feats.to(device)
                    ),
                    "padding_mask": padding_mask.to(device),
                    "output_layer": 9 if version == "v1" else 12,  # layer 9
                }
                with torch.no_grad():
                    try:
                        logits = model.extract_features(**inputs)
                    except Exception:
                        if str(device) != "cpu":
                            printt(traceback.format_exc())
                            printt(
                                "HuBERT feature extraction failed on %s, fallback to CPU"
                                % device
                            )
                            device = "cpu"
                            model = model.to(device)
                            inputs["source"] = feats.to(device)
                            inputs["padding_mask"] = padding_mask.to(device)
                            logits = model.extract_features(**inputs)
                        else:
                            raise
                    feats = (
                        model.final_proj(logits[0]) if version == "v1" else logits[0]
                    )

                feats = feats.squeeze(0).float().cpu().numpy()
                if np.isnan(feats).sum() == 0:
                    np.save(out_path, feats, allow_pickle=False)
                else:
                    printt("%s-contains nan" % file)
                if idx % n == 0:
                    printt("now-%s,all-%s,%s,%s" % (len(todo), idx, file, feats.shape))
        except:
            printt(traceback.format_exc())
    printt("all-feature-done")
