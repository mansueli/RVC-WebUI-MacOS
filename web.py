import os
import sys
from dotenv import load_dotenv

now_dir = os.getcwd()
sys.path.append(now_dir)
load_dotenv()
load_dotenv("sha256.env")

if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

from infer.modules.vc import VC, show_info, hash_similarity
from infer.modules.uvr5.modules import uvr
from infer.lib.train.process_ckpt import (
    change_info,
    extract_small_model,
    merge,
)
from i18n.i18n import I18nAuto
from configs import Config
from sklearn.cluster import MiniBatchKMeans
import torch, platform
import numpy as np
import logging

# Configure root logger early so INFO startup messages (device, assets) are visible.
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Suppress noisy DEBUG loggers BEFORE importing gradio/PIL so the
# suppression is in place when PIL loads its image plugins.
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.pyplot").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("PIL.Image").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

import gradio as gr
import faiss
import pathlib
import json
import shlex
import subprocess
from time import sleep
from subprocess import Popen
from random import shuffle
import warnings
import traceback
import threading
import shutil
import logging


logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "assets", "weights"), exist_ok=True)
os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)


config = Config()
vc = VC(config)

if not config.nocheck:
    from infer.lib.rvcmd import check_all_assets, download_all_assets

    if not check_all_assets(update=config.update):
        if config.update:
            download_all_assets(tmpdir=tmp)
            if not check_all_assets(update=config.update):
                logging.error(
                    "Could not satisfy all required assets.\n"
                    "Run with --update to download missing files, or set "
                    "AUTO_DOWNLOAD_ASSETS=1 before running run.sh."
                )
                exit(1)

if config.dml == True:

    def forward_dml(ctx, x, scale):
        ctx.scale = scale
        res = x.clone().detach()
        return res

    import fairseq

    fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml

i18n = I18nAuto()
logger.info(i18n)
# 判断是否有能用来训练和加速推理的N卡
ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(
            value in gpu_name.upper()
            for value in [
                "10",
                "16",
                "20",
                "30",
                "40",
                "A2",
                "A3",
                "A4",
                "P4",
                "A50",
                "500",
                "A60",
                "70",
                "80",
                "90",
                "M4",
                "T4",
                "TITAN",
                "4060",
                "L",
                "6000",
            ]
        ):
            # A10#A100#V100#A40#P40#M40#K80#A4500
            if_gpu_ok = True  # 至少有一张能用的N卡
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            mem.append(
                int(
                    torch.cuda.get_device_properties(i).total_memory
                    / 1024
                    / 1024
                    / 1024
                    + 0.4
                )
            )
if if_gpu_ok and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
else:
    gpu_info = i18n(
        "Unfortunately, there is no compatible GPU available to support your training."
    )
    default_batch_size = 4
gpus = "-".join([i[0] for i in gpu_infos])


weight_root = os.getenv("weight_root")
weight_uvr5_root = os.getenv("weight_uvr5_root")
index_root = os.getenv("index_root")
outside_index_root = os.getenv("outside_index_root")


def _resolve_dir_env(env_value: str, default_rel_path: str, create=True):
    """Resolve configured directory roots with safe defaults.

    This prevents startup crashes when optional env vars are missing or invalid.
    """
    raw = (env_value or "").strip()
    resolved = raw if raw else os.path.join(now_dir, default_rel_path)
    if not os.path.isabs(resolved):
        resolved = os.path.join(now_dir, resolved)
    resolved = os.path.normpath(resolved)
    if create:
        os.makedirs(resolved, exist_ok=True)
    return resolved


weight_root = _resolve_dir_env(weight_root, "assets/weights")
weight_uvr5_root = _resolve_dir_env(weight_uvr5_root, "assets/uvr5_weights")
index_root = _resolve_dir_env(index_root, "logs")
outside_index_root = _resolve_dir_env(outside_index_root, "assets/indices")

logger.info("Resolved weight_root=%s", weight_root)
logger.info("Resolved weight_uvr5_root=%s", weight_uvr5_root)
logger.info("Resolved index_root=%s", index_root)
logger.info("Resolved outside_index_root=%s", outside_index_root)

names = [""]
index_paths = [""]


def lookup_names(weight_root):
    global names
    if not os.path.isdir(weight_root):
        logger.warning("Weight root does not exist: %s", weight_root)
        return
    for name in os.listdir(weight_root):
        if name.endswith(".pth"):
            names.append(name)


def lookup_indices(index_root):
    global index_paths
    if not os.path.isdir(index_root):
        logger.warning("Index root does not exist: %s", index_root)
        return
    for root, _, files in os.walk(index_root, topdown=False):
        for name in files:
            if name.endswith(".index") and "trained" not in name:
                index_paths.append(str(pathlib.Path(root, name)))


lookup_names(weight_root)
lookup_indices(index_root)
lookup_indices(outside_index_root)
uvr5_names = []
if os.path.isdir(weight_uvr5_root):
    for name in os.listdir(weight_uvr5_root):
        if name.endswith(".pth") or "onnx" in name:
            uvr5_names.append(name.replace(".pth", ""))
else:
    logger.warning("UVR5 weight root does not exist: %s", weight_uvr5_root)


def change_choices():
    global index_paths, names
    names = [""]
    lookup_names(weight_root)
    index_paths = [""]
    lookup_indices(index_root)
    lookup_indices(outside_index_root)
    return {"choices": sorted(names), "__type__": "update"}, {
        "choices": sorted(index_paths),
        "__type__": "update",
    }


def clean():
    return {"value": "", "__type__": "update"}


def export_onnx(ModelPath, ExportedPath):
    from rvc.onnx import export_onnx as eo

    eo(ModelPath, ExportedPath)


sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}


def if_done(done, p):
    while 1:
        if p.poll() is None:
            sleep(0.5)
        else:
            break
    done[0] = True


def if_done_multi(done, ps):
    while 1:
        # poll==None代表进程未结束
        # 只要有一个进程未结束都不停
        flag = 1
        for p in ps:
            if p.poll() is None:
                flag = 0
                sleep(0.5)
                break
        if flag == 1:
            break
    done[0] = True


def _default_rmvpe_worker_devices():
    if sys.platform == "darwin":
        return ["mps" if str(config.device) == "mps" else "cpu"]
    if torch.cuda.is_available() and gpus:
        return [f"cuda:{gpu}" for gpu in gpus.split("-") if gpu]
    return [str(config.device)]


def _parse_rmvpe_worker_devices(workers_text: str):
    if workers_text is None or workers_text.strip() == "":
        return _default_rmvpe_worker_devices()

    devices = []
    for token in workers_text.replace(",", "-").split("-"):
        dev = token.strip().lower()
        if not dev:
            continue
        if dev.isdigit() and sys.platform != "darwin":
            devices.append(f"cuda:{dev}")
        elif dev in ["cpu", "mps"] or dev.startswith("cuda:"):
            devices.append(dev)
        else:
            logger.warning("Ignoring unknown RMVPE worker device: %s", token)

    if not devices:
        devices = _default_rmvpe_worker_devices()

    # Apple Silicon is fastest and most stable with a single MPS RMVPE worker.
    if sys.platform == "darwin" and any(d == "mps" for d in devices):
        devices = ["mps"]

    return devices


def change_f0_method(f0_method):
    visible = f0_method == "rmvpe_gpu"
    default_devices = "mps" if sys.platform == "darwin" else (gpus if gpus else "cpu")
    return {
        "visible": visible,
        "value": default_devices if visible else "",
        "__type__": "update",
    }


def _is_f0_enabled(if_f0_3):
    return if_f0_3 is True or if_f0_3 == i18n("Yes") or str(if_f0_3).lower() == "true"


def _count_files_with_suffix(path_obj: pathlib.Path, suffix: str):
    if not path_obj.exists():
        return 0
    return sum(1 for p in path_obj.iterdir() if p.is_file() and p.name.lower().endswith(suffix))


def _get_experiment_progress(exp_dir1, if_f0_3, version19):
    exp_path = pathlib.Path(now_dir, "logs", exp_dir1)
    wav16k_dir = exp_path / "1_16k_wavs"
    f0_dir = exp_path / "2a_f0"
    f0nsf_dir = exp_path / "2b-f0nsf"
    feature_dir = exp_path / ("3_feature256" if version19 == "v1" else "3_feature768")

    wav_count = _count_files_with_suffix(wav16k_dir, ".wav")
    f0_count = _count_files_with_suffix(f0_dir, ".npy")
    f0nsf_count = _count_files_with_suffix(f0nsf_dir, ".npy")
    feature_count = _count_files_with_suffix(feature_dir, ".npy")

    preprocess_done = wav_count > 0
    f0_required = _is_f0_enabled(if_f0_3)
    f0_done = (not f0_required) or (f0_count >= wav_count and f0nsf_count >= wav_count)
    extract_done = preprocess_done and feature_count >= wav_count and f0_done

    return {
        "exists": exp_path.exists(),
        "exp_path": str(exp_path),
        "wav_count": wav_count,
        "f0_count": f0_count,
        "f0nsf_count": f0nsf_count,
        "feature_count": feature_count,
        "preprocess_done": preprocess_done,
        "extract_done": extract_done,
        "f0_required": f0_required,
    }


def inspect_experiment_progress(exp_dir1, if_f0_3, version19):
    if not exp_dir1 or not exp_dir1.strip():
        return "Please enter an experiment name first."

    p = _get_experiment_progress(exp_dir1.strip(), if_f0_3, version19)
    if not p["exists"]:
        return "No existing experiment folder found yet. A new one will be created." \
            + "\nPath: " + p["exp_path"]

    lines = [
        "Experiment: " + exp_dir1.strip(),
        "Path: " + p["exp_path"],
        "16k wavs: %s" % p["wav_count"],
        "Features: %s" % p["feature_count"],
        "F0: %s" % p["f0_count"],
        "F0 NSF: %s" % p["f0nsf_count"],
        "Step 1 (Preprocess): " + ("DONE" if p["preprocess_done"] else "PENDING"),
        "Step 2 (Extract): " + ("DONE" if p["extract_done"] else "PENDING"),
    ]
    return "\n".join(lines)


def clear_experiment_artifacts(exp_dir1, if_f0_3, version19, clear_step1, clear_step2):
    if not exp_dir1 or not exp_dir1.strip():
        return "Please enter an experiment name first."

    exp_name = exp_dir1.strip()
    exp_path = pathlib.Path(now_dir, "logs", exp_name)
    if not exp_path.exists():
        return (
            "No experiment folder found to clear.\n"
            + inspect_experiment_progress(exp_name, if_f0_3, version19)
        )

    targets = []
    if clear_step1:
        targets.extend(
            [
                exp_path / "0_gt_wavs",
                exp_path / "1_16k_wavs",
                exp_path / "preprocess.log",
            ]
        )
    if clear_step2:
        targets.extend(
            [
                exp_path / "2a_f0",
                exp_path / "2b-f0nsf",
                exp_path / "3_feature256",
                exp_path / "3_feature768",
                exp_path / "extract_f0_feature.log",
            ]
        )

    removed = []
    for target in targets:
        if not target.exists():
            continue
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        else:
            try:
                target.unlink()
            except FileNotFoundError:
                pass
        removed.append(str(target.relative_to(pathlib.Path(now_dir))))

    if not removed:
        msg = "Nothing to clear for the selected options."
    else:
        msg = "Cleared %s items:\n%s" % (len(removed), "\n".join(removed))

    return msg + "\n\n" + inspect_experiment_progress(exp_name, if_f0_3, version19)


def suggest_default_experiment_name():
    logs_dir = pathlib.Path(now_dir, "logs")
    if not logs_dir.exists():
        return "mi-test"

    candidates = []
    for p in logs_dir.iterdir():
        if not p.is_dir() or p.name == "mute":
            continue
        if (
            (p / "1_16k_wavs").exists()
            or (p / "2a_f0").exists()
            or (p / "2b-f0nsf").exists()
            or (p / "3_feature256").exists()
            or (p / "3_feature768").exists()
            or (p / "G_0.pth").exists()
        ):
            candidates.append(p)

    if not candidates:
        return "mi-test"

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest.name


default_exp_dir = suggest_default_experiment_name()


def preprocess_dataset(trainset_dir, exp_dir, sr, n_p, preprocess_mode=None):
    sr = sr_dict[sr]
    exp_path = pathlib.Path(now_dir, "logs", exp_dir)
    os.makedirs(exp_path, exist_ok=True)
    log_file_path = exp_path / "preprocess.log"
    f = open(log_file_path, "w")
    f.close()
    force_preprocess = str(preprocess_mode) == i18n("Full rebuild (reprocess all files)")
    cmd = '"%s" infer/modules/train/preprocess.py "%s" %s %s "%s" %s %.1f %s' % (
        config.python_cmd,
        trainset_dir,
        sr,
        n_p,
        str(exp_path),
        config.noparallel,
        config.preprocess_per,
        "True" if force_preprocess else "False",
    )
    logger.info("Execute: " + cmd)
    # , stdin=PIPE, stdout=PIPE,stderr=PIPE,cwd=now_dir
    p = Popen(cmd, shell=True)
    # 煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
    done = [False]
    threading.Thread(
        target=if_done,
        args=(
            done,
            p,
        ),
    ).start()
    while 1:
        with open(log_file_path, "r") as f:
            yield (f.read())
        sleep(1)
        if done[0]:
            break
    with open(log_file_path, "r") as f:
        log = f.read()
    logger.info(log)
    yield log


# but2.click(extract_f0,[gpus6,np7,f0method8,if_f0_3,trainset_dir4],[info2])
def extract_f0_feature(n_p, f0method, if_f0, exp_dir, version19, rmvpe_workers=""):
    def _count_wavs(inp_root):
        if not os.path.exists(inp_root):
            return 0
        return len([n for n in os.listdir(inp_root) if n.lower().endswith(".wav")])

    def _count_npy(inp_root):
        if not os.path.exists(inp_root):
            return 0
        return len([n for n in os.listdir(inp_root) if n.lower().endswith(".npy")])

    def _build_extract_summary(log_text):
        wav_root = "%s/logs/%s/1_16k_wavs" % (now_dir, exp_dir)
        f0_root = "%s/logs/%s/2a_f0" % (now_dir, exp_dir)
        f0nsf_root = "%s/logs/%s/2b-f0nsf" % (now_dir, exp_dir)
        feat_root = (
            "%s/logs/%s/3_feature256" % (now_dir, exp_dir)
            if version19 == "v1"
            else "%s/logs/%s/3_feature768" % (now_dir, exp_dir)
        )

        wav_count = _count_wavs(wav_root)
        f0_count = _count_npy(f0_root)
        f0nsf_count = _count_npy(f0nsf_root)
        feat_count = _count_npy(feat_root)

        f0_fail_count = log_text.count("f0fail-")
        process_fail_count = log_text.count("process-failed-")
        feature_fail_count = max(log_text.count("Traceback") - f0_fail_count, 0)
        feature_fail_count += process_fail_count

        if if_f0:
            f0_ok = f0_fail_count == 0 and f0_count >= wav_count and f0nsf_count >= wav_count
            f0_line = "F0: %s (%s/%s), NSF: %s/%s, Failures: %s" % (
                "COMPLETE" if f0_ok else "FAILED",
                f0_count,
                wav_count,
                f0nsf_count,
                wav_count,
                f0_fail_count,
            )
        else:
            f0_ok = True
            f0_line = "F0: SKIPPED"

        feat_ok = feature_fail_count == 0 and feat_count >= wav_count
        feat_line = "Features: %s (%s/%s), Failures: %s" % (
            "COMPLETE" if feat_ok else "FAILED",
            feat_count,
            wav_count,
            feature_fail_count,
        )

        overall_ok = f0_ok and feat_ok
        status = "COMPLETE" if overall_ok else "FAILED"
        return (
            "\n\n=== Extraction Status ===\n"
            + "Overall: "
            + status
            + "\n"
            + f0_line
            + "\n"
            + feat_line
        )

    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "w")
    f.close()
    if if_f0:
        if f0method == "rmvpe_gpu":
            worker_devices = _parse_rmvpe_worker_devices(rmvpe_workers)
            logger.info("RMVPE multi-worker mode devices: %s", worker_devices)
            ps = []
            worker_count = len(worker_devices)
            for idx, worker_device in enumerate(worker_devices):
                # fp16 on CPU is slower and less stable than fp32.
                worker_half = str(config.is_half and worker_device != "cpu")
                cmd = (
                    '"%s" infer/modules/train/extract_f0_rmvpe_worker.py "%s/logs/%s" %s %s "%s" %s'
                    % (
                        config.python_cmd,
                        now_dir,
                        exp_dir,
                        worker_count,
                        idx,
                        worker_device,
                        worker_half,
                    )
                )
                logger.info("Execute: " + cmd)
                p = Popen(cmd, shell=True, cwd=now_dir)
                ps.append(p)
            done = [False]
            threading.Thread(
                target=if_done_multi,
                args=(
                    done,
                    ps,
                ),
            ).start()
        else:
            cmd = (
                '"%s" infer/modules/train/extract_f0_print.py "%s/logs/%s" %s %s "%s" %s'
                % (
                    config.python_cmd,
                    now_dir,
                    exp_dir,
                    n_p,
                    f0method,
                    config.device,
                    str(config.is_half),
                )
            )
            logger.info("Execute: " + cmd)
            p = Popen(
                cmd, shell=True, cwd=now_dir
            )  # , stdin=PIPE, stdout=PIPE,stderr=PIPE
            done = [False]
            threading.Thread(
                target=if_done,
                args=(
                    done,
                    p,
                ),
            ).start()
        while 1:
            with open(
                "%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r"
            ) as f:
                yield (f.read())
            sleep(1)
            if done[0]:
                break

        # Surface worker process failures even when stderr is not in the streamed log.
        if f0method == "rmvpe_gpu":
            for idx, proc in enumerate(ps):
                code = proc.poll()
                if code not in (0, None):
                    with open(
                        "%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir),
                        "a",
                    ) as f:
                        f.write("process-failed-f0-worker-%s-exitcode-%s\n" % (idx, code))
        else:
            code = p.poll()
            if code not in (0, None):
                with open(
                    "%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir),
                    "a",
                ) as f:
                    f.write("process-failed-f0-worker-0-exitcode-%s\n" % code)

        with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
            log = f.read()
        logger.info(log)
        yield log + _build_extract_summary(log)
    # 对不同part分别开多进程
    """
    n_part=int(sys.argv[1])
    i_part=int(sys.argv[2])
    i_gpu=sys.argv[3]
    exp_dir=sys.argv[4]
    os.environ["CUDA_VISIBLE_DEVICES"]=str(i_gpu)
    """
    gpu_list = [g for g in gpus.split("-") if g]
    leng = len(gpu_list)
    ps = []
    if leng == 0:
        cmd = (
            '"%s" infer/modules/train/extract_feature_print.py %s %s %s "%s/logs/%s" %s %s'
            % (
                config.python_cmd,
                config.device,
                1,
                0,
                now_dir,
                exp_dir,
                version19,
                config.is_half,
            )
        )
        logger.info("Execute: " + cmd)
        p = Popen(cmd, shell=True, cwd=now_dir)
        ps.append(p)
    else:
        for idx, n_g in enumerate(gpu_list):
            cmd = (
                '"%s" infer/modules/train/extract_feature_print.py %s %s %s %s "%s/logs/%s" %s %s'
                % (
                    config.python_cmd,
                    config.device,
                    leng,
                    idx,
                    n_g,
                    now_dir,
                    exp_dir,
                    version19,
                    config.is_half,
                )
            )
            logger.info("Execute: " + cmd)
            p = Popen(
                cmd, shell=True, cwd=now_dir
            )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
            ps.append(p)
    # 煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
    done = [False]
    threading.Thread(
        target=if_done_multi,
        args=(
            done,
            ps,
        ),
    ).start()
    while 1:
        with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
            yield (f.read())
        sleep(1)
        if done[0]:
            break

    # Surface worker process failures even when stderr is not in the streamed log.
    for idx, proc in enumerate(ps):
        code = proc.poll()
        if code not in (0, None):
            with open(
                "%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir),
                "a",
            ) as f:
                f.write("process-failed-feature-worker-%s-exitcode-%s\n" % (idx, code))

    with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
        log = f.read()
    logger.info(log)
    yield log + _build_extract_summary(log)


def get_pretrained_models(path_str, f0_str, sr2):
    if_pretrained_generator_exist = os.access(
        "assets/pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2), os.F_OK
    )
    if_pretrained_discriminator_exist = os.access(
        "assets/pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2), os.F_OK
    )
    if not if_pretrained_generator_exist:
        logger.warning(
            "assets/pretrained%s/%sG%s.pth not exist, will not use pretrained model",
            path_str,
            f0_str,
            sr2,
        )
    if not if_pretrained_discriminator_exist:
        logger.warning(
            "assets/pretrained%s/%sD%s.pth not exist, will not use pretrained model",
            path_str,
            f0_str,
            sr2,
        )
    return (
        (
            "assets/pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2)
            if if_pretrained_generator_exist
            else ""
        ),
        (
            "assets/pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2)
            if if_pretrained_discriminator_exist
            else ""
        ),
    )


def change_sr2(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    f0_str = "f0" if if_f0_3 else ""
    return get_pretrained_models(path_str, f0_str, sr2)


def change_version19(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    if sr2 == "32k" and version19 == "v1":
        sr2 = "40k"
    if version19 == "v1":
        to_return_sr2 = {
            "choices": ["40k", "48k"],
            "__type__": "update",
            "value": sr2 if sr2 in ["40k", "48k"] else "48k",
        }
    elif version19 == "v3":
        to_return_sr2 = {
            "choices": ["48k"],
            "__type__": "update",
            "value": "48k",
        }
        sr2 = "48k"
    else:
        to_return_sr2 = {
            "choices": ["32k", "40k", "48k"],
            "__type__": "update",
            "value": sr2,
        }
    f0_str = "f0" if if_f0_3 else ""
    # V3 does not use RVC pretrained G/D — return empty paths for v3.
    if version19 == "v3":
        pretrained_g = ""
        pretrained_d = ""
    else:
        pretrained_g, pretrained_d = get_pretrained_models(path_str, f0_str, sr2)
    v3_row_visible = {"visible": version19 == "v3", "__type__": "update"}
    return (
        pretrained_g,
        pretrained_d,
        to_return_sr2,
        v3_row_visible,
    )


def change_f0(if_f0_3, sr2, version19):  # f0method8,pretrained_G14,pretrained_D15
    path_str = "" if version19 == "v1" else "_v2"
    return (
        {"visible": if_f0_3, "__type__": "update"},
        *get_pretrained_models(path_str, "f0" if if_f0_3 == True else "", sr2),
    )


def _ensure_mute_samples(sr: int, fea_dim: int):
    """Generate silent mute training samples under logs/mute/ if missing.

    These padding samples are appended to every filelist so the dataset always
    has at least a few silent examples.  Two seconds of silence gives enough
    frames to survive the segment_size filter in DistributedBucketSampler.
    """
    import wave, struct  # stdlib — no extra deps
    n_seconds = 2
    n_samples = sr * n_seconds                 # 96000 @ 48k
    hop_length = 480 if sr >= 40000 else 160   # match config hop_length
    T_spec = n_samples // hop_length           # spectrogram frames (200 @ 48k)
    T_feat = T_spec // 2                       # feature frames (before np.repeat)

    mute_dir   = os.path.join(now_dir, "logs", "mute")
    wav_dir    = os.path.join(mute_dir, "0_gt_wavs")
    feat_dir   = os.path.join(mute_dir, "3_feature%d" % fea_dim)
    f0_dir     = os.path.join(mute_dir, "2a_f0")
    f0nsf_dir  = os.path.join(mute_dir, "2b-f0nsf")
    for d in (wav_dir, feat_dir, f0_dir, f0nsf_dir):
        os.makedirs(d, exist_ok=True)

    wav_path    = os.path.join(wav_dir,   "mute%dk.wav" % (sr // 1000))
    feat_path   = os.path.join(feat_dir,  "mute.npy")
    f0_path     = os.path.join(f0_dir,    "mute.wav.npy")
    f0nsf_path  = os.path.join(f0nsf_dir, "mute.wav.npy")

    if not os.path.exists(wav_path):
        with wave.open(wav_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)          # 16-bit
            wf.setframerate(sr)
            wf.writeframes(struct.pack("<%dh" % n_samples, *([0] * n_samples)))
        logger.info("Generated mute wav: %s", wav_path)

    if not os.path.exists(feat_path):
        np.save(feat_path, np.zeros((T_feat, fea_dim), dtype=np.float32))
        logger.info("Generated mute feature: %s", feat_path)

    if not os.path.exists(f0_path):
        np.save(f0_path, np.zeros(T_spec, dtype=np.float32))
        logger.info("Generated mute f0: %s", f0_path)

    if not os.path.exists(f0nsf_path):
        np.save(f0nsf_path, np.zeros(T_spec, dtype=np.float32))
        logger.info("Generated mute f0nsf: %s", f0nsf_path)


TRAINING_PROCESSES = {}


def _is_process_alive(pid: int) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _active_training_pid(exp_name: str):
    proc = TRAINING_PROCESSES.get(exp_name)
    if proc is not None and proc.poll() is None:
        return proc.pid

    status_file = pathlib.Path(now_dir, "logs", exp_name, "training_status.json")
    if status_file.exists():
        try:
            status = json.loads(status_file.read_text(encoding="utf-8"))
            sup_pid = int(status.get("supervisor_pid", 0) or 0)
            if status.get("running") and _is_process_alive(sup_pid):
                return sup_pid
        except Exception:
            pass

    pid_file = pathlib.Path(now_dir, "logs", exp_name, "training_pid.txt")
    if not pid_file.exists():
        return None
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except Exception:
        return None

    return pid if _is_process_alive(pid) else None


def _training_artifacts(exp_name: str):
    exp_dir = pathlib.Path(now_dir, "logs", exp_name)
    return {
        "exp_dir": exp_dir,
        "pid_file": exp_dir / "training_pid.txt",
        "status_file": exp_dir / "training_status.json",
        "log_file": exp_dir / "train_webui.log",
        "stop_request_file": exp_dir / "training_stop_after_epoch.flag",
        "stop_ack_file": exp_dir / "training_stopped.flag",
    }


def _build_supervisor_cmd(
    train_cmd: str,
    exp_name: str,
    status_file: pathlib.Path,
    log_file: pathlib.Path,
    stop_request_file: pathlib.Path,
    stop_ack_file: pathlib.Path,
):
    sup_py = config.python_cmd
    return (
        '"%s" tools/cmd/train_supervisor.py --cmd %s --cwd %s --exp %s --log-file %s --status-file %s --stop-request-file %s --stop-ack-file %s --max-retries 3 --retry-delay 20'
        % (
            sup_py,
            shlex.quote(train_cmd),
            shlex.quote(now_dir),
            shlex.quote(exp_name),
            shlex.quote(str(log_file)),
            shlex.quote(str(status_file)),
            shlex.quote(str(stop_request_file)),
            shlex.quote(str(stop_ack_file)),
        )
    )


def request_graceful_training_stop(exp_dir1):
    if not exp_dir1 or not exp_dir1.strip():
        return "Please enter an experiment name first."

    exp_name = exp_dir1.strip()
    artifacts = _training_artifacts(exp_name)
    active_pid = _active_training_pid(exp_name)
    if active_pid is None:
        return (
            "No active training process was found for this experiment.\n"
            + get_training_status(exp_name)
        )

    artifacts["exp_dir"].mkdir(parents=True, exist_ok=True)
    artifacts["stop_request_file"].write_text(
        "stop_after_current_epoch=true\n", encoding="utf-8"
    )

    if artifacts["status_file"].exists():
        try:
            status = json.loads(artifacts["status_file"].read_text(encoding="utf-8"))
            status["message"] = (
                "Graceful stop requested: will save and stop after current epoch"
            )
            status["state"] = "stopping"
            status["updated_at"] = datetime.now().isoformat(timespec="seconds")
            artifacts["status_file"].write_text(
                json.dumps(status, ensure_ascii=True, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass

    return (
        "Graceful stop requested for experiment '%s'. " % exp_name
        + "Training will stop at epoch boundary and save checkpoints before exit.\n"
        + get_training_status(exp_name)
    )


def get_training_status(exp_dir1):
    if not exp_dir1 or not exp_dir1.strip():
        return "Please enter an experiment name first."

    exp_name = exp_dir1.strip()
    artifacts = _training_artifacts(exp_name)
    lines = ["Experiment: %s" % exp_name]

    status = None
    if artifacts["status_file"].exists():
        try:
            status = json.loads(artifacts["status_file"].read_text(encoding="utf-8"))
        except Exception as e:
            lines.append("Status file parse error: %s" % e)

    if status:
        sup_pid = int(status.get("supervisor_pid", 0) or 0)
        child_pid = int(status.get("child_pid", 0) or 0)
        running = bool(status.get("running")) and _is_process_alive(sup_pid)
        state = "running" if running else str(status.get("state", "unknown"))
        lines.extend(
            [
                "State: %s" % state,
                "Attempt: %s/%s"
                % (
                    status.get("attempt", "?"),
                    int(status.get("max_retries", 0) or 0) + 1,
                ),
                "Supervisor PID: %s" % (sup_pid or "n/a"),
                "Child PID: %s" % (child_pid or "n/a"),
                "Last exit code: %s" % status.get("last_exit_code", "n/a"),
                "Last error type: %s" % status.get("last_error_type", "n/a"),
                "Message: %s" % status.get("message", ""),
                "Updated at: %s" % status.get("updated_at", "n/a"),
            ]
        )
    else:
        pid = _active_training_pid(exp_name)
        if pid is not None:
            lines.append("State: running (PID %s)" % pid)
        else:
            lines.append("State: idle (no active training process found)")

    lines.append("Log file: %s" % artifacts["log_file"])
    if artifacts["log_file"].exists():
        with open(artifacts["log_file"], "r", encoding="utf-8", errors="ignore") as f:
            tail = f.readlines()[-25:]
        if tail:
            lines.append("--- Last log lines ---")
            lines.extend([ln.rstrip("\n") for ln in tail])

    return "\n".join(lines)


# but3.click(click_train,[exp_dir1,sr2,if_f0_3,save_epoch10,total_epoch11,batch_size12,if_save_latest13,pretrained_G14,pretrained_D15,gpus16])
def click_train(
    exp_dir1,
    sr2,
    if_f0_3,
    spk_id5,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
    author,
    run_async=True,
):
    # V3 uses the HQ-SVC-oriented training adapter, not the RVC trainer.
    if version19 == "v3":
        return click_train_v3(
            exp_dir1,
            sr2,
            if_f0_3,
            save_epoch10,
            total_epoch11,
            batch_size12,
            if_save_every_weights18,
            gpus16,
            author,
            run_async=run_async,
        )

    # 生成filelist
    exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = "%s/0_gt_wavs" % (exp_dir)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    progress = _get_experiment_progress(exp_dir1, if_f0_3, version19)
    if not os.path.exists(gt_wavs_dir):
        return "Training data is missing. Please run Step 1 (preprocess) first."
    if not os.path.exists(feature_dir):
        return "Feature extraction output is missing. Please run Step 2 feature extraction first."
    if progress["feature_count"] == 0:
        return (
            "No feature files were found in %s. " % feature_dir
            + "Current progress: wav16k=%s, f0=%s, f0nsf=%s, features=%s. "
            % (
                progress["wav_count"],
                progress["f0_count"],
                progress["f0nsf_count"],
                progress["feature_count"],
            )
            + "Run Step 2 (Feature extraction) again."
        )

    if if_f0_3:
        f0_dir = "%s/2a_f0" % (exp_dir)
        f0nsf_dir = "%s/2b-f0nsf" % (exp_dir)
        if not os.path.exists(f0_dir) or not os.path.exists(f0nsf_dir):
            return "F0 extraction output is missing. Please run Step 2 feature extraction with F0 enabled first."
    # Build filelist from actual files in 0_gt_wavs, checking that
    # matching outputs exist in all required directories.
    opt = []
    names = []
    for fname in sorted(os.listdir(gt_wavs_dir)):
        if not fname.endswith(".wav"):
            continue
        gt_path = os.path.join(gt_wavs_dir, fname)
        # Feature extractor uses file.replace("wav", "npy") which
        # replaces ALL occurrences; mirror that here.
        feat_name = fname.replace("wav", "npy")
        feat_path = os.path.join(feature_dir, feat_name)
        if not os.path.exists(feat_path):
            continue
        if if_f0_3:
            f0_path = os.path.join(f0_dir, fname + ".npy")
            f0nsf_path = os.path.join(f0nsf_dir, fname + ".npy")
            if not os.path.exists(f0_path) or not os.path.exists(f0nsf_path):
                continue
            opt.append("%s|%s|%s|%s|%s" % (gt_path, feat_path, f0_path, f0nsf_path, spk_id5))
        else:
            opt.append("%s|%s|%s" % (gt_path, feat_path, spk_id5))
        names.append(fname)
    fea_dim = 256 if version19 == "v1" else 768
    if len(names) == 0:
        return (
            "No matched training samples were found. "
            + "Current progress: wav16k=%s, f0=%s, f0nsf=%s, features=%s. "
            % (
                progress["wav_count"],
                progress["f0_count"],
                progress["f0nsf_count"],
                progress["feature_count"],
            )
            + "Re-run Step 2 (feature extraction), then try training again."
        )
    # Ensure mute (silence padding) samples exist before building the filelist.
    sr_int = int(sr2.replace("k", "")) * 1000
    _ensure_mute_samples(sr_int, fea_dim)
    if if_f0_3:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, now_dir, now_dir, spk_id5)
            )
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, spk_id5)
            )
    shuffle(opt)
    with open("%s/filelist.txt" % exp_dir, "w") as f:
        f.write("\n".join(opt))
    logger.debug("Write filelist done")
    logger.info("Use gpus: %s", str(gpus16))
    if pretrained_G14 == "":
        logger.info("No pretrained Generator")
    if pretrained_D15 == "":
        logger.info("No pretrained Discriminator")
    if version19 == "v1" or sr2 == "40k":  # v2 40k falls back to v1
        config_path = "v1/%s.json" % sr2
    else:
        config_path = "v2/%s.json" % sr2
    config_save_path = os.path.join(exp_dir, "config.json")
    if not pathlib.Path(config_save_path).exists():
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(
                config.json_config[config_path],
                f,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            )
            f.write("\n")
    cmd = (
        '"%s" infer/modules/train/train.py -e "%s" -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s -a "%s"'
        % (
            config.python_cmd,
            exp_dir1,
            sr2,
            1 if if_f0_3 else 0,
            batch_size12,
            total_epoch11,
            save_epoch10,
            '-pg "%s"' % pretrained_G14 if pretrained_G14 != "" else "",
            '-pd "%s"' % pretrained_D15 if pretrained_D15 != "" else "",
            1 if if_save_latest13 == i18n("Yes") else 0,
            1 if if_cache_gpu17 == i18n("Yes") else 0,
            1 if if_save_every_weights18 == i18n("Yes") else 0,
            version19,
            author,
        )
    )
    if gpus16:
        cmd += ' -g "%s"' % (gpus16)

    existing_pid = _active_training_pid(exp_dir1)
    if existing_pid is not None:
        return (
            "Training is already running for this experiment (PID %s). " % existing_pid
            + "You can safely close the browser tab; training will continue in the background."
        )

    logger.info("Execute: " + cmd)
    artifacts = _training_artifacts(exp_dir1)
    webui_log_path = artifacts["log_file"]
    status_file_path = artifacts["status_file"]
    supervisor_cmd = _build_supervisor_cmd(
        cmd,
        exp_dir1,
        status_file_path,
        webui_log_path,
        artifacts["stop_request_file"],
        artifacts["stop_ack_file"],
    )

    # Clear stale graceful-stop markers from previous runs.
    for marker in [artifacts["stop_request_file"], artifacts["stop_ack_file"]]:
        if marker.exists():
            try:
                marker.unlink()
            except OSError:
                pass

    if run_async:
        with open(webui_log_path, "a", encoding="utf-8") as train_log_file:
            p = Popen(
                supervisor_cmd,
                shell=True,
                cwd=now_dir,
                stdin=subprocess.DEVNULL,
                stdout=train_log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        TRAINING_PROCESSES[exp_dir1] = p
        artifacts["pid_file"].write_text(
            str(p.pid), encoding="utf-8"
        )
        return (
            "Training supervisor started in background (PID %s). " % p.pid
            + "You can close the browser tab; training will keep running. "
            + "Auto-resume is enabled for failures (up to 3 retries, with OOM batch-size backoff). "
            + "Logs: %s"
            % webui_log_path
        )

    rc = subprocess.run(supervisor_cmd, shell=True, cwd=now_dir).returncode
    final_msg = get_training_status(exp_dir1)
    if rc == 0:
        return "Training complete.\n" + final_msg
    return "Training failed after automatic retries.\n" + final_msg


# but4.click(train_index, [exp_dir1], info3)
def train_index(exp_dir1, version19):
    # exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    exp_dir = "logs/%s" % (exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if not os.path.exists(feature_dir):
        return "请先进行特征提取!"
    listdir_res = list(os.listdir(feature_dir))
    if len(listdir_res) == 0:
        return "请先进行特征提取！"
    infos = []
    npys = []
    for name in sorted(listdir_res):
        phone = np.load("%s/%s" % (feature_dir, name))
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    if big_npy.shape[0] > 2e5:
        infos.append("Trying doing kmeans %s shape to 10k centers." % big_npy.shape[0])
        yield "\n".join(infos)
        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * config.n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except:
            info = traceback.format_exc()
            logger.info(info)
            infos.append(info)
            yield "\n".join(infos)

    np.save("%s/total_fea.npy" % exp_dir, big_npy)
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    infos.append("%s,%s" % (big_npy.shape, n_ivf))
    yield "\n".join(infos)
    index = faiss.index_factory(256 if version19 == "v1" else 768, "IVF%s,Flat" % n_ivf)
    # index = faiss.index_factory(256if version19=="v1"else 768, "IVF%s,PQ128x4fs,RFlat"%n_ivf)
    infos.append("training")
    yield "\n".join(infos)
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        "%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )
    infos.append("adding")
    yield "\n".join(infos)
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    index_save_path = "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index" % (
        exp_dir,
        n_ivf,
        index_ivf.nprobe,
        exp_dir1,
        version19,
    )
    faiss.write_index(index, index_save_path)
    infos.append(i18n("Successfully built index into") + " " + index_save_path)
    link_target = "%s/%s_IVF%s_Flat_nprobe_%s_%s_%s.index" % (
        outside_index_root,
        exp_dir1,
        n_ivf,
        index_ivf.nprobe,
        exp_dir1,
        version19,
    )
    try:
        link = os.link if platform.system() == "Windows" else os.symlink
        link(index_save_path, link_target)
        infos.append(i18n("Link index to outside folder") + " " + link_target)
    except:
        infos.append(
            i18n("Link index to outside folder")
            + " "
            + link_target
            + " "
            + i18n("Fail")
        )

    # faiss.write_index(index, '%s/added_IVF%s_Flat_FastScan_%s.index'%(exp_dir,n_ivf,version19))
    # infos.append("成功构建索引，added_IVF%s_Flat_FastScan_%s.index"%(n_ivf,version19))
    yield "\n".join(infos)


# but5.click(train1key, [exp_dir1, sr2, if_f0_3, trainset_dir4, spk_id5, gpus6, np7, f0method8, save_epoch10, total_epoch11, batch_size12, if_save_latest13, pretrained_G14, pretrained_D15, gpus16, if_cache_gpu17], info3)
def train1key(
    exp_dir1,
    sr2,
    if_f0_3,
    trainset_dir4,
    preprocess_mode8,
    spk_id5,
    np7,
    f0method8,
    rmvpe_workers8,
    auto_resume8,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
    author,
):
    infos = []

    def get_info_str(strr):
        infos.append(strr)
        return "\n".join(infos)

    exp_name = exp_dir1.strip()
    if_f0_enabled = _is_f0_enabled(if_f0_3)
    force_preprocess = str(preprocess_mode8) == i18n("Full rebuild (reprocess all files)")

    progress = _get_experiment_progress(exp_name, if_f0_enabled, version19)

    # step1:Process data
    if auto_resume8 and progress["preprocess_done"] and not force_preprocess:
        yield get_info_str(
            i18n("Step 1: Processing data")
            + " -> skipped (existing 16k wavs: %s)" % progress["wav_count"]
        )
    else:
        yield get_info_str(i18n("Step 1: Processing data"))
        [
            get_info_str(_)
            for _ in preprocess_dataset(
                trainset_dir4,
                exp_name,
                sr2,
                np7,
                preprocess_mode8,
            )
        ]

    # step2a:提取音高
    progress = _get_experiment_progress(exp_name, if_f0_enabled, version19)
    if auto_resume8 and progress["extract_done"]:
        yield get_info_str(
            i18n("step2:Pitch extraction & feature extraction")
            + " -> skipped (features: %s/%s)"
            % (progress["feature_count"], progress["wav_count"])
        )
    else:
        yield get_info_str(i18n("step2:Pitch extraction & feature extraction"))
        [
            get_info_str(_)
            for _ in extract_f0_feature(
                np7,
                f0method8,
                if_f0_enabled,
                exp_name,
                version19,
                rmvpe_workers8,
            )
        ]

    # step3a:Train model
    yield get_info_str(i18n("Step 3a: Model training started"))
    train_result = click_train(
        exp_name,
        sr2,
        if_f0_enabled,
        spk_id5,
        save_epoch10,
        total_epoch11,
        batch_size12,
        if_save_latest13,
        pretrained_G14,
        pretrained_D15,
        gpus16,
        if_cache_gpu17,
        if_save_every_weights18,
        version19,
        author,
        run_async=False,
    )
    yield get_info_str(train_result)

    # step3b: Build FAISS retrieval index (v1/v2 only; V3 uses its own retrieval path)
    if version19 != "v3":
        [get_info_str(_) for _ in train_index(exp_dir1, version19)]
    else:
        yield get_info_str(
            "Step 3b: FAISS index skipped for V3 (HQ-SVC uses its own retrieval path)."
        )
    yield get_info_str(i18n("All processes have been completed!"))


#                    ckpt_path2.change(change_info_,[ckpt_path2],[sr__,if_f0__])
def change_info_(ckpt_path):
    if not os.path.exists(ckpt_path.replace(os.path.basename(ckpt_path), "train.log")):
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}
    try:
        with open(
            ckpt_path.replace(os.path.basename(ckpt_path), "train.log"), "r"
        ) as f:
            info = eval(f.read().strip("\n").split("\n")[0].split("\t")[-1])
            sr, f0 = info["sample_rate"], info["if_f0"]
            version = "v2" if ("version" in info and info["version"] == "v2") else "v1"
            return sr, str(f0), version
    except:
        traceback.print_exc()
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}


# ---------------------------------------------------------------------------
# V3 (HQ-SVC-oriented) backend functions
# ---------------------------------------------------------------------------

def v3_preprocess_dataset(v3_source_dir, exp_dir1):
    """Run YingMusic vocal isolation for V3 dataset preparation.

    Invokes yingmusic_experiment.py in setup-only mode (environment check) or
    with --source-dir / --output-dir if a source directory is provided.
    Yields log lines streamed to the WebUI status box.
    """
    exp_name = (exp_dir1 or "unnamed").strip()
    isolated_dir = pathlib.Path(now_dir, "logs", exp_name, "v3_isolated")
    isolated_dir.mkdir(parents=True, exist_ok=True)
    log_path = pathlib.Path(now_dir, "logs", exp_name, "v3_preprocess.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if v3_source_dir and v3_source_dir.strip():
        src = v3_source_dir.strip()
        cmd = (
            '"%s" tools/cmd/yingmusic_experiment.py'
            ' --source-dir "%s" --output-dir "%s" --setup-only'
            % (config.python_cmd, src, str(isolated_dir))
        )
        yield "V3 Preprocess: checking YingMusic environment...\n"
        yield "Source dir:   %s\n" % src
        yield "Output dir:   %s\n" % str(isolated_dir)
    else:
        cmd = (
            '"%s" tools/cmd/yingmusic_experiment.py --setup-only'
            % config.python_cmd
        )
        yield "V3 Preprocess: checking YingMusic environment (no source dir specified)...\n"

    logger.info("V3 preprocess execute: " + cmd)
    log_f = open(log_path, "w", encoding="utf-8")
    p = Popen(cmd, shell=True, cwd=now_dir, stdout=log_f, stderr=subprocess.STDOUT)
    done = [False]
    threading.Thread(target=if_done, args=(done, p)).start()

    while True:
        log_f.flush()
        with open(log_path, "r", encoding="utf-8", errors="ignore") as rf:
            yield rf.read()
        sleep(1)
        if done[0]:
            break

    log_f.close()
    with open(log_path, "r", encoding="utf-8", errors="ignore") as rf:
        log = rf.read()
    logger.info(log)
    rc = p.poll()
    if rc == 0:
        yield log + "\n[V3 Preprocess] Environment check passed."
    elif rc == 2:
        yield log + "\n[V3 Preprocess] CUDA not available on this host. " \
            "Run on a CUDA machine or use the isolated vocals directory manually."
    else:
        yield log + "\n[V3 Preprocess] Finished (exit code %s)." % rc


def click_train_v3(
    exp_dir1,
    sr2,
    if_f0_3,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_every_weights18,
    gpus16,
    author,
    run_async=True,
):
    """Dispatch V3 training to the HQ-SVC training adapter.

    Reuses the existing training supervisor infrastructure (same log file,
    status file, and PID tracking) so V3 training is observable and stoppable
    via the same train-status / stop-training buttons.
    """
    exp_dir = pathlib.Path(now_dir, "logs", exp_dir1)
    exp_dir.mkdir(parents=True, exist_ok=True)
    artifacts = _training_artifacts(exp_dir1)

    existing_pid = _active_training_pid(exp_dir1)
    if existing_pid is not None:
        return (
            "V3 training is already running for experiment '%s' (PID %s). "
            % (exp_dir1, existing_pid)
            + "Check training status or wait for it to finish."
        )

    f0_flag = 1 if _is_f0_enabled(if_f0_3) else 0
    save_weights_flag = 1 if if_save_every_weights18 == i18n("Yes") else 0

    adapter_cmd = (
        '"%s" tools/cmd/hqsvc_train_adapter.py'
        ' --exp-dir "%s"'
        ' --sr %s'
        ' --f0 %s'
        ' --total-epoch %s'
        ' --save-epoch %s'
        ' --batch-size %s'
        ' --save-every-weights %s'
        ' --author "%s"'
        ' --status-file "%s"'
        % (
            config.python_cmd,
            exp_dir1,
            sr2,
            f0_flag,
            total_epoch11,
            save_epoch10,
            batch_size12,
            save_weights_flag,
            author,
            str(artifacts["status_file"]),
        )
    )
    if gpus16:
        adapter_cmd += ' --gpus "%s"' % gpus16

    supervisor_cmd = _build_supervisor_cmd(
        adapter_cmd,
        exp_dir1,
        artifacts["status_file"],
        artifacts["log_file"],
        artifacts["stop_request_file"],
        artifacts["stop_ack_file"],
    )

    # Clear stale stop markers from previous runs.
    for marker in [artifacts["stop_request_file"], artifacts["stop_ack_file"]]:
        if marker.exists():
            try:
                marker.unlink()
            except OSError:
                pass

    logger.info("V3 train execute: " + adapter_cmd)

    if run_async:
        with open(artifacts["log_file"], "a", encoding="utf-8") as train_log_file:
            p = Popen(
                supervisor_cmd,
                shell=True,
                cwd=now_dir,
                stdin=subprocess.DEVNULL,
                stdout=train_log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        TRAINING_PROCESSES[exp_dir1] = p
        artifacts["pid_file"].write_text(str(p.pid), encoding="utf-8")
        return (
            "V3 training adapter started in background (PID %s).\n"
            "The adapter validates prerequisites, checks HQ-SVC environment, and "
            "launches training if available.\n"
            "Logs: %s" % (p.pid, artifacts["log_file"])
        )

    rc = subprocess.run(supervisor_cmd, shell=True, cwd=now_dir).returncode
    final_msg = get_training_status(exp_dir1)
    if rc == 0:
        return "V3 training adapter complete.\n" + final_msg
    return "V3 training adapter finished with errors.\n" + final_msg


with gr.Blocks(title="RVC WebUI") as app:
    gr.Markdown("## RVC WebUI")
    gr.Markdown(
        value=i18n(
            "This software is open source under the MIT license. The author does not have any control over the software. Users who use the software and distribute the sounds exported by the software are solely responsible. <br>If you do not agree with this clause, you cannot use or reference any codes and files within the software package. See the root directory <b>LICENSE</b> for details."
        )
    )
    with gr.Tabs():
        with gr.TabItem(i18n("Model Inference")):
            with gr.Row():
                sid0 = gr.Dropdown(
                    label=i18n("Inferencing voice"), choices=sorted(names)
                )
                with gr.Column():
                    refresh_button = gr.Button(
                        i18n("Refresh voice list and index path"), variant="primary"
                    )
                    clean_button = gr.Button(
                        i18n("Unload model to save GPU memory"), variant="primary"
                    )
                spk_item = gr.Slider(
                    minimum=0,
                    maximum=2333,
                    step=1,
                    label=i18n("Select Speaker/Singer ID"),
                    value=0,
                    visible=False,
                    interactive=True,
                )
                clean_button.click(
                    fn=clean, inputs=[], outputs=[sid0], api_name="infer_clean"
                )
            modelinfo = gr.Textbox(label=i18n("Model info"), max_lines=8)
            with gr.TabItem(i18n("Single inference")):
                with gr.Row():
                    with gr.Column():
                        vc_transform0 = gr.Number(
                            label=i18n(
                                "Transpose (integer, number of semitones, raise by an octave: 12, lower by an octave: -12)"
                            ),
                            value=0,
                        )
                        input_audio0 = gr.Audio(
                            label=i18n("The audio file to be processed"),
                            type="filepath",
                        )
                        file_index2 = gr.Dropdown(
                            label=i18n(
                                "Auto-detect index path and select from the dropdown"
                            ),
                            choices=sorted(index_paths),
                            interactive=True,
                        )
                        file_index1 = gr.File(
                            label=i18n(
                                "Path to the feature index file. Leave blank to use the selected result from the dropdown"
                            ),
                        )
                    with gr.Column():
                        f0method0 = gr.Radio(
                            label=i18n(
                                "Select the pitch extraction algorithm ('pm': faster extraction but lower-quality speech; 'harvest': better bass but extremely slow; 'crepe': better quality but GPU intensive), 'rmvpe': best quality, and little GPU requirement"
                            ),
                            choices=(
                                ["pm", "dio", "harvest", "crepe", "rmvpe", "fcpe"]
                            ),
                            value="rmvpe",
                            interactive=True,
                        )
                        resample_sr0 = gr.Slider(
                            minimum=0,
                            maximum=48000,
                            label=i18n(
                                "Resample the output audio in post-processing to the final sample rate. Set to 0 for no resampling"
                            ),
                            value=0,
                            step=1,
                            interactive=True,
                        )
                        rms_mix_rate0 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n(
                                "Adjust the volume envelope scaling. Closer to 0, the more it mimicks the volume of the original vocals. Can help mask noise and make volume sound more natural when set relatively low. Closer to 1 will be more of a consistently loud volume"
                            ),
                            value=0.35,
                            interactive=True,
                        )
                        protect0 = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            label=i18n(
                                "Protect voiceless consonants and breath sounds to prevent artifacts such as tearing in electronic music. Set to 0.5 to disable. Decrease the value to increase protection, but it may reduce indexing accuracy"
                            ),
                            value=0.25,
                            step=0.01,
                            interactive=True,
                        )
                        filter_radius0 = gr.Slider(
                            minimum=0,
                            maximum=7,
                            label=i18n(
                                "If >=3: apply median filtering to the harvested pitch results. The value represents the filter radius and can reduce breathiness."
                            ),
                            value=4,
                            step=1,
                            interactive=True,
                        )
                        index_rate1 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("Feature searching ratio"),
                            value=0.75,
                            interactive=True,
                        )
                        f0_file = gr.File(
                            label=i18n(
                                "F0 curve file (optional). One pitch per line. Replaces the default F0 and pitch modulation"
                            ),
                            visible=False,
                        )
                        but0 = gr.Button(i18n("Convert"), variant="primary")
                        vc_output2 = gr.Audio(
                            label=i18n(
                                "Export audio (click on the three dots in the lower right corner to download)"
                            )
                        )

                        refresh_button.click(
                            fn=change_choices,
                            inputs=[],
                            outputs=[sid0, file_index2],
                            api_name="infer_refresh",
                        )

                vc_output1 = gr.Textbox(label=i18n("Output information"))

                but0.click(
                    vc.vc_single,
                    [
                        spk_item,
                        input_audio0,
                        vc_transform0,
                        f0_file,
                        f0method0,
                        file_index1,
                        file_index2,
                        # file_big_npy1,
                        index_rate1,
                        filter_radius0,
                        resample_sr0,
                        rms_mix_rate0,
                        protect0,
                    ],
                    [vc_output1, vc_output2],
                    api_name="infer_convert",
                )
            with gr.TabItem(i18n("Batch inference")):
                gr.Markdown(
                    value=i18n(
                        "Batch conversion. Enter the folder containing the audio files to be converted or upload multiple audio files. The converted audio will be output in the specified folder (default: 'opt')."
                    )
                )
                with gr.Row():
                    with gr.Column():
                        vc_transform1 = gr.Number(
                            label=i18n(
                                "Transpose (integer, number of semitones, raise by an octave: 12, lower by an octave: -12)"
                            ),
                            value=0,
                        )
                        dir_input = gr.Textbox(
                            label=i18n(
                                "Enter the path of the audio folder to be processed (copy it from the address bar of the file manager)"
                            ),
                            placeholder="C:\\Users\\Desktop\\input_vocal_dir",
                        )
                        inputs = gr.File(
                            file_count="multiple",
                            label=i18n(
                                "Multiple audio files can also be imported. If a folder path exists, this input is ignored."
                            ),
                        )
                        opt_input = gr.Textbox(
                            label=i18n("Specify output folder"), value="opt"
                        )
                        file_index4 = gr.Dropdown(
                            label=i18n(
                                "Auto-detect index path and select from the dropdown"
                            ),
                            choices=sorted(index_paths),
                            interactive=True,
                        )
                        file_index3 = gr.File(
                            label=i18n(
                                "Path to the feature index file. Leave blank to use the selected result from the dropdown"
                            ),
                        )

                        refresh_button.click(
                            fn=lambda: change_choices()[1],
                            inputs=[],
                            outputs=file_index4,
                            api_name="infer_refresh_batch",
                        )
                        # file_big_npy2 = gr.Textbox(
                        #     label=i18n("特征文件路径"),
                        #     value="E:\\codes\\py39\\vits_vc_gpu_train\\logs\\mi-test-1key\\total_fea.npy",
                        #     interactive=True,
                        # )

                    with gr.Column():
                        f0method1 = gr.Radio(
                            label=i18n(
                                "Select the pitch extraction algorithm ('pm': faster extraction but lower-quality speech; 'harvest': better bass but extremely slow; 'crepe': better quality but GPU intensive), 'rmvpe': best quality, and little GPU requirement"
                            ),
                            choices=(
                                ["pm", "dio", "harvest", "crepe", "rmvpe", "fcpe"]
                            ),
                            value="rmvpe",
                            interactive=True,
                        )
                        resample_sr1 = gr.Slider(
                            minimum=0,
                            maximum=48000,
                            label=i18n(
                                "Resample the output audio in post-processing to the final sample rate. Set to 0 for no resampling"
                            ),
                            value=0,
                            step=1,
                            interactive=True,
                        )
                        rms_mix_rate1 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n(
                                "Adjust the volume envelope scaling. Closer to 0, the more it mimicks the volume of the original vocals. Can help mask noise and make volume sound more natural when set relatively low. Closer to 1 will be more of a consistently loud volume"
                            ),
                            value=0.35,
                            interactive=True,
                        )
                        protect1 = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            label=i18n(
                                "Protect voiceless consonants and breath sounds to prevent artifacts such as tearing in electronic music. Set to 0.5 to disable. Decrease the value to increase protection, but it may reduce indexing accuracy"
                            ),
                            value=0.25,
                            step=0.01,
                            interactive=True,
                        )
                        filter_radius1 = gr.Slider(
                            minimum=0,
                            maximum=7,
                            label=i18n(
                                "If >=3: apply median filtering to the harvested pitch results. The value represents the filter radius and can reduce breathiness."
                            ),
                            value=4,
                            step=1,
                            interactive=True,
                        )
                        index_rate2 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("Feature searching ratio"),
                            value=1,
                            interactive=True,
                        )
                        format1 = gr.Radio(
                            label=i18n("Export file format"),
                            choices=["wav", "flac", "mp3", "m4a"],
                            value="wav",
                            interactive=True,
                        )
                        but1 = gr.Button(i18n("Convert"), variant="primary")
                        vc_output3 = gr.Textbox(label=i18n("Output information"))

                but1.click(
                    vc.vc_multi,
                    [
                        spk_item,
                        dir_input,
                        opt_input,
                        inputs,
                        vc_transform1,
                        f0method1,
                        file_index3,
                        file_index4,
                        # file_big_npy2,
                        index_rate2,
                        filter_radius1,
                        resample_sr1,
                        rms_mix_rate1,
                        protect1,
                        format1,
                    ],
                    [vc_output3],
                    api_name="infer_convert_batch",
                )
                sid0.change(
                    fn=vc.get_vc,
                    inputs=[sid0, protect0, protect1, file_index2, file_index4],
                    outputs=[
                        spk_item,
                        protect0,
                        protect1,
                        file_index2,
                        file_index4,
                        modelinfo,
                    ],
                    api_name="infer_change_voice",
                )
        with gr.TabItem(
            i18n("Vocals/Accompaniment Separation & Reverberation Removal")
        ):
            gr.Markdown(
                value=i18n(
                    "Batch processing for vocal accompaniment separation using the UVR5 model.<br>Example of a valid folder path format: D:\\path\\to\\input\\folder (copy it from the file manager address bar).<br>The model is divided into three categories:<br>1. Preserve vocals: Choose this option for audio without harmonies. It preserves vocals better than HP5. It includes two built-in models: HP2 and HP3. HP3 may slightly leak accompaniment but preserves vocals slightly better than HP2.<br>2. Preserve main vocals only: Choose this option for audio with harmonies. It may weaken the main vocals. It includes one built-in model: HP5.<br>3. De-reverb and de-delay models (by FoxJoy):<br>  (1) MDX-Net: The best choice for stereo reverb removal but cannot remove mono reverb;<br>&emsp;(234) DeEcho: Removes delay effects. Aggressive mode removes more thoroughly than Normal mode. DeReverb additionally removes reverb and can remove mono reverb, but not very effectively for heavily reverberated high-frequency content.<br>De-reverb/de-delay notes:<br>1. The processing time for the DeEcho-DeReverb model is approximately twice as long as the other two DeEcho models.<br>2. The MDX-Net-Dereverb model is quite slow.<br>3. The recommended cleanest configuration is to apply MDX-Net first and then DeEcho-Aggressive."
                )
            )
            with gr.Row():
                with gr.Column():
                    dir_wav_input = gr.Textbox(
                        label=i18n(
                            "Enter the path of the audio folder to be processed"
                        ),
                        placeholder="C:\\Users\\Desktop\\todo-songs",
                    )
                    wav_inputs = gr.File(
                        file_count="multiple",
                        label=i18n(
                            "Multiple audio files can also be imported. If a folder path exists, this input is ignored."
                        ),
                    )
                with gr.Column():
                    model_choose = gr.Dropdown(label=i18n("Model"), choices=uvr5_names)
                    agg = gr.Slider(
                        minimum=0,
                        maximum=20,
                        step=1,
                        label="人声提取激进程度",
                        value=10,
                        interactive=True,
                        visible=False,  # 先不开放调整
                    )
                    opt_vocal_root = gr.Textbox(
                        label=i18n("Specify the output folder for vocals"),
                        value="opt",
                    )
                    opt_ins_root = gr.Textbox(
                        label=i18n("Specify the output folder for accompaniment"),
                        value="opt",
                    )
                    format0 = gr.Radio(
                        label=i18n("Export file format"),
                        choices=["wav", "flac", "mp3", "m4a"],
                        value="flac",
                        interactive=True,
                    )
                but2 = gr.Button(i18n("Convert"), variant="primary")
                vc_output4 = gr.Textbox(label=i18n("Output information"))
                but2.click(
                    uvr,
                    [
                        model_choose,
                        dir_wav_input,
                        opt_vocal_root,
                        wav_inputs,
                        opt_ins_root,
                        agg,
                        format0,
                    ],
                    [vc_output4],
                    api_name="uvr_convert",
                )
        with gr.TabItem(i18n("Train")):
            gr.Markdown(
                value=i18n(
                    "### Step 1. Fill in the experimental configuration.\nExperimental data is stored in the 'logs' folder, with each experiment having a separate folder. Manually enter the experiment name path, which contains the experimental configuration, logs, and trained model files."
                )
            )
            with gr.Row():
                exp_dir1 = gr.Textbox(
                    label=i18n("Enter the experiment name"), value=default_exp_dir
                )
                author = gr.Textbox(label=i18n("Model Author (Nullable)"))
                np7 = gr.Slider(
                    minimum=0,
                    maximum=config.n_cpu,
                    step=1,
                    label=i18n(
                        "Number of CPU processes used for pitch extraction and data processing"
                    ),
                    value=int(np.ceil(config.n_cpu / 1.5)),
                    interactive=True,
                )
            with gr.Row():
                detect_resume_btn = gr.Button(
                    i18n("Detect Existing Progress"),
                    variant="secondary",
                )
                auto_resume8 = gr.Checkbox(
                    label=i18n("Auto-resume one-click training (skip completed Step 1/2)"),
                    value=True,
                    interactive=True,
                )
            with gr.Row():
                clear_step1_ck = gr.Checkbox(
                    label=i18n("Clear Step 1 outputs (0_gt_wavs, 1_16k_wavs, preprocess.log)"),
                    value=False,
                    interactive=True,
                )
                clear_step2_ck = gr.Checkbox(
                    label=i18n("Clear Step 2 outputs (2a_f0, 2b-f0nsf, 3_feature*, extract log)"),
                    value=True,
                    interactive=True,
                )
                clear_resume_btn = gr.Button(
                    i18n("Clear Selected Artifacts"),
                    variant="stop",
                )
            resume_info8 = gr.Textbox(
                label=i18n("Detected progress"),
                value=inspect_experiment_progress(default_exp_dir, i18n("Yes"), "v2"),
                lines=8,
            )
            with gr.Row():
                sr2 = gr.Radio(
                    label=i18n("Target sample rate"),
                    choices=["32k", "40k", "48k"],
                    value="48k",
                    interactive=True,
                )
                if_f0_3 = gr.Radio(
                    label=i18n(
                        "Whether the model has pitch guidance (required for singing, optional for speech)"
                    ),
                    choices=[i18n("Yes"), i18n("No")],
                    value=i18n("Yes"),
                    interactive=True,
                )
                version19 = gr.Radio(
                    label=i18n("Version"),
                    choices=["v1", "v2", "v3"],
                    value="v3",
                    interactive=True,
                    visible=True,
                )
            # V3 preprocess section — visible only when V3 is selected.
            with gr.Row(visible=True) as v3_preprocess_row:
                gr.Markdown(
                    value=i18n(
                        "### V3 Preprocess (YingMusic Vocal Isolation)\n"
                        "V3 training benefits from isolated vocals. "
                        "Use this section to check the YingMusic environment and optionally "
                        "run batch vocal isolation before dataset preparation.\n"
                        "On macOS/CPU hosts, this runs a setup-only check and shows "
                        "readiness status. Actual isolation requires a CUDA host."
                    )
                )
                with gr.Column():
                    v3_source_dir4 = gr.Textbox(
                        label=i18n(
                            "V3 Source audio directory (songs/raw vocals to isolate)"
                        ),
                        placeholder=i18n("Leave empty to only check environment readiness"),
                    )
                    but_v3_preprocess = gr.Button(
                        i18n("Check / Run V3 Vocal Isolation (YingMusic)"),
                        variant="secondary",
                    )
                with gr.Column():
                    info_v3_preprocess = gr.Textbox(
                        label=i18n("V3 Preprocess status"),
                        value="",
                        lines=6,
                    )
                    but_v3_preprocess.click(
                        v3_preprocess_dataset,
                        [v3_source_dir4, exp_dir1],
                        [info_v3_preprocess],
                        api_name="v3_preprocess",
                    )
            gr.Markdown(
                value=i18n(
                    "### Step 2. Audio processing. \n#### 1. Slicing.\nAutomatically traverse all files in the training folder that can be decoded into audio and perform slice normalization. Generates 2 wav folders in the experiment directory. Currently, only single-singer/speaker training is supported."
                )
            )
            with gr.Row():
                with gr.Column():
                    trainset_dir4 = gr.Textbox(
                        label=i18n("Enter the path of the training folder"),
                    )
                    preprocess_mode8 = gr.Radio(
                        label=i18n("Step 1 mode"),
                        choices=[
                            i18n("Incremental (skip existing files)"),
                            i18n("Full rebuild (reprocess all files)"),
                        ],
                        value=i18n("Incremental (skip existing files)"),
                        interactive=True,
                    )
                    spk_id5 = gr.Slider(
                        minimum=0,
                        maximum=4,
                        step=1,
                        label=i18n("Please specify the speaker/singer ID"),
                        value=0,
                        interactive=True,
                    )
                    but1 = gr.Button(i18n("Process data"), variant="primary")
                with gr.Column():
                    info1 = gr.Textbox(label=i18n("Output information"), value="")
                    but1.click(
                        preprocess_dataset,
                        [trainset_dir4, exp_dir1, sr2, np7, preprocess_mode8],
                        [info1],
                        api_name="train_preprocess",
                    )
            gr.Markdown(
                value=i18n(
                    "#### 2. Feature extraction.\nUse CPU to extract pitch (if the model has pitch), use GPU to extract features (select GPU index)."
                )
            )
            with gr.Row():
                with gr.Column():
                    gpu_info9 = gr.Textbox(
                        label=i18n("GPU Information"),
                        value=gpu_info,
                    )
                    f0method8 = gr.Radio(
                        label=i18n(
                            "Select the pitch extraction algorithm: when extracting singing, you can use 'pm' to speed up. For high-quality speech with fast performance, but worse CPU usage, you can use 'dio'. 'harvest' results in better quality but is slower.  'rmvpe' has the best results and consumes less CPU/GPU"
                        ),
                        choices=[
                            "pm",
                            "dio",
                            "harvest",
                            "crepe",
                            "rmvpe",
                            "rmvpe_gpu",
                            "fcpe",
                        ],
                        value="rmvpe",
                        interactive=True,
                    )
                    rmvpe_workers8 = gr.Textbox(
                        label=i18n(
                            "RMVPE worker devices (for rmvpe_gpu), separated by '-' or ','. Example: mps or 0-1"
                        ),
                        value="mps" if sys.platform == "darwin" else gpus,
                        interactive=True,
                        visible=False,
                    )
                with gr.Column():
                    but2 = gr.Button(i18n("Feature extraction"), variant="primary")
                    info2 = gr.Textbox(label=i18n("Output information"), value="")
                f0method8.change(change_f0_method, [f0method8], [rmvpe_workers8])
                but2.click(
                    extract_f0_feature,
                    [
                        np7,
                        f0method8,
                        if_f0_3,
                        exp_dir1,
                        version19,
                        rmvpe_workers8,
                    ],
                    [info2],
                    api_name="train_extract_f0_feature",
                )
            gr.Markdown(
                value=i18n(
                    "### Step 3. Start training.\nFill in the training settings and start training the model and index."
                )
            )
            with gr.Row():
                with gr.Column():
                    save_epoch10 = gr.Slider(
                        minimum=1,
                        maximum=50,
                        step=1,
                        label=i18n("Save frequency (save_every_epoch)"),
                        value=20,
                        interactive=True,
                    )
                    total_epoch11 = gr.Slider(
                        minimum=2,
                        maximum=1000,
                        step=1,
                        label=i18n("Total training epochs (total_epoch)"),
                        value=600,
                        interactive=True,
                    )
                    batch_size12 = gr.Slider(
                        minimum=1,
                        maximum=40,
                        step=1,
                        label=i18n("Batch size per GPU"),
                        value=default_batch_size,
                        interactive=True,
                    )
                    if_save_latest13 = gr.Radio(
                        label=i18n(
                            "Save only the latest '.ckpt' file to save disk space"
                        ),
                        choices=[i18n("Yes"), i18n("No")],
                        value=i18n("No"),
                        interactive=True,
                    )
                    if_cache_gpu17 = gr.Radio(
                        label=i18n(
                            "Cache all training sets to GPU memory. Caching small datasets (less than 10 minutes) can speed up training, but caching large datasets will consume a lot of GPU memory and may not provide much speed improvement"
                        ),
                        choices=[i18n("Yes"), i18n("No")],
                        value=i18n("No"),
                        interactive=True,
                    )
                    if_save_every_weights18 = gr.Radio(
                        label=i18n(
                            "Save a small final model to the 'weights' folder at each save point"
                        ),
                        choices=[i18n("Yes"), i18n("No")],
                        value=i18n("Yes"),
                        interactive=True,
                    )
                with gr.Column():
                    pretrained_G14 = gr.Textbox(
                        label=i18n("Load pre-trained base model G path"),
                        value="assets/pretrained_v2/f0G48k.pth",
                        interactive=True,
                    )
                    pretrained_D15 = gr.Textbox(
                        label=i18n("Load pre-trained base model D path"),
                        value="assets/pretrained_v2/f0D48k.pth",
                        interactive=True,
                    )
                    gpus16 = gr.Textbox(
                        label=i18n(
                            "Enter the GPU index(es) separated by '-', e.g., 0-1-2 to use GPU 0, 1, and 2"
                        ),
                        value=gpus,
                        interactive=True,
                    )
                    sr2.change(
                        change_sr2,
                        [sr2, if_f0_3, version19],
                        [pretrained_G14, pretrained_D15],
                    )
                    version19.change(
                        change_version19,
                        [sr2, if_f0_3, version19],
                        [pretrained_G14, pretrained_D15, sr2, v3_preprocess_row],
                    )
                    if_f0_3.change(
                        change_f0,
                        [if_f0_3, sr2, version19],
                        [f0method8, pretrained_G14, pretrained_D15],
                    )

                    but3 = gr.Button(i18n("Train model"), variant="primary")
                    but4 = gr.Button(i18n("Train feature index"), variant="primary")
                    but5 = gr.Button(i18n("One-click training"), variant="primary")
                    but_train_status = gr.Button(i18n("Train status"), variant="secondary")
                    but_stop_train = gr.Button(
                        i18n("Stop training after current epoch"),
                        variant="secondary",
                    )
            with gr.Row():
                info3 = gr.Textbox(label=i18n("Output information"), value="")
                but3.click(
                    click_train,
                    [
                        exp_dir1,
                        sr2,
                        if_f0_3,
                        spk_id5,
                        save_epoch10,
                        total_epoch11,
                        batch_size12,
                        if_save_latest13,
                        pretrained_G14,
                        pretrained_D15,
                        gpus16,
                        if_cache_gpu17,
                        if_save_every_weights18,
                        version19,
                        author,
                    ],
                    info3,
                    api_name="train_start",
                )
                but4.click(train_index, [exp_dir1, version19], info3)
                but_train_status.click(
                    get_training_status,
                    [exp_dir1],
                    info3,
                    api_name="train_status",
                )
                but_stop_train.click(
                    request_graceful_training_stop,
                    [exp_dir1],
                    info3,
                    api_name="train_stop_graceful",
                )
                but5.click(
                    train1key,
                    [
                        exp_dir1,
                        sr2,
                        if_f0_3,
                        trainset_dir4,
                        preprocess_mode8,
                        spk_id5,
                        np7,
                        f0method8,
                        rmvpe_workers8,
                        auto_resume8,
                        save_epoch10,
                        total_epoch11,
                        batch_size12,
                        if_save_latest13,
                        pretrained_G14,
                        pretrained_D15,
                        gpus16,
                        if_cache_gpu17,
                        if_save_every_weights18,
                        version19,
                        author,
                    ],
                    info3,
                    api_name="train_start_all",
                )
                detect_resume_btn.click(
                    inspect_experiment_progress,
                    [exp_dir1, if_f0_3, version19],
                    [resume_info8],
                )
                exp_dir1.change(
                    inspect_experiment_progress,
                    [exp_dir1, if_f0_3, version19],
                    [resume_info8],
                )
                if_f0_3.change(
                    inspect_experiment_progress,
                    [exp_dir1, if_f0_3, version19],
                    [resume_info8],
                )
                version19.change(
                    inspect_experiment_progress,
                    [exp_dir1, if_f0_3, version19],
                    [resume_info8],
                )
                clear_resume_btn.click(
                    clear_experiment_artifacts,
                    [exp_dir1, if_f0_3, version19, clear_step1_ck, clear_step2_ck],
                    [resume_info8],
                )

        with gr.TabItem(i18n("ckpt Processing")):
            gr.Markdown(
                value=i18n(
                    "### Model comparison\n> You can get model ID (long) from `View model information` below.\n\nCalculate a similarity between two models."
                )
            )
            with gr.Row():
                with gr.Column():
                    id_a = gr.Textbox(label=i18n("ID of model A (long)"), value="")
                    id_b = gr.Textbox(label=i18n("ID of model B (long)"), value="")
                with gr.Column():
                    butmodelcmp = gr.Button(i18n("Calculate"), variant="primary")
                    infomodelcmp = gr.Textbox(
                        label=i18n("Similarity (from 0 to 1)"),
                        value="",
                        max_lines=1,
                    )
            butmodelcmp.click(
                hash_similarity,
                [
                    id_a,
                    id_b,
                ],
                infomodelcmp,
                api_name="ckpt_merge",
            )

            gr.Markdown(
                value=i18n("### Model fusion\nCan be used to test timbre fusion.")
            )
            with gr.Row():
                with gr.Column():
                    ckpt_a = gr.Textbox(
                        label=i18n("Path to Model A"), value="", interactive=True
                    )
                    ckpt_b = gr.Textbox(
                        label=i18n("Path to Model B"), value="", interactive=True
                    )
                    alpha_a = gr.Slider(
                        minimum=0,
                        maximum=1,
                        label=i18n("Weight (w) for Model A"),
                        value=0.5,
                        interactive=True,
                    )
                with gr.Column():
                    sr_ = gr.Radio(
                        label=i18n("Target sample rate"),
                        choices=["32k", "40k", "48k"],
                        value="48k",
                        interactive=True,
                    )
                    if_f0_ = gr.Radio(
                        label=i18n("Whether the model has pitch guidance"),
                        choices=[i18n("Yes"), i18n("No")],
                        value=i18n("Yes"),
                        interactive=True,
                    )
                    info__ = gr.Textbox(
                        label=i18n("Model information to be placed"),
                        value="",
                        max_lines=8,
                        interactive=True,
                    )
                with gr.Column():
                    name_to_save0 = gr.Textbox(
                        label=i18n("Saved model name (without extension)"),
                        value="",
                        max_lines=1,
                        interactive=True,
                    )
                    version_2 = gr.Radio(
                        label=i18n("Model architecture version"),
                        choices=["v1", "v2"],
                        value="v1",
                        interactive=True,
                    )
                    but6 = gr.Button(i18n("Fusion"), variant="primary")
            with gr.Row():
                info4 = gr.Textbox(label=i18n("Output information"), value="")
            but6.click(
                merge,
                [
                    ckpt_a,
                    ckpt_b,
                    alpha_a,
                    sr_,
                    if_f0_,
                    info__,
                    name_to_save0,
                    version_2,
                ],
                info4,
                api_name="ckpt_merge",
            )  # def merge(path1,path2,alpha1,sr,f0,info):

            gr.Markdown(
                value=i18n(
                    "### Modify model information\n> Only supported for small model files extracted from the 'weights' folder."
                )
            )
            with gr.Row():
                with gr.Column():
                    ckpt_path0 = gr.Textbox(
                        label=i18n("Path to Model"), value="", interactive=True
                    )
                    info_ = gr.Textbox(
                        label=i18n("Model information to be modified"),
                        value="",
                        max_lines=8,
                        interactive=True,
                    )
                    name_to_save1 = gr.Textbox(
                        label=i18n("Save file name (default: same as the source file)"),
                        value="",
                        max_lines=1,
                        interactive=True,
                    )
                with gr.Column():
                    but7 = gr.Button(i18n("Modify"), variant="primary")
                    info5 = gr.Textbox(label=i18n("Output information"), value="")
            but7.click(
                change_info,
                [ckpt_path0, info_, name_to_save1],
                info5,
                api_name="ckpt_modify",
            )

            gr.Markdown(
                value=i18n(
                    "### View model information\n> Only supported for small model files extracted from the 'weights' folder."
                )
            )
            with gr.Row():
                with gr.Column():
                    ckpt_path1 = gr.File(label=i18n("Path to Model"))
                    but8 = gr.Button(i18n("View"), variant="primary")
                with gr.Column():
                    info6 = gr.Textbox(label=i18n("Output information"), value="")
            but8.click(show_info, [ckpt_path1], info6, api_name="ckpt_show")

            gr.Markdown(
                value=i18n(
                    "### Model extraction\n> Enter the path of the large file model under the 'logs' folder.\n\nThis is useful if you want to stop training halfway and manually extract and save a small model file, or if you want to test an intermediate model."
                )
            )
            with gr.Row():
                with gr.Column():
                    ckpt_path2 = gr.Textbox(
                        label=i18n("Path to Model"),
                        value="E:\\codes\\py39\\logs\\mi-test_f0_48k\\G_23333.pth",
                        interactive=True,
                    )
                    save_name = gr.Textbox(
                        label=i18n("Save name"), value="", interactive=True
                    )
                    with gr.Row():
                        sr__ = gr.Radio(
                            label=i18n("Target sample rate"),
                            choices=["32k", "40k", "48k"],
                            value="48k",
                            interactive=True,
                        )
                        if_f0__ = gr.Radio(
                            label=i18n(
                                "Whether the model has pitch guidance (1: yes, 0: no)"
                            ),
                            choices=["1", "0"],
                            value="1",
                            interactive=True,
                        )
                        version_1 = gr.Radio(
                            label=i18n("Model architecture version"),
                            choices=["v1", "v2"],
                            value="v2",
                            interactive=True,
                        )
                    info___ = gr.Textbox(
                        label=i18n("Model information to be placed"),
                        value="",
                        max_lines=8,
                        interactive=True,
                    )
                    extauthor = gr.Textbox(
                        label=i18n("Model Author"),
                        value="",
                        max_lines=1,
                        interactive=True,
                    )
                with gr.Column():
                    but9 = gr.Button(i18n("Extract"), variant="primary")
                    info7 = gr.Textbox(label=i18n("Output information"), value="")
                    ckpt_path2.change(
                        change_info_, [ckpt_path2], [sr__, if_f0__, version_1]
                    )
            but9.click(
                extract_small_model,
                [
                    ckpt_path2,
                    save_name,
                    extauthor,
                    sr__,
                    if_f0__,
                    info___,
                    version_1,
                ],
                info7,
                api_name="ckpt_extract",
            )

        with gr.TabItem(i18n("Export Onnx")):
            with gr.Row():
                ckpt_dir = gr.Textbox(
                    label=i18n("RVC Model Path"), value="", interactive=True
                )
            with gr.Row():
                onnx_dir = gr.Textbox(
                    label=i18n("Onnx Export Path"), value="", interactive=True
                )
            with gr.Row():
                infoOnnx = gr.Label(label="info")
            with gr.Row():
                butOnnx = gr.Button(i18n("Export Onnx Model"), variant="primary")
            butOnnx.click(
                export_onnx, [ckpt_dir, onnx_dir], infoOnnx, api_name="export_onnx"
            )

        tab_faq = i18n("FAQ (Frequently Asked Questions)")
        with gr.TabItem(tab_faq):
            try:
                if tab_faq == "FAQ (Frequently Asked Questions)":
                    with open("docs/cn/faq.md", "r", encoding="utf8") as f:
                        info = f.read()
                else:
                    with open("docs/en/faq_en.md", "r", encoding="utf8") as f:
                        info = f.read()
                gr.Markdown(value=info)
            except:
                gr.Markdown(traceback.format_exc())

try:
    import signal

    def cleanup(signum, frame):
        signame = signal.Signals(signum).name
        print(f"Got signal {signame} ({signum})")
        app.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    if config.global_link:
        app.queue(max_size=1022).launch(share=True, max_threads=511)
    else:
        app.queue(max_size=1022).launch(
            max_threads=511,
            server_name="0.0.0.0",
            inbrowser=not config.noautoopen,
            server_port=config.listen_port,
            quiet=True,
        )
except Exception as e:
    logger.error(str(e))
