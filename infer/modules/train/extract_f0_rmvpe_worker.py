import os
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv

now_dir = os.getcwd()
sys.path.append(now_dir)
load_dotenv()
load_dotenv("sha256.env")

import logging

import numpy as np

from infer.lib.audio import load_audio
from rvc.f0 import Generator

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.pyplot").setLevel(logging.WARNING)

exp_dir = sys.argv[1]
n_parts = int(sys.argv[2])
part_idx = int(sys.argv[3])
device = sys.argv[4]
is_half = sys.argv[5] == "True"

f = open(f"{exp_dir}/extract_f0_feature.log", "a+")


def printt(s):
    print(s)
    f.write(f"{s}\n")
    f.flush()


class FeatureInput:
    def __init__(self, is_half: bool, device="cpu", samplerate=16000, hop_size=160):
        self.fs = samplerate
        self.hop = hop_size

        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

        self.f0_gen = Generator(
            Path(os.environ["rmvpe_root"]),
            is_half,
            0,
            device,
            hop_size,
            samplerate,
        )

    def go(self, paths):
        if len(paths) == 0:
            printt(f"rmvpe-worker-{part_idx}: no-f0-todo")
            return

        printt(f"rmvpe-worker-{part_idx}: todo-f0-{len(paths)} on {device}")
        n = max(len(paths) // 5, 1)
        for idx, (inp_path, opt_path1, opt_path2) in enumerate(paths):
            try:
                if idx % n == 0:
                    printt(
                        f"rmvpe-worker-{part_idx}: f0ing,now-{idx},all-{len(paths)},-{inp_path}"
                    )

                if os.path.exists(opt_path1 + ".npy") and os.path.exists(opt_path2 + ".npy"):
                    continue

                x = load_audio(inp_path, self.fs)
                coarse_pit, feature_pit = self.f0_gen.calculate(
                    x, x.shape[0] // self.hop, 0, "rmvpe", None
                )
                np.save(opt_path2, feature_pit, allow_pickle=False)
                np.save(opt_path1, coarse_pit, allow_pickle=False)
            except Exception:
                printt(
                    "rmvpe-worker-%s: f0fail-%s-%s-%s"
                    % (part_idx, idx, inp_path, traceback.format_exc())
                )


if __name__ == "__main__":
    printt(" ".join(sys.argv))

    paths = []
    inp_root = f"{exp_dir}/1_16k_wavs"
    opt_root1 = f"{exp_dir}/2a_f0"
    opt_root2 = f"{exp_dir}/2b-f0nsf"

    os.makedirs(opt_root1, exist_ok=True)
    os.makedirs(opt_root2, exist_ok=True)

    for name in sorted(list(os.listdir(inp_root))):
        inp_path = f"{inp_root}/{name}"
        if "spec" in inp_path:
            continue
        opt_path1 = f"{opt_root1}/{name}"
        opt_path2 = f"{opt_root2}/{name}"
        paths.append([inp_path, opt_path1, opt_path2])

    worker_paths = paths[part_idx::n_parts]
    feature_input = FeatureInput(is_half, device)
    feature_input.go(worker_paths)
