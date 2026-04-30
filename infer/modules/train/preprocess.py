import multiprocessing
import os
import re
import sys

from scipy import signal

now_dir = os.getcwd()
sys.path.append(now_dir)
print(*sys.argv[1:])
inp_root = sys.argv[1]
sr = int(sys.argv[2])
n_p = int(sys.argv[3])
exp_dir = sys.argv[4]
noparallel = sys.argv[5] == "True"
per = float(sys.argv[6])
# Optional argument: when True, re-process all files even if slices already exist.
force = len(sys.argv) > 7 and sys.argv[7] == "True"
device_name = sys.argv[8] if len(sys.argv) > 8 else "auto"
import os
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from infer.lib.audio import load_audio, save_audio
from infer.lib.slicer2 import Slicer

f = open("%s/preprocess.log" % exp_dir, "a+")

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")


def println(strr):
    print(strr)
    f.write("%s\n" % strr)
    f.flush()


def choose_device(name):
    if name == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class PreProcess:
    def __init__(self, sr, exp_dir, per=3.7, device="auto"):
        self.slicer = Slicer(
            sr=sr,
            threshold=-42,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500,
        )
        self.sr = sr
        self.bh, self.ah = signal.butter(N=5, Wn=48, btype="high", fs=self.sr)
        self.per = per
        self.overlap = 0.3
        self.tail = self.per + self.overlap
        self.max = 0.9
        self.alpha = 0.75
        self.device = choose_device(device)
        self.use_torch_accel = self.device.type in ("mps", "cuda")
        self.exp_dir = exp_dir
        self.gt_wavs_dir = "%s/0_gt_wavs" % exp_dir
        self.wavs16k_dir = "%s/1_16k_wavs" % exp_dir
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)
        self.existing_slice_bases = set()
        for name in os.listdir(self.gt_wavs_dir):
            m = re.match(r"^(.*)_\d+\.wav$", name)
            if m:
                self.existing_slice_bases.add(m.group(1))

    def has_existing_slices(self, base_name):
        return base_name in self.existing_slice_bases

    def _accelerated_audio_pair(self, tmp_audio):
        audio = torch.from_numpy(np.asarray(tmp_audio, dtype=np.float32)).to(self.device)
        tmp_max = torch.abs(audio).max()
        tmp_max_value = float(tmp_max.item()) if tmp_max.numel() else 0.0
        if tmp_max_value <= 0.0:
            zeros = np.zeros_like(tmp_audio, dtype=np.float32)
            target_len = max(1, int(round(len(tmp_audio) * 16000 / self.sr)))
            return zeros, np.zeros(target_len, dtype=np.float32), tmp_max_value
        audio = (audio / tmp_max * (self.max * self.alpha)) + (1 - self.alpha) * audio
        target_len = max(1, int(round(audio.shape[0] * 16000 / self.sr)))
        audio16k = F.interpolate(
            audio.view(1, 1, -1),
            size=target_len,
            mode="linear",
            align_corners=False,
        ).view(-1)
        return (
            audio.detach().cpu().numpy().astype(np.float32, copy=False),
            audio16k.detach().cpu().numpy().astype(np.float32, copy=False),
            tmp_max_value,
        )

    def norm_write(self, tmp_audio, idx0, idx1):
        if self.use_torch_accel:
            norm_audio, wav16k, tmp_max = self._accelerated_audio_pair(tmp_audio)
        else:
            tmp_max = np.abs(tmp_audio).max()
            norm_audio = None
            wav16k = None
        if tmp_max > 2.5:
            print("%s-%s-%s-filtered" % (idx0, idx1, tmp_max))
            return
        if norm_audio is None:
            if tmp_max <= 0:
                norm_audio = np.zeros_like(tmp_audio, dtype=np.float32)
            else:
                norm_audio = (tmp_audio / tmp_max * (self.max * self.alpha)) + (
                    1 - self.alpha
                ) * tmp_audio
        save_audio(
            "%s/%s_%s.wav" % (self.gt_wavs_dir, idx0, idx1),
            norm_audio,
            self.sr,
            f32=True,
        )
        if wav16k is None:
            wav16k = load_audio(
                "%s/%s_%s.wav" % (self.gt_wavs_dir, idx0, idx1),
                sr=16000,
            )
        save_audio(
            "%s/%s_%s.wav" % (self.wavs16k_dir, idx0, idx1),
            wav16k,
            16000,
            f32=True,
        )

    def pipeline(self, path):
        try:
            # Skip .DS_Store files
            if os.path.basename(path) == '.DS_Store':
                return None

            base_name = os.path.basename(path)
            if not force and self.has_existing_slices(base_name):
                println("%s\t-> Skipped (already processed)" % path)
                return None
            
            audio = load_audio(path, self.sr)
            if audio is None:
                return None
            
            # zero phased digital filter cause pre-ringing noise...
            # audio = signal.filtfilt(self.bh, self.ah, audio)
            audio = signal.lfilter(self.bh, self.ah, audio)

            idx1 = 0
            for audio in self.slicer.slice(audio):
                i = 0
                while 1:
                    start = int(self.sr * (self.per - self.overlap) * i)
                    i += 1
                    if len(audio[start:]) > self.tail * self.sr:
                        tmp_audio = audio[start : start + int(self.per * self.sr)]
                        #self.norm_write(tmp_audio, path, idx1)
                        self.norm_write(tmp_audio, base_name, idx1)
                        idx1 += 1
                    else:
                        tmp_audio = audio[start:]
                        idx1 += 1
                        break
                #self.norm_write(tmp_audio, path, idx1)
                self.norm_write(tmp_audio, base_name, idx1)
            self.existing_slice_bases.add(base_name)
            println("%s\t-> Success" % path)
            return audio
        except Exception as e:
            print(f"{path}\t-> {str(e)}")
            return None

    def pipeline_mp(self, infos):
        for path, idx0 in infos:
            self.pipeline(path)

    def pipeline_mp_inp_dir(self, inp_root, n_p):
        try:
            infos = [
                ("%s/%s" % (inp_root, name), name)
                for name in sorted(list(os.listdir(inp_root)))
            ]
            if self.use_torch_accel:
                println("Using %s-accelerated preprocess in single-process mode" % self.device)
                self.pipeline_mp(infos)
            elif noparallel:
                for i in range(n_p):
                    self.pipeline_mp(infos[i::n_p])
            else:
                ps = []
                for i in range(n_p):
                    p = multiprocessing.Process(
                        target=self.pipeline_mp, args=(infos[i::n_p],)
                    )
                    ps.append(p)
                    p.start()
                for i in range(n_p):
                    ps[i].join()
        except:
            println("Fail. %s" % traceback.format_exc())


def preprocess_trainset(inp_root, sr, n_p, exp_dir, per, device="auto"):
    pp = PreProcess(sr, exp_dir, per, device=device)
    println("start preprocess")
    println("preprocess device: %s" % pp.device)
    pp.pipeline_mp_inp_dir(inp_root, n_p)
    println("end preprocess")


if __name__ == "__main__":
    preprocess_trainset(inp_root, sr, n_p, exp_dir, per, device_name)
