# Experimental HQ-SVC V3 Pipeline (In-Repo)

This repository now includes an **experimental but runnable** V3 pipeline with CPU fallback:

1. Ying-style preprocessing
2. HQ-SVC-style training
3. Native V3 inference

## What is implemented

- CUDA path (preferred): `tools/cmd/yingmusic_experiment.py`
- CPU preprocess fallback: `tools/cmd/yingmusic_cpu_preprocess.py`
- External HQ-SVC training launcher: `tools/cmd/hqsvc_native_train.py`
- In-repo experimental trainer: `tools/cmd/hqsvc_local_train.py`
- External HQ-SVC native inference launcher: `tools/cmd/hqsvc_native_infer.py`
- In-repo experimental native inference: `tools/cmd/hqsvc_local_infer.py`
- One-command orchestrator: `tools/cmd/hqsvc_happy_path.py`

## Fastest happy path

Run from repository root:

```bash
python tools/cmd/hqsvc_happy_path.py \
  --exp-dir my_v3_exp \
  --source-dir /absolute/path/to/raw_audio \
  --isolated-output-dir logs/my_v3_exp/v3_isolated \
  --train-backend local_experimental \
  --device cpu \
  --steps 600 \
  --infer-source /absolute/path/to/test_input.wav \
  --infer-checkpoint logs/my_v3_exp/hqsvc_local/G_latest.pt \
  --infer-output logs/my_v3_exp/native_infer.wav
```

## Backend selection notes

- Training adapter (`tools/cmd/hqsvc_train_adapter.py`) supports:
  - `--backend external`
  - `--backend local_experimental`
  - `--backend auto` (external if available, else local_experimental)

- V3 inference in `infer/modules/vc/modules.py` is native-first.
  - Explicitly force old fallback only if needed:

```bash
export RVC_V3_BACKEND=fallback
```

## Dataset expectations

For training adapter flow, default dataset location is:

- `logs/<exp>/0_gt_wavs/*.wav`

For local experimental backend, `3_feature768` is not required.

## Important caveat

The in-repo local trainer is a **paper-inspired approximation** designed to provide a runnable development loop. It is not guaranteed to match upstream HQ-SVC quality or exact architecture.
