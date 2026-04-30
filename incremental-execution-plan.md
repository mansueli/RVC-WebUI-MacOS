# Incremental Execution Plan

This document replaces the prior research-heavy roadmap with a repo-grounded execution plan.

It is designed for incremental delivery in this fork, with an emphasis on macOS reliability, narrow validation, and improvements that fit the current code layout before any major architectural rewrite.

## Goal

Improve this repository in stages without destabilizing the current WebUI and realtime workflow.

Primary objective:

- make startup, asset loading, inference, and training behavior reliable on this fork
- add enough automated validation to refactor safely
- improve operator experience before taking on high-cost research work

## Current Baseline

These are confirmed from the current workspace and should be treated as the starting point for all planning.

- Main WebUI entrypoint is `web.py`
- Realtime GUI entrypoint is `gui.py`
- Runtime and device selection are centralized in `configs/config.py`
- Training scripts live under `infer/modules/train/`
- Voice conversion code lives under `infer/modules/vc/`
- Asset checks and downloads already exist via `infer/lib/rvcmd.py` and `tools/download_models.py`
- Docker support already exists via `Dockerfile`, `docker-compose.yml`, and `.github/workflows/docker.yml`
- CI already exists via `.github/workflows/unitest.yml`, but coverage is narrow and should be strengthened rather than created from scratch
- The repo already has multilingual documentation in `docs/`

## Planning Rules

The backlog should be executed with the following constraints.

1. Prefer fixes to the current architecture over large refactors.
2. Add narrow validation before widening scope.
3. Prioritize reliability and observability over speculative model work.
4. Treat macOS behavior as a first-class target in this fork.
5. Keep research work isolated from mainline maintenance until baseline stability is established.

## Phase 0: Planning And Baseline Capture ✅ DONE

Objective:

- convert vague future work into a trackable backlog with concrete owners of change and validation commands

Work items:

- keep this document as the single planning source of truth
- capture a known-good startup path on macOS using `run.sh` and `python web.py`
- record one known-good inference smoke path and one known-good preprocessing or training smoke path
- identify the current failure modes around missing assets, missing indices, unsupported Python versions, and device fallback behavior

Primary files:

- `incremental-execution-plan.md`
- `run.sh`
- `README.md`
- `web.py`
- `configs/config.py`

Exit criteria:

- the repo has one planning document instead of a speculative research report
- the basic operator flows and current risks are documented in actionable terms

## Phase 1: Startup, Assets, And Device Reliability ✅ DONE

**Completed changes:**
- `web.py`: added `logging.basicConfig(INFO)` so device/asset startup messages are visible; fixed silent asset-check failure (typo + added actionable `--update` guidance)
- `configs/config.py`: replaced misleading `"No supported Nvidia GPU found"` in the MPS path with `"Apple MPS detected, using MPS device (fp32)"` and clarified CPU fallback message
- `run.sh`: added Python version and startup banners (`Python X.Y.Z ready.` / `Starting RVC WebUI...`)
- `.github/workflows/unitest.yml`: restricted matrix to Python 3.10 only; added config-bootstrap smoke test step

Objective:

- make first launch and repeat launch predictable on macOS and acceptable on other supported environments

Scope:

- normalize startup behavior between `run.sh`, `web.py`, and `gui.py`
- harden asset presence checks and error messages
- reduce silent fallback behavior where the user cannot tell whether the app is using CUDA, MPS, or CPU
- make weights and index discovery more defensive when configured paths are missing or partially populated

Likely change surfaces:

- `run.sh`
- `web.py`
- `gui.py`
- `configs/config.py`
- `infer/lib/rvcmd.py`
- `tools/download_models.py`

Deliverables:

- consistent startup messages for Python version, device choice, and asset status
- clearer failure paths when models or indices are absent
- reduced environment-specific surprises on macOS MPS fallback

Validation:

- launch WebUI from `run.sh`
- launch WebUI from `python web.py`
- verify asset check-only flow
- verify startup with missing optional assets fails gracefully

Exit criteria:

- a new user can reach the UI with understandable status output
- missing resources produce clear action-oriented errors
- device selection is visible and matches actual runtime behavior

## Phase 2: Test And CI Hardening ✅ DONE

**Completed changes:**
- `tests/test_smoke.py`: new pytest suite — config bootstrap, audio utility correctness (float_to_int16, wav buffer), index path discovery (with/without matching index), and asset hash check (valid / missing / mismatch)
- `requirements/main.txt`: added `pytest>=7.0`
- `.github/workflows/unitest.yml`: replaced inline config-bootstrap assert with `pytest tests/test_smoke.py -v --tb=short`

Objective:

- extend the existing CI from partial script execution into repeatable smoke coverage for the current product surface

Scope:

- audit `.github/workflows/unitest.yml` for what it already proves and what it misses
- add a lightweight startup smoke test
- add narrow checks for config bootstrap, asset validation, and model or index discovery
- add at least one inference-adjacent sanity check that does not require a large full-fidelity model benchmark

Likely change surfaces:

- `.github/workflows/unitest.yml`
- test files to be added under a new or existing test directory
- `configs/config.py`
- `web.py`
- selected helpers under `infer/lib/` and `infer/modules/`

Deliverables:

- automated coverage for launch-critical behavior
- clearer failure diagnostics in CI
- a foundation for later refactors

Validation:

- run the narrow local test target added in this phase
- ensure the workflow still completes within a practical CI budget

Exit criteria:

- CI catches basic regressions in startup and preprocessing flow
- the team can change config or asset-loading code with lower breakage risk

## Phase 3: Workflow And UX Improvements ← NEXT

Objective:

- improve the operator experience without changing the core model architecture

Scope:

- improve model and index refresh behavior in the WebUI
- tighten progress, status, and error reporting during long-running actions
- make common user paths easier, especially training preparation and conversion execution
- identify whether batch conversion belongs in `web.py` now or should be shipped as a separate script first

Likely change surfaces:

- `web.py`
- `gui.py`
- `infer/modules/vc/`
- `infer/modules/train/`
- `README.md`
- selected docs under `docs/en/`

Deliverables:

- clearer UI feedback for long-running operations
- fewer manual refresh or restart steps when assets, weights, or indices change
- documentation aligned with actual current behavior

Validation:

- manual smoke test of the refreshed UI paths
- one documented end-to-end flow in the README or English docs

Exit criteria:

- users can understand what the app is doing without reading source
- common workflows require fewer restarts and less guesswork

## Phase 4: Performance And Quality Improvements Within The Current Architecture

Objective:

- improve output quality and runtime cost without jumping to a new model family

Scope:

- profile current inference hotspots
- investigate safer precision, export, or caching improvements
- tune retrieval behavior and index handling where there is a clear measurable win
- explore low-risk training data improvements such as pitch augmentation only after baseline tests exist

Likely change surfaces:

- `infer/modules/vc/pipeline.py`
- `infer/modules/vc/modules.py`
- `infer/modules/vc/rmvpe.py`
- `infer/modules/train/`
- `rvc/onnx/`
- `tools/cmd/onnx/`

Deliverables:

- benchmark notes for latency and memory on target environments
- one or two measurable wins that do not require architecture replacement

Validation:

- compare real-time factor or elapsed conversion time before and after
- verify no regression in the baseline smoke tests from Phase 2

Exit criteria:

- at least one performance or quality improvement is measurable and repeatable
- improvements can be reverted independently if they regress quality

## Phase 5: Experimental V3 Research Track

Objective:

- build a concrete, hardware-realistic path toward a v3 without blocking mainline work
- evaluate candidates from the 2025–2026 SVC landscape against the current RVC v2 baseline
- ship improvements incrementally for singing quality first: retrieval and fine-tuning efficiency first, decoder quality second

Scope boundary:

- v3 in this fork is singing-focused by design
- speech-oriented conversion remains supported through v1/v2 paths and is not a v3 optimization target

Entry gate (must be complete before any active implementation here):

- Phase 4 benchmark harness exists for the relevant inference path
- each sub-track is isolated behind a flag, separate branch, or experimental script
- no v3 work lands in `main` until it passes the Phase 2 smoke suite

---

### Candidate Evaluation Summary

Five candidates were evaluated against three criteria: integration effort on macOS consumer hardware, synergy with the existing RVC v2 codebase, and singing quality ceiling.

**kNN-SVC (SmoothKen, 2025)**

Directly evolves v2's Faiss top-1 retrieval. Adds additive harmonic synthesis and concatenation smoothness optimization, which removes the discontinuity artifacts that are v2's most common complaint. Zero-shot with pretrained checkpoints. Integration is surgical: replace the retrieval and post-processing logic in `infer/modules/vc/pipeline.py`, rebuild the Faiss index in the new format, keep RMVPE F0 and the rest of the WebUI unchanged. Lowest risk of all candidates.

Assessment: strongest candidate for the v3-core track. Measurable win, reversible, no new training pipeline required.

**LoRA-SVC / PlayVoice ecosystem (various, 2025)**

LoRA adapters on Whisper + BigVGAN/NSF-HiFiGAN backbone. Data preparation is identical to v2 (UVR + slicer). Fine-tuning runs on a single consumer GPU with under 20 minutes of target data. Can be applied on top of v2 weights or any of the other candidates as a base. Reduces VRAM and training time significantly.

Assessment: best near-term training UX improvement. Pairs naturally with kNN-SVC retrieval. Together they form the entire v3-core track.

**YingMusic-SVC (GiantAILab, Dec 2025)**

Flow-matching decoder + Flow-GRPO reinforcement learning fine-tuning + energy-balanced loss + singing-specific inductive biases. Highest singing quality ceiling of all candidates. Critically, the paper's pipeline explicitly uses an RVC timbre shifter as a preprocessing step, which means RVC v2 output can feed directly into it. Full code and HF checkpoints available. Multi-stage training is expensive (continuous pretrain → SFT → GRPO) but inference-only use is immediately viable.

Assessment: strongest decoder replacement candidate. Start inference-only to validate quality delta against v2. Only attempt fine-tuning after a benchmark harness and sufficient compute budget exist. This is the v3-experimental branch.

**CoMoSVC (Grace9994, 2025)**

Consistency-model-based SVC. 1–few step diffusion sampling at near-deterministic speed. Intended as a decoder speed optimization, not a base architecture. Requires a trained diffusion base to distill from (e.g., YingMusic or HQ-SVC).

Assessment: relevant only after Track B (YingMusic decoder) is validated. Not a starting point.

**HQ-SVC (ShawnPi233, AAAI 2026)**

Unified codec + EVA module + diffusion refinement. Designed for under 80 hours of training data on consumer hardware. Strong low-resource guarantees on paper. However, the codec + diffusion stack introduces too many new integration surfaces simultaneously for the current codebase state.

Assessment: revisit after Track B (YingMusic) completes its inference-only validation. Could become the low-resource fine-tuning path if YingMusic proves too expensive to fine-tune.

---

### Two-Track Implementation Plan

#### Track A — v3-core (target: ships alongside or after Phase 4)

Goal: a version of the current WebUI with measurably better retrieval quality and faster fine-tuning, no new decoder, no new training infrastructure.

Work items:

1. Fork `infer/modules/vc/pipeline.py` retrieval section into a replaceable interface
2. Implement kNN-SVC additive-synthesis retrieval + smoothness post-processing behind an `--retrieval-mode knn` flag
3. Measure RTF and MOS-proxy (UTMOS or similar) before and after on a held-out test set
4. Add LoRA adapter loading to the training pipeline in `infer/modules/train/`; expose in WebUI as an alternative to full fine-tune
5. Validate: existing `tests/test_smoke.py` must still pass; add one retrieval regression test

Change surfaces:

- `infer/modules/vc/pipeline.py`
- `infer/modules/train/` (LoRA adapter loading)
- `web.py` (retrieval mode selector, LoRA training path)
- `tests/test_smoke.py` (retrieval regression)

Exit criteria:

- kNN retrieval produces no audible discontinuities on the held-out set
- LoRA fine-tuning runs end-to-end on a single consumer GPU with under 20 min of target data
- no regression vs. v2 on the smoke suite

#### Track B — v3-experimental (separate branch: `exp/v3-yingmusic-singing`)

Goal: build a singing-optimized hybrid pipeline that uses YingMusic where it is strongest (accompaniment-robust preprocessing + flow decoder + singing-oriented timbre behavior), while keeping RVC's compatible pieces (HuBERT + RMVPE) to reduce integration risk.

Rationale for a singing-first hybrid rather than a voice-first split:

- this fork's v3 goal is singing quality, not general speech conversion
- YingMusic's singing inductive biases, energy-balanced loss behavior, and flow-matching decoder are aligned with the target domain
- keeping HuBERT and RMVPE preserves compatibility with current RVC data/training/inference tooling while still letting YingMusic drive the quality-critical synthesis path
- speech-specific identity optimization is explicitly deferred; users who need speech-first behavior should continue using v1/v2

Pipeline:

```
Raw input
	→ YingMusic vocal isolation  (preprocessing only, separate from inference graph)
	→ HuBERT               (content encoder, unchanged from v2)
	→ RMVPE                (F0, unchanged from v2)
	→ YingMusic timbre path (singing-oriented)
	→ YingMusic flow decoder  (fine-tuned on singing via LoRA)
	→ waveform
```

Work items:

1. Integrate YingMusic vocal isolation as an optional preprocessing step in `training_data/prepare_rvc_dataset.py` behind a `--isolate yingmusic` flag; measure training data SNR improvement on a held-out set of in-the-wild recordings
2. Wrap the YingMusic flow-matching decoder in `rvc/synthesizer.py` behind a `--decoder flow` flag; verify it accepts HuBERT content + RMVPE F0 + YingMusic timbre conditioning without architectural changes
3. Add LoRA adapters to the flow decoder; run a singing fine-tune pass on the existing singing-focused datasets; compare MOS-proxy (UTMOS or singing MOS proxy) and RTF vs. v2 HiFi-GAN baseline on MPS
4. Keep an optional fallback switch to v2 timbre conditioning for A/B tests (`--timbre-mode v2|yingmusic`) to reduce migration risk while benchmarks are collected
5. Run Phase 2 smoke suite on the branch; add one decoder smoke test (tensor shapes and dtype checks only, no large model download required)
6. If MOS-proxy improves and RTF on MPS is within 2× of v2, open a PR to merge the hybrid pipeline into main

Change surfaces (branch only until PR):

- `training_data/prepare_rvc_dataset.py` (vocal isolation flag)
- `infer/modules/vc/pipeline.py` (decoder abstraction, timbre-mode switch)
- `rvc/synthesizer.py` (flow decoder wrapper)
- `tests/` (decoder smoke test)
- `tools/download_models.py` (YingMusic assets/checkpoints entry)

Gate: Track B does not begin until Track A is complete and the Phase 4 benchmark harness exists.

---

### Research Candidates Deferred To Later

These remain worth watching but are not active plan items:

- CoMoSVC — consistency-model speed optimization on top of Track B's flow decoder; revisit after Track B validation produces a working decoder baseline to distill from
- HQ-SVC — low-resource codec + diffusion path; revisit if ECAPA + flow-decoder fine-tuning proves too expensive on consumer hardware; its <80h training guarantee is the main appeal
- Staged pretraining on mixed speech and singing data — longer-term; requires large compute budget and a benchmark harness before it is meaningful to measure
- Diffusion refinement as a post-processing pass — relevant only if Track B's flow decoder does not close the breathiness and high-frequency detail gap on its own

## Prioritized Backlog

P0: ✅ DONE

- ~~audit and harden startup flow across `run.sh`, `web.py`, and `configs/config.py`~~
- ~~make asset and path failure modes explicit~~
- ~~add launch-critical smoke coverage to CI~~

P1: ← NEXT

- improve model and index refresh behavior in the WebUI
- align README and English docs with the actual startup and asset flows
- add one narrow inference sanity test and one training-prep sanity test

P2:

- profile inference and identify low-risk optimization targets
- evaluate retrieval tuning and limited data augmentation behind a measurable benchmark

P3:

- begin experimental work on vocoder, pretraining, or diffusion only after the earlier phases are stable

## Definition Of Done For Each Increment

Each increment should satisfy all of the following.

- the change is scoped to one primary behavior slice
- there is a narrow validation command or smoke test
- docs are updated if user-facing behavior changes
- the change does not assume a future rewrite to be correct
- rollback is simple if the change regresses quality or compatibility

## Notes Preserved From The Prior Research Report

The previous document contained useful direction, but most of it belongs in a deferred research track rather than the main execution plan.

Keep these conclusions:

- better preprocessing and training data diversity are more realistic near-term wins than a full architecture replacement
- model quality improvements should be evaluated with both objective and subjective checks where practical
- diffusion, codec, and large-scale pretraining ideas are interesting but expensive and should be treated as experiments, not baseline roadmap commitments
- any major architecture work should come after stronger test coverage and clearer benchmarking

Do not carry forward these assumptions as active plan items:

- CI and Docker need to be created from scratch
- the current repo lacks multilingual docs
- a major modularization rewrite should happen before stabilization work
- the roadmap can assume unlimited compute or long speculative milestones without measurable gates
