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

## Phase 3: Workflow And UX Improvements ← SUPPORTING LANE

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

## Phase 4: Performance And Quality Improvements Within The Current Architecture ← DEFERRED UNTIL POST-PHASE 5

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

### Phase 5: Experimental V3 Research Track ← ACTIVE FOCUS (Updated with RVC Discriminator Design)

**Objective:**  
Ship a fully WebUI-native V3 workflow that implements the **HQ-SVC paper architecture (`FullHQSVC`) + RVC MultiPeriodDiscriminator adversarial fine-tuning** as the final polishing step, while keeping v1/v2 100 % stable.

User audio / dataset input
→ V3 preprocess in WebUI (YingMusic vocal isolation)
→ Dataset prep with V3 preset (fixed 48k, enforced resample/filter policy)
→ Stage 1: HQ-SVC paper training (FullHQSVC with FACodec/EVA/DDSP/diffusion + paper losses)
→ Stage 2: RVC MultiPeriodDiscriminator adversarial fine-tuning (generator + feature-matching loss)
→ V3 inference backend (loads fine-tuned checkpoint)

**Architecture rule (unchanged):**  
- v1/v2 backend: existing RVC training/inference pipeline (untouched)  
- v3 backend: dedicated HQ-SVC-oriented path using `FullHQSVC` + optional RVC fine-tune stage  
- No cross-coupling; v3 is explicitly routed

**New design principles added:**
- Stage 2 (RVC discriminator fine-tuning) is optional but recommended as the final quality step.
- `FullHQSVC` and `MultiPeriodDiscriminator` live in new dedicated modules under `tools/cmd/`.
- All MPS/Apple Silicon paths remain first-class and resumable at both stages.
- Stage 2 can be triggered via a **separate “RVC Fine-Tune (Stage 2)” button** in the WebUI (in addition to the combined two-stage flow).

#### Lane A — Routing And UX Contracts (must land first)

Work items:
1. Keep `v1`/`v2` → RVC, `v3` → HQ-SVC backend.
2. Add UI toggle / **separate button** “Enable RVC Discriminator Fine-Tune (Stage 2)” when `version19 == "v3"`.
3. Enforce 48k sample-rate lock for all V3 paths.

#### Lane B — V3 Preprocess Integration (YingMusic baked in)

*(unchanged — already aligns perfectly)*

#### Lane C — V3 Training Backend (HQ-SVC-oriented + RVC fine-tune) ← MAJOR UPDATE

Work items:
1. Create `tools/cmd/FullHQSVC.py` — the paper-aligned `FullHQSVC` class (FACodec frozen, EVA, DDSP, WaveNet denoiser, NSF-HiFiGAN).
2. Create `tools/cmd/rvc_discriminator.py` — exact RVC `MultiPeriodDiscriminator` + `PeriodDiscriminator`.
3. Extend `FullHQSVC` with `enable_rvc_fine_tune_mode()` and `compute_rvc_adv_losses()`.
4. Create `tools/cmd/hqsvc_train_adapter.py` — orchestrates **two-stage training**:
   - Stage 1: paper losses + AdamW (lr=1.5e-4)
   - Stage 2: low-lr adversarial + feature-matching (RVC disc) + weighted paper losses
5. Add `--stage 1|2` flag and support for the separate WebUI “RVC Fine-Tune” button.
6. Reuse existing training supervisor for start/stop/status.

#### Lane D — V3 Inference Backend (HQ-SVC-oriented)

Work items:
1. Add V3 dispatch in `infer/modules/vc/modules.py` that loads `FullHQSVC` checkpoint (Stage 2 fine-tuned weights preferred).
2. Update `rvc/synthesizer.py` with a non-intrusive V3 compatibility path.
3. Show “RVC Fine-Tuned” badge in UI when checkpoint contains discriminator state.

#### Lane E — Assets, Validation, And Rollout

Work items:
1. Extend `tools/download_models.py` with V3 assets (FACodec, RMVPE, optional NSF-HiFiGAN, YingMusic RoFormer checkpoint).
2. Add preflight checks for both training stages.
3. Add `tests/test_v3_full_pipeline_smoke.py` covering routing, preprocess, Stage 1 → Stage 2 hand-off, and separate fine-tune button.
4. Update `README.md` and `docs/en/` with V3 quickstart that mentions “RVC Discriminator Fine-Tuning (recommended final step via dedicated button)”.

### Phase 5 Definition Of Done

All items below must be true:

1. V3 preprocess, **Stage 1 (HQ-SVC paper training)**, **Stage 2 (RVC MultiPeriodDiscriminator adversarial fine-tuning)**, and inference are all executable from WebUI.  
   Stage 2 is available both as part of a combined two-stage flow **and** via a **separate “RVC Fine-Tune (Stage 2)” button** in the WebUI.
2. The `FullHQSVC` model fully implements the HQ-SVC paper architecture and supports seamless hand-off to the RVC discriminator for the final adversarial polishing step.
3. YingMusic preprocessing is part of the V3 workflow (not an external manual-only requirement).
4. V3 uses fixed 48k training target and explicit data policy.
5. v1/v2 continue to use existing RVC backend without any regressions.
6. V3 smoke coverage (including Stage 1 → Stage 2 hand-off **and** the separate fine-tune button) exists and Phase 2 baseline tests still pass.
7. RVC MultiPeriodDiscriminator fine-tuning is exposed as the recommended final polishing step, accessible via the dedicated UI button.

### Phase 5 Execution Checklist (Implementation-Ready — Updated)

#### Lane A Checklist — Routing And UX Contracts
- `web.py`: add V3 routing + “RVC Fine-Tune (Stage 2)” button/checkbox
- Acceptance: UI correctly exposes separate Stage 2 button and forces 48k

#### Lane B Checklist — V3 Preprocess Integration (YingMusic)
*(unchanged)*

#### Lane C Checklist — V3 Training Backend (HQ-SVC + RVC discriminator)
1. `tools/cmd/FullHQSVC.py` — full class + losses + RVC fine-tune methods
2. `tools/cmd/rvc_discriminator.py` — exact RVC MPD implementation
3. `tools/cmd/hqsvc_train_adapter.py` — two-stage orchestration + separate Stage 2 entry point
4. `web.py` — V3 train dispatch supporting both combined and separate Stage 2 button
**Acceptance tests:**
- Manual: Stage 1 runs paper losses; separate Stage 2 button runs adversarial + feature matching
- Smoke: `test_v3_training_stages.py` verifies both stages and the dedicated button

#### Lane D Checklist
*(unchanged except for “RVC Fine-Tuned” badge)*

#### Lane E Checklist
- `tests/test_v3_full_pipeline_smoke.py` (includes separate fine-tune button test)
- Docs updated with RVC fine-tune button instructions

### Suggested Sequencing (PR-sized — Updated)

1. PR-1: Lane A routing + separate “RVC Fine-Tune (Stage 2)” button  
2. PR-2: Lane B preprocess  
3. PR-3: Lane C — `FullHQSVC` + RVC discriminator + two-stage adapter  
4. PR-4: Lane D inference backend  
5. PR-5: Lane E assets/tests/docs + full smoke suite

### Per-PR Exit Gate

Each PR must satisfy:

1. no regression in Phase 2 smoke tests
2. lane-specific acceptance checks pass
3. rollback path documented in PR description

---

### Research Candidates Deferred To Later

These remain worth watching but are not active plan items:

- CoMoSVC — consistency-model speed optimization on top of Track B's flow decoder; revisit after Track B validation produces a working decoder baseline to distill from
- HQ-SVC — keep as v3 training-base target, but full adoption is currently blocked by upstream training-code availability; use pretrained inference path now and promote when training code is released
- Staged pretraining on mixed speech and singing data — longer-term; requires large compute budget and a benchmark harness before it is meaningful to measure
- Diffusion refinement as a post-processing pass — relevant only if Track B's flow decoder does not close the breathiness and high-frequency detail gap on its own

## Prioritized Backlog

P0: ✅ DONE

- ~~audit and harden startup flow across `run.sh`, `web.py`, and `configs/config.py`~~
- ~~make asset and path failure modes explicit~~
- ~~add launch-critical smoke coverage to CI~~

P1: ← ACTIVE NOW (Phase 5)

- implement V3 WebUI-native routing (v1/v2 on RVC, v3 on HQ-SVC-oriented backend)
- bake YingMusic preprocessing into V3 flow in WebUI and enforce 48k V3 data policy
- add V3 train/infer backend adapters plus asset readiness checks
- add minimal V3 smoke coverage while keeping Phase 2 baseline green

P2: ← SUPPORTING (Phase 3 workflow enhancements)

- improve model and index refresh behavior in the WebUI
- tighten progress, status, and error reporting for long-running training/conversion jobs
- align README and English docs with the actual startup, training, and asset flows

P3: ← DEFERRED RETURN (Phase 4)

- resume profiling and architecture-internal performance work after the current Phase 5 cycle and Phase 3 support work are complete
- evaluate retrieval tuning and low-risk data augmentation using the benchmark harness introduced in P1

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
