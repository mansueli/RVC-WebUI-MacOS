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

## Phase 5: Experimental V3 Research Track ← ACTIVE FOCUS

Objective:

- ship a usable V3 workflow inside WebUI (not as external scripts)
- keep v1/v2 fully stable on the current RVC backend while V3 uses an HQ-SVC-oriented backend
- standardize V3 data flow around YingMusic preprocessing and a fixed 48k WebUI training target

Scope boundary:

- v3 in this fork is singing-focused by design
- speech-oriented conversion remains supported through v1/v2 paths and is not a v3 optimization target

Architecture rule for this phase:

- v1/v2 backend: existing RVC training/inference pipeline (unchanged by default)
- v3 backend: HQ-SVC-oriented training/inference pipeline, routed explicitly in WebUI
- no cross-coupling where v3 model loading depends on v1/v2 internals

Entry gate (must be complete before any active implementation here):

- a minimal benchmark harness exists for the relevant inference path (can be implemented now as a narrow pre-Phase-4 slice)
- each sub-track is isolated behind a flag, separate branch, or experimental script
- no v3 work lands in `main` until it passes the Phase 2 smoke suite

---

### V3 Implementation Plan (WebUI-Native)

Goal:

- make V3 fully executable from WebUI as a first-class workflow
- include YingMusic preprocessing inside the V3 path
- support HQ-SVC-style train/fine-tune/inference for V3
- preserve current v1/v2 behavior and compatibility

Core V3 pipeline:

```
User audio / dataset input
    -> V3 preprocess in WebUI (YingMusic vocal isolation)
    -> Dataset prep with V3 preset (fixed 48k, enforced resample/filter policy)
    -> V3 train/fine-tune backend (HQ-SVC-oriented)
    -> V3 inference backend (HQ-SVC-oriented)
```

### Lanes And Milestones

#### Lane A — Routing And UX Contracts (must land first)

Objective:

- make version selection deterministic in WebUI routing

Work items:

1. Keep `v1`/`v2` mapped to current RVC train/infer stack
2. Map `v3` to a dedicated backend path, not mixed with v1/v2 internals
3. Keep V3 sample rate fixed at 48k in UI and training command generation
4. Hide or disable incompatible RVC-only controls when V3 is active

Exit criteria:

- choosing `v3` always routes to V3 command builders
- choosing `v1`/`v2` remains behaviorally unchanged

#### Lane B — V3 Preprocess Integration (YingMusic baked in)

Objective:

- remove dependency on external manual preprocessing for V3

Work items:

1. Add a V3 preprocess action in WebUI that invokes YingMusic workflow
2. Persist isolated vocals in a deterministic experiment folder
3. Feed isolated outputs directly into dataset prep
4. Keep provenance manifest generation enabled by default
5. Enforce V3 preset in dataset prep (48k + resample + filter defaults)

Exit criteria:

- V3 preprocessing can be launched and monitored from WebUI
- processed files are ready for immediate V3 training without manual path juggling

#### Lane C — V3 Training Backend (HQ-SVC-oriented)

Objective:

- enable V3 fine-tuning flow on HQ-SVC-style runtime path

Work items:

1. Add a dedicated V3 training adapter module/command
2. Keep V3 checkpoint and log layout separate from v1/v2
3. Add resumable status and stop controls aligned with existing training supervisor behavior
4. Add preflight checks for required V3 assets/environment before train start

Exit criteria:

- WebUI can start/monitor/stop V3 training
- V3 training no longer depends on v1/v2-specific assumptions

#### Lane D — V3 Inference Backend (HQ-SVC-oriented)

Objective:

- ensure V3 models can be loaded and tested from WebUI after training

Work items:

1. Add V3 model load/infer route that does not rely on v1/v2-specific loaders
2. Keep backward compatibility in existing RVC loaders for v1/v2
3. Provide clear model info/status for V3 checkpoints in UI

Exit criteria:

- V3 checkpoints can be selected and inferred directly in WebUI
- v1/v2 inference remains unaffected

#### Lane E — Assets, Validation, And Rollout

Objective:

- make V3 operationally reliable, not just functionally present

Work items:

1. Add V3 asset readiness checks and optional download/bootstrap helpers
2. Add smoke tests for V3 routing, preprocess command construction, and backend dispatch
3. Keep full Phase 2 smoke suite green
4. Update docs for V3 flow and compatibility matrix

Exit criteria:

- V3 path is testable on clean setups with clear actionable errors
- CI catches V3 routing/preprocess regressions

### Progress update (implemented in this fork)

- Added `tools/cmd/hqsvc_training_simulation.py` to generate HQ-SVC-aligned simulation blueprints
- Added `--resample-if-needed` and `--v3-preset` in `training_data/prepare_rvc_dataset.py`
- Added V3 sample-rate enforcement behavior in WebUI (`v3` -> 48k only)
- Added V3 compatibility fallback in `rvc/synthesizer.py` for model-loading continuity

### Activation Constraints

- HQ-SVC full training integration depends on a stable, validated training path in the selected runtime environment
- local macOS remains first-class for orchestration/preprocess and lightweight validation; heavier train runs may require dedicated compute depending on backend requirements

### Phase 5 Definition Of Done

All items below must be true:

1. V3 preprocess, train/fine-tune, and inference are all executable from WebUI
2. YingMusic preprocessing is part of V3 workflow (not an external manual-only requirement)
3. V3 uses fixed 48k training target and explicit data policy
4. v1/v2 continue to use existing RVC backend without regressions
5. V3 smoke coverage exists and Phase 2 baseline tests still pass

### Phase 5 Execution Checklist (Implementation-Ready)

This checklist translates Lanes A-E into concrete file edits and acceptance tests.

#### Lane A Checklist — Routing And UX Contracts

File-level tasks:

1. `web.py`
   - centralize version routing into explicit helpers (`v1|v2 -> rvc`, `v3 -> hqsvc`)
   - enforce V3 sample-rate lock at all train entry points (train + one-click train)
   - hide/disable incompatible RVC-only controls when `version19 == "v3"`
2. `infer/modules/vc/modules.py`
   - keep current loader path as default for v1/v2
   - add version-aware dispatch surface for V3 inference backend calls

Acceptance tests:

1. Manual UI check: selecting V3 forces 48k and disables incompatible controls.
2. Manual UI check: selecting v1/v2 restores existing controls and behavior.
3. Smoke: existing `tests/test_smoke.py` still passes unchanged.

#### Lane B Checklist — V3 Preprocess Integration (YingMusic)

File-level tasks:

1. `web.py`
   - add V3 preprocess action/button and status output
   - wire V3 preprocess into one-click V3 flow before feature extraction/train
2. `tools/cmd/yingmusic_experiment.py`
   - expose deterministic output directory arguments for WebUI orchestration
   - provide clear setup-only and failure status messages consumable by WebUI logs
3. `training_data/prepare_rvc_dataset.py`
   - keep `--v3-preset` as canonical V3 preparation path
   - ensure provenance manifest includes source type and preprocess flags

Acceptance tests:

1. Manual command test: V3 preprocess writes isolated vocals and manifest in expected paths.
2. Manual UI test: V3 preprocess can be started from WebUI and logs progress/errors.
3. Smoke: V3 preprocess command construction test added and passing.

#### Lane C Checklist — V3 Training Backend (HQ-SVC-oriented)

File-level tasks:

1. `tools/cmd/hqsvc_experiment.py`
   - split setup vs train/fine-tune invocation surfaces
   - add stable return codes/messages for WebUI supervisor integration
2. `tools/cmd/` (new)
   - add `hqsvc_train_adapter.py` to normalize train/fine-tune command contract for WebUI
3. `web.py`
   - add V3 train dispatch to adapter command (not RVC trainer)
   - reuse training supervisor/status files for V3 process lifecycle

Acceptance tests:

1. Manual command test: adapter validates required paths and emits actionable errors.
2. Manual UI test: V3 train start/status/stop works via existing supervisor UX.
3. Smoke: V3 train command builder test added and passing.

#### Lane D Checklist — V3 Inference Backend (HQ-SVC-oriented)

File-level tasks:

1. `infer/modules/vc/modules.py`
   - add V3 inference dispatch branch separated from RVC model loader path
   - report model metadata/backend in UI status for transparency
2. `rvc/synthesizer.py`
   - keep compatibility fallback only for RVC checkpoints
   - avoid coupling HQ-SVC runtime loading into RVC internals
3. `web.py`
   - version-aware inference routing (`v3 -> hqsvc inference adapter`)

Acceptance tests:

1. Manual inference test: V3 checkpoint can be selected and inferred from WebUI.
2. Manual inference test: v1/v2 checkpoints still infer exactly as before.
3. Smoke: loader/routing test verifies no v1/v2 regression.

#### Lane E Checklist — Assets, Validation, And Rollout

File-level tasks:

1. `tools/download_models.py`
   - add V3/YingMusic/HQ-SVC asset readiness helpers and download entries
2. `web.py`
   - add preflight checks before V3 preprocess/train/infer actions
   - provide clear missing-asset remediation text and setup shortcuts
3. `tests/`
   - add `test_v3_routing_smoke.py` for routing/preprocess/train command wiring
   - keep existing `test_smoke.py` green
4. `README.md` and `docs/en/`
   - add V3 quickstart, prerequisites, and compatibility matrix

Acceptance tests:

1. Local smoke: `pytest tests/test_smoke.py -v --tb=short` passes.
2. Local smoke: new V3 smoke suite passes on non-CUDA host with setup-only mode.
3. Manual clean-environment test: V3 preflight detects missing assets and guides recovery.

### Suggested Sequencing (PR-sized)

1. PR-1: Lane A routing + UX lock (no backend replacement yet).
2. PR-2: Lane B WebUI-native YingMusic preprocess + manifest plumbing.
3. PR-3: Lane C V3 train adapter + supervisor wiring.
4. PR-4: Lane D V3 inference adapter + model selection UX.
5. PR-5: Lane E assets/preflight/tests/docs and release checklist.

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
