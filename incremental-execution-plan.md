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

## Phase 0: Planning And Baseline Capture

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

## Phase 1: Startup, Assets, And Device Reliability

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

## Phase 2: Test And CI Hardening

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

## Phase 3: Workflow And UX Improvements

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

## Phase 5: Research Track, Separate From Mainline

Objective:

- preserve worthwhile research directions from the prior report without allowing them to block nearer-term product work

Research candidates worth retaining:

- staged pretraining on mixed speech and singing data
- pitch augmentation and broader training-time data diversity
- stronger vocoder evaluation
- richer speaker representations to reduce leakage
- diffusion-based decoding
- codec-based or unified latent approaches for future major versions

Rules for entering active implementation:

- baseline startup and CI work from earlier phases is complete
- a benchmark harness exists for the relevant path
- the work can be isolated behind a flag, separate branch, or experimental script

## Prioritized Backlog

P0:

- audit and harden startup flow across `run.sh`, `web.py`, and `configs/config.py`
- make asset and path failure modes explicit
- add launch-critical smoke coverage to CI

P1:

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
