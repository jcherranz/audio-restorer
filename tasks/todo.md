# Task List

> Track current work items here. See `docs/WORKFLOW_ORCHESTRATION.md` for workflow.

## In Progress

<!-- Tasks currently being worked on -->
<!-- None -->

## Pending

<!-- Tasks queued for implementation -->
- [ ] Improve noisy-source SIG (cglDoG0GzyA: 2.96, FGDqFZZabsY: 2.33) — source material ceiling
  - Requires generative model (speech super-resolution) or better source recordings
  - Not actionable with current signal processing tools

## Completed

<!-- Move completed tasks here with date -->
- [x] 2026-02-06: Iteration 36 — Dereverberation DNSMOS benchmark (Phase 5 closed)
  - WPE post-enhancement: mean ΔOVRL=-1.30, all 6 tests degraded
  - WPE pre-enhancement: ΔOVRL=-0.10, SIG still hurt
  - DeepFilterNet handles reverb implicitly — separate dereverb is harmful
  - `--dereverb` flag retained but `_run_stage()` quality check auto-skips it
- [x] 2026-02-06: Iteration 35 — Multi-video DNSMOS validation (5 recordings)
  - Mean enhanced: SIG=3.39, BAK=4.01, OVRL=3.11 (all improve vs original)
  - BAK consistently > 3.5 across all recordings (σ=0.30)
  - Average OVRL above 3.0 target
- [x] 2026-02-06: Iteration 34 — SIG-aware quality monitoring + atten_lim benchmark
  - `_run_stage()` now checks SIG degradation (not just OVRL)
  - `_quick_dnsmos()` returns dict with sig/bak/ovrl
  - Added `atten_lim_db` param to DeepFilterNet + config + CLI `--atten-lim`
  - Benchmarked atten_lim=None/20/15/12 dB: **None is optimal** (all limits degrade scores)
- [x] 2026-02-06: Iterations 24-33 (Quality optimization phase)
  - Fixed two-pass loudness normalization (-41 → -18.3 LUFS)
  - Strength benchmark: 1.0 is optimal (changed from 0.8)
  - Processing chain reorder (loudnorm last)
  - 34 unit tests (synthetic audio, no network)
  - Per-stage quality monitoring (skip degrading stages)
  - Preset system (lecture/panel/noisy)
  - Automatic quality report (DNSMOS + JSON)
  - Quality score recalibration (capped at 100)
  - GPU tensor fix in DeepFilterNet
  - Quality gate now passes
- [x] 2026-02-06: Short-audio guards, preserved original audio path, fixed pipeline test flags
- [x] 2026-02-06: Conservative repo cleanup (remove caches/empty dirs)
- [x] 2026-02-06: Iterations 20-22 (Code cleanup and shared utilities)
- [x] 2026-02-01: Iterations 13-18 (Audio quality refinement)
- [x] 2026-02-01: Iterations 9-12 (Speaker enhancement)
- [x] 2026-01-31: Iterations 0-8 (ML enhancement, infrastructure)
