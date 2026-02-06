# Task List

> Track current work items here. See `docs/WORKFLOW_ORCHESTRATION.md` for workflow.

## In Progress

<!-- Tasks currently being worked on -->
<!-- None -->

## Pending

<!-- Tasks queued for implementation -->
- [ ] Multi-video statistical validation (run 5+ diverse recordings)
- [ ] Improve DNSMOS OVRL beyond 3.0 (currently 2.63)
- [ ] Improve DNSMOS SIG beyond 3.0 (currently 2.96)

## Completed

<!-- Move completed tasks here with date -->
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
