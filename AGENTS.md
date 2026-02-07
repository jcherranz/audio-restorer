# AGENTS.md - LLM Agent Instructions

> **This file is for AI agents/LLMs working on this project.**
> It contains the context, rules, and procedures for autonomous development.

## Document Hierarchy

This project uses three complementary documents:

| Document | Purpose | Changes |
|----------|---------|---------|
| `docs/SENIOR_ENGINEER_PROMPT.md` | **Mindset** - How to think (principles, behaviors) | Rarely |
| `docs/WORKFLOW_ORCHESTRATION.md` | **Process** - How to work (planning, verification, tasks) | Rarely |
| `AGENTS.md` (this file) | **Project** - What to do on THIS project | Often |

**Read order:** SENIOR_ENGINEER → WORKFLOW_ORCHESTRATION → AGENTS.md

When in doubt:
- Behavioral questions → SENIOR_ENGINEER_PROMPT.md
- Process questions → WORKFLOW_ORCHESTRATION.md
- Project-specific questions → This file

## Project Goal

Create the best possible audio restoration tool for conference recordings.
**Current Focus:** Pipeline is mature — DeepFilterNet achieves mean OVRL=3.11 across 5 diverse recordings.
**Video processing is NOT a priority.**

## Kaizen Approach

This project follows **Kaizen** (continuous improvement) principles:
- **One iteration per session** - Complete, test, document before next
- **Measurable outcomes** - Every change has before/after metrics
- **No regressions** - Quality score must stay >= 75, SNR >= 25dB
- **Document everything** - Future agents need full context

## Current Status (Last Updated: 2026-02-07)

### What Works Now
- ✅ YouTube audio downloading (yt-dlp)
- ✅ DeepFilterNet neural denoising (best quality, GPU-accelerated)
- ✅ ML spectral gating fallback (CPU-only systems)
- ✅ Per-stage DNSMOS quality monitoring (auto-skips degrading stages)
- ✅ Two-pass EBU R128 loudness normalization
- ✅ Automatic quality report (DNSMOS + grade)
- ✅ Preset system (lecture/panel/noisy)
- ✅ 39 unit tests passing

### Test Videos

| Video | Type | Noise | DNSMOS OVRL (enhanced) |
|-------|------|-------|------------------------|
| cglDoG0GzyA | Conference (primary reference) | moderate | 2.63 |
| FGDqFZZabsY | Conference | moderate | 2.11 |
| UF8uR6Z6KLc | Steve Jobs keynote | low | 3.78 |
| 8jPQjjsBbIc | Technical talk | moderate | 3.70 |
| arj7oStGLkU | TED talk | low | 3.34 |
| **Mean** | | | **3.11 ± 0.65** |

### Current Audio Pipeline
```
YouTube URL → Download → Extract Audio
  → [Pre-processing: Hum Removal, Click Removal]
  → Enhancement (DeepFilterNet / ML Spectral Gating / Simple)
  → [Super-resolution: VoiceFixer (optional)]
  → [Post-processing: Dereverb, Diarization, Isolation, Distance-robust, AGC, De-essing, Comfort Noise]
  → Loudness Normalization (two-pass, always last) → Quality Report → Output
```

### Quality Metrics (Primary Reference: cglDoG0GzyA)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| DNSMOS OVRL | 2.63 | > 2.0 | Passing |
| DNSMOS SIG | 2.96 | > 2.0 | Passing |
| DNSMOS BAK | 3.84 | > 3.5 | Passing |
| Quality Score | 87.9/100 | > 75 | Passing |
| SNR | 47 dB | > 25 dB | Passing |
| Loudness | -18.3 LUFS | -16 LUFS | Passing |
| Unit Tests | 39 | > 20 | Passing |

### Multi-Video Aggregate (5 recordings)

| Metric | Original | Enhanced | Improvement |
|--------|----------|----------|-------------|
| SIG | 2.86 ± 1.15 | 3.39 ± 0.65 | +0.53 |
| BAK | 2.60 ± 1.02 | 4.01 ± 0.30 | +1.41 |
| OVRL | 2.45 ± 1.01 | 3.11 ± 0.65 | +0.66 |

## 🏗️ Project Architecture

### Core Principle: AUDIO ONLY
- Video processing exists but is SECONDARY
- All quality improvements focus on AUDIO
- Output format: WAV (lossless)

### Directory Structure
```
audio-restorer/
├── AGENTS.md           ← YOU ARE HERE - Read first!
├── ITERATION_LOG.md    ← Log of all changes made
├── docs/               ← Detailed documentation
├── run.py              ← Main entry point
├── config.py           ← Configuration (settings, paths, presets)
├── src/
│   ├── audio_utils.py      ← Shared I/O: load_mono_audio, save_audio, prevent_clipping
│   ├── pipeline.py          ← Main orchestrator
│   ├── downloader.py        ← YouTube downloading
│   ├── audio_enhancer.py    ← SimpleEnhancer (ffmpeg fallback)
│   ├── ml_enhancer.py       ← TorchEnhancer, AdvancedMLEnhancer
│   ├── deepfilter_enhancer.py ← DeepFilterNet neural denoising
│   ├── dereverb_enhancer.py ← NARA-WPE de-reverberation
│   ├── diarization.py       ← Speaker diarization
│   ├── speaker_isolation.py ← Main speaker extraction
│   ├── distance_enhancer.py ← Distance-robust gain/EQ
│   ├── speaker_agc.py       ← Per-speaker AGC
│   ├── deesser.py           ← Sibilance reduction
│   ├── hum_remover.py       ← 50/60Hz notch filter
│   ├── click_remover.py     ← Transient artifact removal
│   ├── comfort_noise.py     ← Pink noise for silence regions
│   ├── voicefixer_enhancer.py ← VoiceFixer speech super-resolution
│   ├── video_merger.py      ← Audio+video merging
│   └── sota_metrics.py      ← DNSMOS, PESQ, STOI
├── tests/              ← Test files
│   ├── test_modules.py      ← 39 unit tests (synthetic audio, no network)
│   ├── test_pipeline.py     ← Pipeline integration tests
│   ├── reference_videos.json ← 5 test video definitions
│   └── ...
├── tasks/              ← Task tracking
│   ├── todo.md
│   └── lessons.md
├── output/             ← Processed outputs
├── temp/               ← Temporary/cached files
└── benchmarks/         ← Quality benchmarks
```

## 📝 Project-Specific Rules

> For general principles (simplicity, assumptions, scope), see `docs/SENIOR_ENGINEER_PROMPT.md`.
> For workflow (planning, verification, tasks), see `docs/WORKFLOW_ORCHESTRATION.md`.

### 1. Documentation Updates
- Update `ITERATION_LOG.md` with every change (what, why, metrics)
- Track work in `tasks/todo.md`

### 2. Testing with Reference Video
- Use `https://youtu.be/cglDoG0GzyA` for all quality tests
- Run unit tests before committing: `python -m pytest tests/test_modules.py -v`
- Compare before/after with DNSMOS metrics

### 3. Audio Quality Focus
- Video features are secondary
- Every iteration should improve audio OR add audio testing
- New features must be optional (CLI flags)

## 🔄 Iteration Workflow

### Step 1: Read Current State
1. Read `AGENTS.md` (this file)
2. Read `ITERATION_LOG.md` - see what's been done
3. Read `tasks/todo.md` - see current priorities

### Step 2: Plan Your Work
1. Check todo.md for pending tasks
2. Create a plan with checkable items
3. Document expected outcomes

### Step 3: Implement
1. Make changes to code
2. Add/update tests
3. Run tests

### Step 4: Document
1. Update `ITERATION_LOG.md` with results
2. Update `tasks/todo.md`

### Step 5: Verify
1. Check that all 34+ tests pass
2. Verify audio quality didn't degrade
3. Confirm documentation is complete

## Testing Requirements

### Quick Verification
```bash
source venv/bin/activate

# Unit tests (34 tests, ~48s)
python -m pytest tests/test_modules.py -v

# Process reference video
python run.py "https://youtu.be/cglDoG0GzyA" --audio-only --keep-temp

# Measure quality
python tests/measure_quality.py output/audio_cglDoG0GzyA_enhanced.wav
```

### DNSMOS Benchmarking
```bash
# Run SOTA metrics on enhanced audio
python tests/sota_benchmark.py output/enhanced.wav

# Multi-video validation
python tests/multi_video_benchmark.py --videos 5 --sota
```

## Key Findings from Benchmarking

### What Works
- **DeepFilterNet at full strength (1.0)** — optimal across all metrics
- **`atten_lim_db = None` (unlimited suppression)** — best for conference audio
- **Two-pass loudnorm** — accurate to within 0.5 LUFS of target
- **Per-stage quality monitoring** — auto-skips degrading stages

### What Doesn't Work (Measured)
| Technique | Impact | Why |
|-----------|--------|-----|
| Strength mixing (< 1.0) | All metrics degrade | Blending noisy original back in |
| `atten_lim_db` 12-20 dB | OVRL -0.46 to -0.99 | Limits suppression, more residual noise |
| NARA-WPE dereverb (post) | OVRL -1.30 | Distorts speech on clean signal |
| NARA-WPE dereverb (pre) | OVRL -0.10 | Still hurts SIG |
| Hum removal (on enhanced) | OVRL -0.25 | False positives on clean audio |
| Click removal (on enhanced) | OVRL -0.23 | Detects speech transients as clicks |

### Neutral (No Measurable Impact)
- De-essing (ΔOVRL ≈ 0.00) — no sibilance issues on typical conference audio
- Comfort noise (ΔOVRL ≈ -0.04) — negligible effect

### SIG Ceiling
- SIG is the bottleneck for OVRL (listeners weight speech distortion > noise)
- Realistic SIG ceiling for YouTube conference audio: 2.8-3.5
- Source material quality is the primary limiter, not processing
- Improving beyond this requires generative models (speech super-resolution)

## Enhancer Interface Contract

All enhancers must implement an `enhance()` method:

```python
from pathlib import Path

class MyEnhancer:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    @property
    def name(self) -> str:
        return "MyEnhancer"

    def enhance(self, input_path: Path, output_path: Path, target_sr: int = 48000) -> Path:
        # Process audio file, write to output_path
        return output_path
```

Post-processing modules use `process()` instead of `enhance()`:
```python
def process(self, input_path: Path, output_path: Path) -> Path:
    # Process audio, write to output_path
    return output_path
```

All modules should use shared utilities from `src/audio_utils.py`:
- `load_mono_audio(path)` — load audio as mono float32
- `save_audio(audio, path, sr)` — save with auto mkdir
- `prevent_clipping(audio)` — normalize peaks to 0.95

### Available Enhancers (Best to Basic)
1. `--enhancer deepfilter` - Neural network denoising (best quality, GPU)
2. `--enhancer torch_advanced` - PyTorch + Silero VAD (default)
3. `--enhancer torch` - PyTorch spectral gating
4. `--enhancer simple` - ffmpeg filters only (fastest)

### Optional Flags

| Flag | File | DNSMOS Impact | Notes |
|------|------|---------------|-------|
| `--preset lecture/panel/noisy` | config.py | Varies | Enables flag combos |
| `--dereverb` | dereverb_enhancer.py | **-1.30 OVRL** | Auto-skipped by quality check |
| `--diarize` | diarization.py | N/A | Analysis only |
| `--isolate-speaker` | speaker_isolation.py | N/A | Extracts main speaker |
| `--distance-robust` | distance_enhancer.py | N/A | Quality-checked |
| `--speaker-agc` | speaker_agc.py | N/A | Quality-checked |
| `--deess` | deesser.py | Neutral | Quality-checked |
| `--remove-hum` | hum_remover.py | **-0.25 OVRL** | Auto-skipped by quality check |
| `--remove-clicks` | click_remover.py | **-0.23 OVRL** | Auto-skipped by quality check |
| `--comfort-noise` | comfort_noise.py | Neutral | Quality-checked |
| `--super-resolve` | voicefixer_enhancer.py | **+0.23 OVRL** | Generative model, runs after DeepFilterNet |
| `--atten-lim` | deepfilter_enhancer.py | Degrades | None is optimal |

### Module Categories

| Category | Modules | Status |
|----------|---------|--------|
| **Active by default** | DeepFilterNet, loudnorm, quality report | Always runs |
| **Useful for specific cases** | diarization, speaker isolation, AGC, distance-robust | Structural transforms for multi-speaker audio |
| **Neutral** | de-essing, comfort noise | No measurable impact on DNSMOS |
| **Harmful on enhanced audio** | dereverb, hum removal, click removal | Quality-checked, auto-skipped; see docstrings for deprecation notes |

## Iteration Plan

### All Major Phases Complete

| Phase | Iterations | Status |
|-------|------------|--------|
| 1: High-Impact Quality | 0-5 | **Complete** |
| 2: SOTA Upgrades | 6-12 | **Complete** |
| 4: Audio Refinement | 13-18 | **Complete** |
| Cleanup | 19-22 | **Complete** |
| Quality Optimization | 23-33 | **Complete** |
| SIG-Aware Monitoring | 34 | **Complete** |
| Multi-Video Validation | 35 | **Complete** |
| Dereverb Benchmark | 36 | **Complete** (Phase 5 closed) |
| Optional Stage Benchmark | 37 | **Complete** |
| Fix Presets + Quality-Check Pre-proc | 38 | **Complete** |
| README Refresh | 39 | **Complete** |
| Integration Test + DNSMOS | 40 | **Complete** |
| Preset DNSMOS Benchmark | 41 | **Complete** |
| Dead Code Audit | 42 | **Complete** |
| VoiceFixer Integration | 43 | **Complete** |

### Remaining Opportunity
- VoiceFixer real-speech DNSMOS validation (benchmark on reference recordings)
- Consider adding `--super-resolve` to presets if validated
- Pipeline robustness, UX improvements

## 🚨 Common Pitfalls

1. **Use `python3` or `source venv/bin/activate && python`** — no bare `python`
2. **Use `FFMPEG_PATH` from config** — local binary at project root
3. **DeepFilterNet handles device** — do NOT move tensor to CUDA manually
4. **Strength 1.0 is optimal** — mixing original back degrades all metrics
5. **Two-pass loudnorm** — single-pass is inaccurate by ~25 LUFS
6. **`_quick_dnsmos()` returns dict** — `{sig, bak, ovrl}`, not float
7. **NARA-WPE hurts quality** — in both pre/post positions
8. **Background scripts need `PYTHONUNBUFFERED=1`** — stdout is fully buffered in non-TTY
9. **48kHz sample rate throughout** — matches DeepFilterNet native rate

## 🔧 Useful Commands

```bash
# Setup environment
cd audio-restorer
source venv/bin/activate

# Run main tool
python run.py "URL" --audio-only

# Run with preset
python run.py "URL" --audio-only --preset lecture

# Run tests
python -m pytest tests/test_modules.py -v

# DNSMOS on a file
python tests/sota_benchmark.py output/audio.wav
```

## 🌐 GitHub Integration

**Repository:** https://github.com/jcherranz/audio-restorer
**Remote:** SSH (`git@github.com:jcherranz/audio-restorer.git`)

### Commit Guidelines
- `feat:` New feature | `fix:` Bug fix | `docs:` Documentation
- `test:` Tests | `refactor:` Code restructuring | `chore:` Maintenance

---

**Remember:** This is an AUDIO QUALITY project. Video is secondary.
Every iteration should make conference audio clearer and more intelligible.

**Last updated:** 2026-02-07 (Iteration 43 — VoiceFixer speech super-resolution)
**Current Phase:** Pipeline mature — all major phases complete
**Best Enhanced OVRL:** 3.11 mean across 5 diverse conference recordings
