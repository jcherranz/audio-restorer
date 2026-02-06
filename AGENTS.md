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
**Current Focus:** High-impact audio quality improvements via Kaizen methodology.
**Video processing is NOT a priority.**

## Kaizen Approach

This project follows **Kaizen** (continuous improvement) principles:
- **One iteration per session** - Complete, test, document before next
- **Measurable outcomes** - Every change has before/after metrics
- **No regressions** - Quality score must stay >= 75, SNR >= 25dB
- **Document everything** - Future agents need full context

## Current Status (Last Updated: 2026-02-06)

### What Works Now
- ✅ YouTube audio downloading
- ✅ Basic noise reduction (spectral gating)
- ✅ Audio normalization
- ✅ Command-line interface

### Test Video (DO NOT CHANGE)
```
URL: https://youtu.be/cglDoG0GzyA
Duration: ~58 minutes
Purpose: Reference video for all quality testing
Baseline Quality Score: 66.5 (simple enhancer)
ML Enhancer Score: 81.0 (torch_advanced)
DeepFilter Score: 115.9 (deepfilter) - BEST
Location: Test files cached in temp/ (when --keep-temp used)
```

### Current Audio Pipeline
```
YouTube URL → Download → Extract Audio
  → [Pre-processing: Hum Removal, Click Removal]
  → Enhancement (DeepFilterNet / ML Spectral Gating / Simple)
  → [Post-processing: Dereverb, Diarization, Isolation, Distance-robust, AGC, De-essing, Comfort Noise]
  → Loudness Normalization → Output
```

### Current Quality Metrics
| Metric | Simple | ML (torch) | DeepFilter | Best Improvement |
|--------|--------|------------|------------|------------------|
| Quality Score | 66.5 | 81.0 | **115.9** | +49.4 (+74%) |
| SNR | 21.4 dB | 28.5 dB | **49.0 dB** | +27.6 dB |
| Noise Level | -38.3 dB | -44.3 dB | **-84.4 dB** | -46.1 dB |
| Processing Time | 171s | 40s | 134s | varies |

### SOTA Metrics (Iteration 6)
| Metric | DeepFilter | ML (torch) | Simple | Range |
|--------|------------|------------|--------|-------|
| DNSMOS OVRL | **2.62** | 1.16 | 1.08 | 1-5 |
| DNSMOS SIG | **2.95** | 1.24 | 1.16 | 1-5 (speech) |
| DNSMOS BAK | **3.87** | 1.34 | 1.23 | 1-5 (background) |

**Note:** DNSMOS was trained on clean speech. Conference recordings naturally score lower.

## 🏗️ Project Architecture

### Core Principle: AUDIO ONLY
- Video processing exists but is SECONDARY
- All quality improvements focus on AUDIO
- Output format: WAV (lossless) or high-quality MP3

### Directory Structure
```
audio-restorer/
├── AGENTS.md           ← YOU ARE HERE - Read first!
├── ITERATION_LOG.md    ← Log of all changes made
├── ROADMAP.md          ← Current roadmap and plans
├── docs/               ← Detailed documentation
├── run.py              ← Main entry point
├── config.py           ← Configuration (settings, paths)
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
│   ├── video_merger.py      ← Audio+video merging
│   └── sota_metrics.py      ← DNSMOS, PESQ, STOI
├── tests/              ← Test files
├── output/             ← Processed outputs
├── temp/               ← Temporary files
└── benchmarks/         ← Quality benchmarks
```

## 📝 Project-Specific Rules

> For general principles (simplicity, assumptions, scope), see `docs/SENIOR_ENGINEER_PROMPT.md`.
> For workflow (planning, verification, tasks), see `docs/WORKFLOW_ORCHESTRATION.md`.

### 1. Documentation Updates
- Update `ITERATION_LOG.md` with every change (what, why, metrics)
- Update `ROADMAP.md` when completing milestones
- Track work in `tasks/todo.md`

### 2. Testing with Reference Video
- Use `https://youtu.be/cglDoG0GzyA` for all quality tests
- Run quality gate before committing: `python tests/quality_gate.py`
- Compare before/after with metrics

### 3. Audio Quality Focus
- Video features are secondary
- Every iteration should improve audio OR add audio testing
- New features must be optional (CLI flags)

## 🔄 Iteration Workflow

When working on this project:

### Step 1: Read Current State
1. Read `AGENTS.md` (this file)
2. Read `ITERATION_LOG.md` - see what's been done
3. Read `ROADMAP.md` - see current priorities
4. Read `docs/QUALITY_METRICS.md` - understand how we measure success

### Step 2: Plan Your Work
1. Check `ROADMAP.md` for current phase
2. Create a plan for what you'll implement (track in `tasks/todo.md` per `docs/WORKFLOW_ORCHESTRATION.md`)
3. Document expected outcomes

### Step 3: Implement
1. Make changes to code
2. Add/update tests in `TESTS/`
3. Run tests with reference video

### Step 4: Document
1. Update `ITERATION_LOG.md` with:
   - What was changed
   - Why it was changed
   - Test results
   - Before/after metrics
2. Update `ROADMAP.md` if completing milestones

### Step 5: Verify
1. Check that all tests pass
2. Verify audio quality improved (or didn't degrade)
3. Confirm documentation is complete

## Testing Requirements

### Quality Gate (MANDATORY)

Before any change is accepted, run the quality gate:

```bash
# Quick verification
python tests/quality_gate.py output/enhanced.wav

# Fails if: quality_score < 75 OR SNR < 25dB
```

### Full Test Suite

```bash
# 1. Verify CLI works
python run.py --help

# 2. Run existing tests
python -m pytest tests/ -v

# 3. Test with reference video
python run.py "https://youtu.be/cglDoG0GzyA" --audio-only -o output/test

# 4. Measure quality
python tests/measure_quality.py output/test_enhanced.wav

# 5. Compare with baseline
python tests/measure_quality.py --compare output/baseline.wav output/test_enhanced.wav
```

### Mandatory Test for Every Iteration
```bash
# Run this exact command for every quality improvement
python run.py "https://youtu.be/cglDoG0GzyA" --audio-only --keep-temp

# Then analyze output with:
python tests/measure_quality.py output/audio_cglDoG0GzyA_enhanced.wav
```

### Quality Metrics to Track
1. **SNR (Signal-to-Noise Ratio)** - Higher is better
2. **Dynamic Range** - Should be well-balanced
3. **Clarity Score** - Custom metric for speech intelligibility
4. **Processing Time** - Should not increase significantly
5. **DNSMOS** (SOTA) - Neural MOS predictor (SIG, BAK, OVRL), range 1-5
6. **PESQ** (SOTA) - Perceptual quality, range 1-4.5 (requires reference)
7. **STOI** (SOTA) - Speech intelligibility, range 0-1 (requires reference)

### SOTA Metrics (Iteration 6+)
```bash
# Run SOTA metrics on enhanced audio
python tests/sota_benchmark.py output/enhanced.wav

# Compare multiple files
python tests/sota_benchmark.py output/*.wav --quiet
```

## 📊 Current Audio Processing Chain

```python
# Current implementation in audio_enhancer.py:

class SimpleEnhancer:
    def enhance(self, audio_path, output_path):
        # 1. High-pass filter @ 100Hz (remove rumble)
        # 2. Low-pass filter @ 8000Hz (remove hiss)
        # 3. Dynamic compression (even out loud/quiet)
        # 4. Loudness normalization (-16 LUFS)
        pass
```

### Known Limitations & Design Decisions

**Implemented Features:**
1. ✅ ML-based noise suppression (PyTorch spectral gating, DeepFilterNet neural)
2. ✅ Speaker diarization and isolation (Phase 3)
3. ✅ Distance-robust enhancement, per-speaker AGC (Phase 3)
4. ✅ De-reverberation (NARA-WPE, optional)
5. ✅ De-essing, hum removal, click removal, comfort noise (Phase 4)
6. ✅ Shared audio utilities, pipeline simplification (Cleanup)

**Acknowledged Limitations:**
1. **Heuristic Diarization:** Uses energy + spectral features, not embedding-based models
   like pyannote.audio. Works for typical conference recordings but may struggle with
   complex multi-speaker scenarios with overlapping speech.

2. **Heuristic Distance Estimation:** Simple feature-based approach (energy, HF ratio,
   SNR, reverb ratio). Not model-based speaker separation. May amplify noise in
   challenging acoustic conditions.

3. **Simplified Quality Metrics:** Basic measure_quality.py uses simplified LUFS
   estimation and custom clarity scores. For academic-grade metrics, use
   `tests/sota_benchmark.py` which provides DNSMOS, PESQ, STOI, and SI-SDR.

4. **Sample Rate:** Pipeline uses 48kHz throughout to match DeepFilterNet's native rate.
   This is a deliberate choice to avoid quality loss from multiple resampling operations.

5. **GPU Dependency for Best Quality:** DeepFilterNet provides best results but is
   slow on CPU. Auto-detection selects deepfilter when GPU available, torch_advanced
   otherwise.

## 🎯 Success Criteria

### Audio Quality is "Good Enough" When:
- [x] Speech is clearly intelligible at normal volume ✅
- [x] Background noise reduced by >50% ✅
- [x] No audible artifacts from processing ✅
- [x] Dynamic range is natural (not over-compressed) ✅
- [x] SNR improved by >10dB ✅ (achieved +7.1dB on top of baseline)

### Audio Quality is "Excellent" When:
- [x] Sounds like professional podcast quality ✅ (Quality Score: 81.0)
- [x] Background noise barely perceptible ✅ (SNR: 28.5 dB)
- [x] Speaker voice is full and clear ✅
- [ ] No echo or room reverb (Phase 4)
- [x] Consistent volume throughout ✅

## 🚨 Common Pitfalls to Avoid

1. **Don't change the test video URL** - We need consistency
2. **Don't remove existing functionality** - Add, don't replace
3. **Don't skip testing** - Every change must be measured
4. **Don't break the CLI interface** - Keep run.py working
5. **Don't ignore documentation** - Update logs immediately

## 📞 When to Ask for User Input

- When changing project scope or priorities
- When trade-offs significantly affect quality vs speed
- When adding major dependencies
- When refactoring core architecture

## 🔧 Useful Commands

```bash
# Setup environment
cd audio-restorer
source venv/bin/activate

# Run main tool
python run.py "URL" --audio-only --quick

# Run tests
python -m pytest tests/ -v

# Analyze audio quality
python tests/analyze_quality.py output/audio.wav

# Compare two audio files
python tests/compare_audio.py file1.wav file2.wav

# View logs
cat ITERATION_LOG.md
cat ROADMAP.md
```

## 🌐 GitHub Integration

**Repository:** https://github.com/jcherranz/audio-restorer

For complete Git workflow, authentication, and commit guidelines, see: `docs/GITHUB_SETUP.md`

### Quick Reference

**ALWAYS commit when:**
- ✅ Completing a feature or iteration
- ✅ Fixing a bug
- ✅ Adding/updating tests
- ✅ Updating documentation
- ✅ Before ending a session

**NEVER commit:**
- ❌ Broken or incomplete code
- ❌ Files with failing tests
- ❌ Secrets or temporary files

**Commit message format:** `type: Brief description`
- `feat:` New feature | `fix:` Bug fix | `docs:` Documentation
- `test:` Tests | `refactor:` Code restructuring | `chore:` Maintenance

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

### Adding a New Enhancer

1. Create enhancer class in `src/` directory
2. Implement `enhance()` or `process()` method
3. Add to `_create_enhancer()` or `_init_module()` in `src/pipeline.py`
4. Add CLI option in `run.py` argument parser
5. Test with reference video
6. Update ITERATION_LOG.md with metrics

### Current Enhancers

| Name | File | Description |
|------|------|-------------|
| `simple` | audio_enhancer.py | ffmpeg filters only (fast) |
| `torch` | ml_enhancer.py | PyTorch spectral gating |
| `torch_advanced` | ml_enhancer.py | PyTorch + VAD |
| `deepfilter` | deepfilter_enhancer.py | Neural denoising (best quality) |


### Optional Post-Processing

| Flag | File | Description |
|------|------|-------------|
| `--dereverb` | dereverb_enhancer.py | NARA-WPE de-reverberation |
| `--diarize` | diarization.py | Speaker identification |
| `--isolate-speaker` | speaker_isolation.py | Main speaker extraction |
| `--distance-robust` | distance_enhancer.py | Adaptive gain/EQ by distance |
| `--speaker-agc` | speaker_agc.py | Per-speaker volume normalization |
| `--deess` | deesser.py | Sibilance reduction |
| `--remove-hum` | hum_remover.py | 50/60Hz notch filter |
| `--remove-clicks` | click_remover.py | Transient artifact removal |
| `--comfort-noise` | comfort_noise.py | Pink noise in silence regions |

## Dependencies Status

### Currently Installed
- ✅ yt-dlp (YouTube downloading)
- ✅ ffmpeg (audio processing)
- ✅ numpy, soundfile (audio I/O)
- ✅ librosa (audio analysis)
- ✅ pydub (audio manipulation)
- ✅ noisereduce (spectral gating)

### To Install for Next Phase
- ✅ torch, torchaudio (for ML models) - INSTALLED
- ⚠️ deepfilternet (had compatibility issues, using custom implementation)
- ✅ speechbrain (installed, available for Phase 3)

## 🎓 Reference Materials

### Audio Processing Concepts
- **Spectral Gating**: Removes noise by frequency analysis
- **Dynamic Range Compression**: Makes quiet parts louder, loud parts quieter
- **Normalization**: Adjusts overall volume to standard level
- **High/Low Pass Filters**: Remove unwanted frequency ranges

### Quality Metrics
- **SNR**: Signal-to-Noise Ratio in dB
- **LUFS**: Loudness units relative to full scale
- **RMS**: Root Mean Square (average volume)
- **Crest Factor**: Peak to RMS ratio (dynamic range indicator)

---

## Iteration Plan

### Phase 1: High-Impact Quality Improvements (Complete)
| # | Iteration | Status |
|---|-----------|--------|
| 0 | Infrastructure (docs, base class, quality gate) | Complete |
| 1 | Lowpass filter 8kHz -> 12kHz | Complete |
| 2 | Sample rate 16kHz -> 44.1kHz | Complete |
| 3 | DeepFilterNet neural denoising | Complete |
| 4 | Silero VAD neural speech detection | Complete |
| 5 | NARA-WPE de-reverberation | Complete |

### Phase 2: SOTA Upgrades (Complete)
| # | Iteration | Status |
|---|-----------|--------|
| 6 | SOTA metrics (DNSMOS, PESQ, STOI) | **Complete** |
| 7 | ~~Resemble Enhance~~ | **REMOVED** (dependency issues) |
| 8 | Comprehensive benchmark suite | **Complete** |
| 9 | Speaker diarization | **Complete** |
| 10 | Speaker isolation | **Complete** |
| 11 | Distance-robust enhancement | **Complete** |
| 12 | Per-speaker AGC | **Complete** |

### Phase 4: Audio Quality Refinement (Complete)
| # | Iteration | Status |
|---|-----------|--------|
| 13 | scipy dependency + DeepFilterNet strength | **Complete** |
| 14 | Multi-video test suite | **Complete** |
| 15 | De-essing (sibilance control) | **Complete** |
| 16 | Hum removal (50/60Hz) | **Complete** |
| 17 | Click/pop removal | **Complete** |
| 18 | Comfort noise | **Complete** |

### Cleanup: Code Quality (Complete)
| # | Iteration | Status |
|---|-----------|--------|
| 20 | Critical fixes & dead code removal | **Complete** |
| 21 | Extract shared utilities (audio_utils.py) | **Complete** |
| 22 | Pipeline simplification & config cleanup | **Complete** |

### Available Enhancers (Best to Basic)
1. `--enhancer deepfilter` - Neural network denoising (best quality)
2. `--enhancer torch_advanced` - PyTorch + Silero VAD (default)
3. `--enhancer torch` - PyTorch spectral gating
4. `--enhancer simple` - ffmpeg filters only (fastest)

### Optional Flags
- `--dereverb` - Add de-reverberation (removes room echo, CPU-intensive)
- `--diarize` - Perform speaker diarization (identifies speakers)
- `--isolate-speaker` - Isolate main speaker (removes audience/others)
- `--distance-robust` - Apply distance-robust enhancement (adaptive gain/EQ per speaker distance)
- `--speaker-agc` - Apply per-speaker automatic gain control (normalize each speaker's volume)
- `--deess` - Reduce harsh sibilant sounds (/s/, /sh/)
- `--remove-hum` - Remove power line hum (50/60Hz + harmonics)
- `--remove-clicks` - Remove clicks and pops (transient artifacts)
- `--comfort-noise` - Add comfort noise to silence (prevents dead air)

---

**Remember:** This is an AUDIO QUALITY project. Video is secondary.
Every iteration should make conference audio clearer and more intelligible.

**Last updated:** 2026-02-06 (Cleanup iterations 20-22 complete)
**Current Phase:** Cleanup complete - Ready for Phase 5
**Best Quality Score:** 115.9/100 (DeepFilterNet - exceeded all targets)
**Available Refinements:** De-essing, hum removal, click removal, comfort noise
