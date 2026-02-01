# AGENTS.md - LLM Agent Instructions

> **This file is for AI agents/LLMs working on this project.**
> It contains the context, rules, and procedures for autonomous development.

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

## Current Status (Last Updated: 2026-01-31)

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

### Current Audio Pipeline (Phase 2 - ML Enhancement)
```
YouTube URL → Download → Extract Audio → VAD → ML Spectral Gating → Filters → Normalization → Output
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

### Directory Structure (MUST NOT CHANGE)
```
audio-restorer/
├── AGENTS.md           ← YOU ARE HERE - Read first!
├── ITERATION_LOG.md    ← Log of all changes made
├── ROADMAP.md          ← Current roadmap and plans
├── QUALITY_METRICS.md  ← How we measure audio quality
├── TESTS/              ← All test files
│   ├── test_pipeline.py
│   ├── test_quality.py
│   └── test_benchmarks.py
├── run.py              ← Main entry point
├── config.py           ← Configuration
├── src/
│   ├── downloader.py
│   ├── audio_enhancer.py ← MAIN FOCUS FOR IMPROVEMENTS
│   └── pipeline.py
├── output/             ← Test outputs go here
├── temp/               ← Temporary files
└── benchmarks/         ← Quality benchmarks and samples
```

## 📝 Agent Rules

### 1. ALWAYS Update Documentation
- Update `ITERATION_LOG.md` with every change
- Update `ROADMAP.md` when completing milestones
- Never leave code undocumented

### 2. ALWAYS Test with Reference Video
- Use `https://youtu.be/cglDoG0GzyA` for all quality tests
- Keep test outputs in `benchmarks/` with timestamps
- Compare before/after with metrics

### 3. NEVER Break Existing Functionality
- Run tests before committing changes
- Maintain backward compatibility
- If adding features, make them optional/configurable

### 4. ALWAYS Focus on Audio Quality
- Video features are secondary
- Every iteration should improve audio OR add audio testing
- Document audio quality improvements with metrics

### 5. ALWAYS Be Explicit
- No hidden assumptions
- Document WHY changes were made
- Include examples in docstrings

## 🔄 Iteration Workflow

When working on this project:

### Step 1: Read Current State
1. Read `AGENTS.md` (this file)
2. Read `ITERATION_LOG.md` - see what's been done
3. Read `ROADMAP.md` - see current priorities
4. Read `QUALITY_METRICS.md` - understand how we measure success

### Step 2: Plan Your Work
1. Check `ROADMAP.md` for current phase
2. Create a plan for what you'll implement
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

### Known Limitations
1. ✅ ML-based noise suppression implemented (PyTorch spectral gating with VAD)
2. ⬜ No speaker isolation yet (Phase 3)
3. ⬜ No echo cancellation yet (Phase 4)
4. Fixed filter frequencies (100Hz/8000Hz) - could be adaptive

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
**Documentation:** `docs/GITHUB_SETUP.md`

### When to Commit & Push

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
- ❌ Binary outputs (already in .gitignore)

### Quick Workflow

```bash
# 1. Setup (do this first in every session)
source ~/.config/github/audio-restorer.env
git remote set-url origin https://${GITHUB_USERNAME}:${GITHUB_TOKEN}@github.com/${GITHUB_USERNAME}/${GITHUB_REPO}.git

# 2. Check current status
git status

# 3. Pull latest changes (always pull before starting)
git pull origin main

# 4. Do your work...

# 5. Before committing - verify tests pass
python -m pytest tests/ -v --tb=short

# 6. Stage, commit, push
git add .
git commit -m "type: Description of changes"
git push origin main

# 7. Cleanup (remove token from URL)
git remote set-url origin https://github.com/${GITHUB_USERNAME}/${GITHUB_REPO}.git
```

### Commit Message Format

```
type: Brief description (50 chars)

- feat: New feature
- fix: Bug fix  
- docs: Documentation
- test: Tests
- refactor: Code restructuring
- perf: Performance
- chore: Maintenance
```

### Pre-Commit Checklist

Before every commit:
1. ✅ `git status` - Know what changed
2. ✅ Tests pass: `python -m pytest tests/ -v`
3. ✅ Quality gate: `python tests/quality_gate.py output/test.wav`
4. ✅ Meaningful commit message
5. ✅ No secrets in staged files
```

## Enhancer Interface Contract

All enhancers must follow this interface (defined in `src/enhancer_base.py`):

```python
from pathlib import Path
from src.enhancer_base import BaseEnhancer

class MyEnhancer(BaseEnhancer):
    def __init__(self,
                 noise_reduction_strength: float = 0.8,
                 use_gpu: bool = False,
                 verbose: bool = True):
        # Initialize enhancer
        pass

    @property
    def name(self) -> str:
        return "MyEnhancer"

    def enhance(self, input_path: Path, output_path: Path) -> Path:
        # Process audio
        # Return path to enhanced file
        return output_path
```

### Adding a New Enhancer

1. Create enhancer class in `src/` directory
2. Inherit from `BaseEnhancer` or follow the interface
3. Add to `_create_enhancer()` in `src/pipeline.py`
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

| Name | File | Description |
|------|------|-------------|
| `--dereverb` | dereverb_enhancer.py | NARA-WPE de-reverberation (removes room echo) |

**Note:** De-reverberation is disabled by default. Use `--dereverb` flag for recordings with significant room echo. Best for files < 10 minutes due to CPU-intensive processing.

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

### Phase 2: SOTA Upgrades (In Progress)
| # | Iteration | Status |
|---|-----------|--------|
| 6 | SOTA metrics (DNSMOS, PESQ, STOI) | **Complete** |
| 7 | Resemble Enhance integration | Pending |
| 8 | Comprehensive benchmark suite | Pending |
| 9 | Pipeline optimization | Pending |

### Available Enhancers (Best to Basic)
1. `--enhancer deepfilter` - Neural network denoising (best quality)
2. `--enhancer torch_advanced` - PyTorch + Silero VAD (default)
3. `--enhancer torch` - PyTorch spectral gating
4. `--enhancer simple` - ffmpeg filters only (fastest)

### Optional Flags
- `--dereverb` - Add de-reverberation (removes room echo, CPU-intensive)

---

**Remember:** This is an AUDIO QUALITY project. Video is secondary.
Every iteration should make conference audio clearer and more intelligible.

**Last updated by:** Iteration 5 (NARA-WPE De-reverberation)
**Current Phase:** High-Impact Improvements COMPLETE (Iterations 0-5)
**Final Quality Score:** 115.9/100 (DeepFilterNet - exceeded all targets)
**Optional:** --dereverb flag for room echo removal
