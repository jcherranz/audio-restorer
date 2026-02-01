# Audio Quality Metrics

> **How we objectively measure audio quality improvements**

## Overview

We need objective ways to measure audio quality. This document defines:
1. What metrics we track
2. How to calculate them
3. What "good" values look like
4. How to compare before/after

## Metrics We Track

### 1. Signal-to-Noise Ratio (SNR)

**What it measures:** How much louder the speech is compared to background noise.

**Calculation:**
```python
SNR (dB) = 10 * log10(P_signal / P_noise)
```

**Good Values:**
- < 10 dB: Poor (noise very noticeable)
- 10-20 dB: Fair (noise audible but not overwhelming)
- 20-30 dB: Good (noise present but not distracting)
- > 30 dB: Excellent (noise barely perceptible)

**Target:** > 25 dB after processing

**How to Measure:**
```python
# Estimate from audio file
import numpy as np
import librosa

# Load at native sample rate (pipeline uses 48kHz)
audio, sr = librosa.load('audio.wav', sr=None)

# Estimate noise from silent segments
# Estimate signal from voiced segments
# Calculate ratio
```

---

### 2. Background Noise Level (RMS in dB)

**What it measures:** Average volume of background noise (without speech).

**Calculation:**
```python
RMS = sqrt(mean(audio^2))
Noise_dB = 20 * log10(RMS)
```

**Good Values:**
- > -30 dB: Very noisy
- -40 to -30 dB: Noisy
- -50 to -40 dB: Moderate noise
- < -50 dB: Quiet background

**Target:** < -50 dB during speech pauses

---

### 3. Dynamic Range

**What it measures:** Difference between loudest and quietest parts.

**Calculation:**
```python
Dynamic_Range = Peak_dB - RMS_dB
# OR
Dynamic_Range = Crest_Factor = Peak / RMS
```

**Good Values:**
- 10-15 dB: Over-compressed (poor)
- 15-25 dB: Good for speech
- 25-35 dB: Excellent for speech
- > 35 dB: May have volume inconsistencies

**Target:** 15-25 dB for speech

---

### 4. Loudness (LUFS)

**What it measures:** Perceived loudness following broadcast standards.

**Standards:**
- -16 LUFS: Podcast standard
- -14 LUFS: YouTube standard
- -23 LUFS: Broadcast standard

**Target:** -16 LUFS (podcast standard)

---

### 5. Clarity Score (Custom Metric)

**What it measures:** How clear/intelligible the speech is.

**Calculation:**
```python
# Based on:
# - High frequency content (4-8 kHz) - consonants
# - Low noise floor
# - Consistent volume

clarity = (HF_energy / Total_energy) * (1 - Noise_ratio) * Consistency_factor
```

**Good Values:**
- < 0.3: Poor intelligibility
- 0.3-0.5: Fair
- 0.5-0.7: Good
- > 0.7: Excellent

**Target:** > 0.7

---

### 6. Processing Artifacts Score

**What it measures:** Unwanted sounds introduced by processing.

**Indicators:**
- Distortion (clipping)
- "Underwater" sound (over-processing)
- Echo/ringing
- Musical noise (from aggressive noise reduction)

**Measurement:** Subjective listening + spectral analysis

**Target:** No audible artifacts

---

### 7. Processing Time

**What it measures:** How long enhancement takes (real-time factor).

**Calculation:**
```python
RTF = Processing_Time / Audio_Duration
```

**Good Values:**
- RTF < 0.1: Very fast (10x real-time)
- RTF < 0.5: Acceptable (2x real-time)
- RTF < 1.0: Real-time capable
- RTF > 1.0: Slower than real-time

**Target:** RTF < 0.5 (process 1-hour audio in <30 minutes)

---

## Measurement Tools

### Automated Analysis Script

Create `tests/measure_quality.py`:

```python
#!/usr/bin/env python3
"""
Audio Quality Measurement Tool
Analyzes audio file and reports quality metrics
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path


def measure_snr(audio, sr):
    """Estimate SNR using voice activity detection"""
    # Separate speech and noise segments
    # Calculate power in each
    # Return ratio in dB
    pass


def measure_loudness(audio, sr):
    """Measure LUFS loudness"""
    # Use pyloudnorm or similar
    pass


def measure_clarity(audio, sr):
    """Calculate clarity score"""
    # Analyze frequency content
    # Check for high-frequency energy (consonants)
    # Return 0-1 score
    pass


def measure_all(audio_path):
    """Measure all quality metrics"""
    # Load at native sample rate (48kHz for this pipeline)
    audio, sr = librosa.load(audio_path, sr=None)

    results = {
        'snr_db': measure_snr(audio, sr),
        'noise_level_db': measure_noise_level(audio, sr),
        'dynamic_range_db': measure_dynamic_range(audio, sr),
        'loudness_lufs': measure_loudness(audio, sr),
        'clarity_score': measure_clarity(audio, sr),
        'duration_s': len(audio) / sr,
    }

    return results


def print_report(results):
    """Print formatted quality report"""
    print("=" * 50)
    print("AUDIO QUALITY REPORT")
    print("=" * 50)
    print(f"SNR:              {results['snr_db']:.1f} dB")
    print(f"Noise Level:      {results['noise_level_db']:.1f} dB")
    print(f"Dynamic Range:    {results['dynamic_range_db']:.1f} dB")
    print(f"Loudness:         {results['loudness_lufs']:.1f} LUFS")
    print(f"Clarity Score:    {results['clarity_score']:.2f}/1.0")
    print(f"Duration:         {results['duration_s']:.1f} s")
    print("=" * 50)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python measure_quality.py <audio_file>")
        sys.exit(1)
    
    results = measure_all(sys.argv[1])
    print_report(results)
```

---

## Comparison Methodology

### Before/After Comparison

For each iteration, we must measure:

1. **Original YouTube audio** (before any processing)
2. **Current pipeline output** (after processing)
3. **Calculate improvements**

### Comparison Table Template

| Metric | Before | After | Improvement | Target |
|--------|--------|-------|-------------|--------|
| SNR | X dB | Y dB | +Z dB | >25 dB |
| Noise Level | -X dB | -Y dB | -Z dB | <-50 dB |
| Dynamic Range | X dB | Y dB | ±Z dB | 15-25 dB |
| Loudness | -X LUFS | -Y LUFS | ±Z LUFS | -16 LUFS |
| Clarity | X | Y | +Z | >0.7 |
| Processing Time | - | Xs | - | <5min/hour |

### Quality Score

Calculate overall quality score:

```python
def calculate_quality_score(metrics):
    score = 0
    
    # SNR (30% weight)
    score += min(metrics['snr_db'] / 30, 1.0) * 30
    
    # Clarity (25% weight)
    score += metrics['clarity_score'] * 25
    
    # Noise level (20% weight)
    score += max(0, (-30 - metrics['noise_level_db']) / 20) * 20
    
    # Dynamic range (15% weight)
    dr = metrics['dynamic_range_db']
    if 15 <= dr <= 25:
        score += 15
    else:
        score += 15 * (1 - abs(dr - 20) / 20)
    
    # Loudness (10% weight)
    loudness_diff = abs(metrics['loudness_lufs'] - (-16))
    score += max(0, 10 - loudness_diff / 2)
    
    return score  # 0-100
```

**Quality Levels:**
- 90-100: Excellent
- 80-90: Very Good
- 70-80: Good
- 60-70: Acceptable
- < 60: Needs improvement

---

## Reference Measurements

### Reference Video: https://youtu.be/cglDoG0GzyA

**Current State (Iteration 0):**

| Metric | Original | After Simple Enhancement | Improvement |
|--------|----------|--------------------------|-------------|
| SNR | ~10 dB* | ~15 dB* | +5 dB |
| Noise Level | ~-40 dB* | ~-50 dB* | -10 dB |
| Dynamic Range | ~30 dB* | ~20 dB* | -10 dB |
| Loudness | -23 LUFS* | -16 LUFS | +7 LUFS |
| Clarity | 0.4* | 0.5* | +0.1 |
| Quality Score | ~50 | ~65 | +15 |

*Estimated values - need proper measurement

**Target (After Phase 2):**

| Metric | Target |
|--------|--------|
| SNR | >25 dB |
| Noise Level | <-55 dB |
| Dynamic Range | 15-25 dB |
| Loudness | -16 LUFS |
| Clarity | >0.7 |
| Quality Score | >85 |

---

## Testing Protocol

### Every Iteration MUST:

1. Process reference video:
   ```bash
   python run.py "https://youtu.be/cglDoG0GzyA" --audio-only --keep-temp
   ```

2. Measure quality:
   ```bash
   python tests/measure_quality.py output/audio_*.wav
   ```

3. Compare to previous iteration

4. Document in ITERATION_LOG.md

### Batch Testing

For comparing multiple settings:

```bash
# Test different noise reduction strengths
for strength in 0.5 0.7 0.9 0.95; do
    python run.py "URL" --audio-only --noise-reduction $strength
    python tests/measure_quality.py output/audio_*.wav > "benchmark/nr_${strength}.txt"
done
```

---

**Last Updated:** 2026-02-01
**Note:** For quick analysis use `tests/measure_quality.py`. For academic-grade
metrics (DNSMOS, PESQ, STOI), use `tests/sota_benchmark.py`.
**Pipeline Sample Rate:** 48kHz (matches DeepFilterNet native rate)
