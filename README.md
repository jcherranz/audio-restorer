# Audio Restoration Tool for YouTube Conference Videos

Transform poor-quality conference recordings into clear, intelligible audio using AI-powered noise reduction and speech enhancement.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## What This Tool Does

Downloads audio from YouTube, enhances it with DeepFilterNet neural noise suppression, normalizes loudness to broadcast standards, and outputs a clean WAV file with a DNSMOS quality report.

**Measured quality (mean across 5 diverse conference recordings):**

| Metric | Original | Enhanced | Improvement |
|--------|----------|----------|-------------|
| DNSMOS SIG | 2.86 | 3.39 | +0.53 |
| DNSMOS BAK | 2.60 | 4.01 | +1.41 |
| DNSMOS OVRL | 2.45 | 3.11 | +0.66 |

## Quick Start

```bash
# Setup
git clone https://github.com/jcherranz/audio-restorer.git
cd audio-restorer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Basic usage (audio only, recommended)
python run.py "https://youtu.be/cglDoG0GzyA" --audio-only

# Use a preset for common scenarios
python run.py "https://youtu.be/VIDEO_ID" --audio-only --preset lecture
python run.py "https://youtu.be/VIDEO_ID" --audio-only --preset panel
python run.py "https://youtu.be/VIDEO_ID" --audio-only --preset noisy
```

Output is saved to `./output/` with a `.quality.json` report alongside each enhanced file.

## Presets

| Preset | Description | Enabled Stages |
|--------|-------------|----------------|
| `lecture` | Single speaker, mild processing | DeepFilterNet + de-essing |
| `panel` | Multi-speaker discussions | DeepFilterNet + diarization + AGC + distance-robust + de-essing |
| `noisy` | Aggressive cleanup | DeepFilterNet + de-essing + hum removal + click removal + comfort noise |

All presets use DeepFilterNet as the core enhancer with two-pass EBU R128 loudness normalization. Optional stages are quality-checked with DNSMOS and automatically skipped if they degrade the signal.

## Command Line Options

```
python run.py <youtube_url> [options]

Core:
  -o, --output NAME       Custom output filename (without extension)
  --audio-only            Process audio only (no video) - RECOMMENDED
  --preset PRESET         Use a preset: lecture, panel, noisy
  --enhancer TYPE         Choose: simple, torch, torch_advanced, deepfilter
  --noise-reduction N     Noise reduction strength 0.0-1.0 (default: 1.0)
  --atten-lim DB          DeepFilterNet attenuation limit (default: None = max)

Pre-processing:
  --remove-hum            Remove power line hum (50/60Hz + harmonics)
  --remove-clicks         Remove clicks and pops (transient artifacts)

Enhancement:
  --super-resolve         Apply VoiceFixer speech super-resolution (generative model)

Post-processing:
  --dereverb              Remove room echo (NARA-WPE)
  --diarize               Perform speaker diarization
  --isolate-speaker       Isolate main speaker
  --distance-robust       Adaptive gain/EQ per speaker distance
  --speaker-agc           Per-speaker automatic gain control
  --deess                 Reduce sibilant sounds (/s/, /sh/)
  --comfort-noise         Add comfort noise to silence regions

Output:
  --output-dir PATH       Custom output directory
  --keep-temp             Keep temporary files for debugging
  --quiet                 Reduce output verbosity
  --quick                 Use only ffmpeg filters (no ML)
  --comparison            Create before/after comparison video
```

## Processing Pipeline

```
YouTube URL --> Download --> Extract Audio
  --> [Pre-processing: Hum Removal, Click Removal]
  --> Enhancement (DeepFilterNet / ML Spectral Gating / Simple)
  --> [Super-resolution: VoiceFixer (optional, generative model)]
  --> [Post-processing: Dereverb, Diarization, Isolation, Distance-robust, AGC, De-essing, Comfort Noise]
  --> Loudness Normalization (two-pass EBU R128, always last)
  --> Quality Report (DNSMOS)
  --> Output WAV + quality.json
```

Every optional stage is quality-checked with DNSMOS: if OVRL or SIG decreases by more than 0.05, the stage is automatically skipped.

## Project Structure

```
audio-restorer/
├── run.py                     # Main entry point
├── config.py                  # Configuration, paths, presets
├── requirements.txt           # Python dependencies
├── src/
│   ├── pipeline.py            # Main orchestrator
│   ├── deepfilter_enhancer.py # DeepFilterNet neural denoising
│   ├── ml_enhancer.py         # PyTorch ML enhancement (CPU fallback)
│   ├── audio_enhancer.py      # SimpleEnhancer (ffmpeg fallback)
│   ├── audio_utils.py         # Shared I/O utilities
│   ├── downloader.py          # YouTube downloading (yt-dlp)
│   ├── sota_metrics.py        # DNSMOS, PESQ, STOI metrics
│   ├── dereverb_enhancer.py   # NARA-WPE de-reverberation
│   ├── diarization.py         # Speaker diarization
│   ├── speaker_isolation.py   # Main speaker extraction
│   ├── distance_enhancer.py   # Distance-robust gain/EQ
│   ├── speaker_agc.py         # Per-speaker AGC
│   ├── deesser.py             # Sibilance reduction
│   ├── hum_remover.py         # 50/60Hz notch filter
│   ├── click_remover.py       # Transient artifact removal
│   ├── comfort_noise.py       # Pink noise for silence regions
│   ├── voicefixer_enhancer.py # VoiceFixer speech super-resolution
│   └── video_merger.py        # Audio+video merging
├── tests/
│   ├── test_modules.py        # Unit tests (synthetic audio, no network)
│   ├── test_pipeline.py       # Pipeline integration tests
│   └── reference_videos.json  # Test video definitions
├── output/                    # Enhanced files
├── temp/                      # Temporary/cached files
└── benchmarks/                # Quality benchmarks
```

## Requirements

- Python 3.8+
- ffmpeg (system install or local binary)
- GPU recommended for DeepFilterNet (CPU fallback available)

```bash
pip install -r requirements.txt
```

## Benchmark Findings

Extensive DNSMOS benchmarking across 5 diverse conference recordings established what works and what doesn't:

**Effective:**
- DeepFilterNet at full strength (1.0) with unlimited suppression (`atten_lim_db=None`)
- Two-pass EBU R128 loudness normalization (accurate to within 0.5 LUFS)
- Per-stage DNSMOS quality monitoring (auto-skips degrading stages)

**Neutral (no measurable impact):**
- De-essing (no sibilance issues on typical conference audio)
- Comfort noise (negligible effect on DNSMOS)

**Harmful on already-enhanced audio (auto-skipped by quality check):**
- NARA-WPE dereverberation: OVRL -1.30 (DeepFilterNet handles reverb implicitly)
- Hum removal: OVRL -0.25 (false positives on clean audio)
- Click removal: OVRL -0.23 (detects speech transients as clicks)
- Strength mixing (< 1.0): all metrics degrade
- `atten_lim_db` 12-20 dB: OVRL -0.46 to -0.99

**SIG ceiling:** Speech signal quality (SIG ~2.96 for noisy sources, ~3.39 mean) is limited by source material quality. Improving beyond this requires generative models.

## Python API

```python
from src.pipeline import AudioRestorationPipeline
from config import TEMP_DIR, OUTPUT_DIR

pipeline = AudioRestorationPipeline(
    temp_dir=TEMP_DIR,
    output_dir=OUTPUT_DIR,
    enhancer_type="deepfilter",
)

result = pipeline.restore(
    url="https://youtu.be/VIDEO_ID",
    audio_only=True,
)

if result.success:
    print(f"Enhanced audio: {result.enhanced_audio}")
```

## Testing

```bash
source venv/bin/activate

# Unit tests (34+ tests, ~48s)
python -m pytest tests/test_modules.py -v

# Process reference video
python run.py "https://youtu.be/cglDoG0GzyA" --audio-only --keep-temp

# Measure DNSMOS quality
python tests/sota_benchmark.py output/audio_cglDoG0GzyA_enhanced.wav
```

## License

MIT License

## Acknowledgments

- [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) - Neural noise suppression
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - YouTube downloading
- [ffmpeg](https://ffmpeg.org/) - Audio/video processing
- [SpeechBrain](https://speechbrain.github.io/) - Speech processing toolkit
- [NARA-WPE](https://github.com/fgnt/nara_wpe) - De-reverberation
