# 🎙️ Audio Restoration Tool for YouTube Conference Videos

Transform poor-quality conference recordings into clear, intelligible audio using AI-powered noise reduction and speech enhancement.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 What This Tool Does

- **Downloads** videos from YouTube (audio-only mode recommended)
- **Enhances** audio using AI-powered processing:
  - DeepFilterNet neural noise suppression (best quality, GPU auto-selected)
  - Speaker diarization and isolation
  - Distance-robust enhancement (consistent volume)
  - Audio normalization to broadcast standards
- **Outputs** enhanced WAV or MP3 files

## 🧭 Development Standards

This project uses a layered documentation approach for AI agents:

| Layer | Document | Purpose |
|-------|----------|---------|
| Mindset | `docs/SENIOR_ENGINEER_PROMPT.md` | Behavioral principles |
| Process | `docs/WORKFLOW_ORCHESTRATION.md` | Workflow and task management |
| Project | `AGENTS.md` | Project-specific instructions |

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- ffmpeg (must be installed on your system)

### Installing ffmpeg

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
1. Download from https://ffmpeg.org/download.html
2. Extract and add to PATH

### Python Dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Basic Usage - Enhance a Video

```bash
python run.py "https://youtu.be/cglDoG0GzyA"
```

Output will be saved to `./output/video_[ID]_enhanced.mp4`

### 2. Audio Only (Faster)

```bash
python run.py "https://youtu.be/cglDoG0GzyA" --audio-only
```

### 3. Create Comparison Video

```bash
python run.py "https://youtu.be/cglDoG0GzyA" --comparison
```

Creates a side-by-side video showing before/after.

### 4. Quick Mode (No ML Models)

```bash
python run.py "https://youtu.be/cglDoG0GzyA" --quick
```

Uses only ffmpeg filters - faster but less sophisticated.

## 📁 Project Structure

```
audio-restorer/
├── run.py                 # Main entry point
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── src/                   # Core source code
│   ├── downloader.py      # YouTube video/audio downloader
│   ├── audio_enhancer.py  # Basic audio enhancement (ffmpeg)
│   ├── ml_enhancer.py     # PyTorch-based ML enhancement
│   ├── deepfilter_enhancer.py  # Neural noise suppression
│   ├── dereverb_enhancer.py    # Room echo removal
│   ├── pipeline.py        # Main orchestration
│   └── video_merger.py    # Video/audio merging (deprecated)
├── tests/                 # Test and benchmark scripts
├── tools/                 # Utility tools
│   ├── audio_compare.py   # Compare original vs enhanced
│   └── cleanup_outputs.py # Clean old test files
├── output/                # Enhanced files (created automatically)
├── temp/                  # Temporary files (auto-cleaned)
└── models/                # ML models (downloaded automatically)
```

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Noise reduction strength (0.0 = none, 1.0 = maximum)
"noise_reduction_strength": 0.8

# Enhancer type: simple, torch, torch_advanced, or deepfilter
"enhancer_type": "torch_advanced"

# Normalize audio levels
"normalize": True

# Target loudness in LUFS (broadcast standard: -16)
"target_loudness": -16
```

## 🎛️ Command Line Options

```
python run.py <youtube_url> [options]

Options:
  -h, --help            Show help message
  -o, --output NAME     Custom output filename
  --audio-only          Process audio only (no video) - RECOMMENDED
  --enhancer TYPE       Choose: simple, torch, torch_advanced, deepfilter
  --noise-reduction N   Set noise reduction 0.0-1.0

Pre-processing:
  --remove-hum          Remove power line hum (50/60Hz + harmonics)
  --remove-clicks       Remove clicks and pops (transient artifacts)

Post-processing:
  --dereverb            Remove room echo (slower, for short files)
  --diarize             Perform speaker diarization (identify speakers)
  --isolate-speaker     Isolate main speaker (removes audience/others)
  --distance-robust     Adaptive gain/EQ per speaker distance
  --speaker-agc         Per-speaker automatic gain control
  --deess               Reduce harsh sibilant sounds (/s/, /sh/)
  --comfort-noise       Add comfort noise to silence (prevents dead air)

Other:
  --comparison          Create side-by-side comparison (deprecated)
  --quick               Use only ffmpeg (faster, no ML)
  --output-dir PATH     Custom output directory
  --keep-temp           Keep temporary files
  --quiet               Reduce output verbosity
```

## 🔧 Advanced Usage

### Python API

```python
from src.pipeline import AudioRestorationPipeline
from config import TEMP_DIR, OUTPUT_DIR

# Initialize pipeline (GPU auto-detected, deepfilter used if available)
pipeline = AudioRestorationPipeline(
    temp_dir=TEMP_DIR,
    output_dir=OUTPUT_DIR,
    enhancer_type="deepfilter",  # or "torch_advanced", "simple"
    noise_reduction_strength=0.8
)

# Restore audio from a video
result = pipeline.restore(
    url="https://youtu.be/cglDoG0GzyA",
    audio_only=True  # Recommended: audio-only is faster
)

if result.success:
    print(f"Enhanced audio: {result.enhanced_audio}")
```

### Batch Processing

```python
urls = [
    "https://youtu.be/video1",
    "https://youtu.be/video2",
    "https://youtu.be/video3"
]

results = pipeline.restore_batch(urls, audio_only=True)
```

## 🧪 Enhancement Techniques

### 1. DeepFilterNet (Neural Network) - DEFAULT WITH GPU
- State-of-the-art speech enhancement
- Significant noise reduction (SNR: 49.0 dB)
- Quality Score: 115.9/100
- **Auto-selected when GPU available**
- Manual: `--enhancer deepfilter`

### 2. PyTorch + VAD (Default on CPU)
- Spectral gating with Voice Activity Detection
- Good balance of quality and speed
- Quality Score: 81.0/100
- **Auto-selected when no GPU**
- Manual: `--enhancer torch_advanced`

### 3. ffmpeg Filters (Quick Mode)
- Fast processing, basic quality
- No ML models required
- Quality Score: 66.5/100
- Use with: `--quick` or `--enhancer simple`

## 🐛 Troubleshooting

### "ffmpeg not found"
Install ffmpeg using the instructions in Requirements.

### "DeepFilterNet not available"
The tool will fall back to spectral gating. To use DeepFilterNet:
```bash
pip install deepfilternet
```

### "Video download fails"
- Check your internet connection
- Ensure the video is publicly accessible
- Some videos may have download restrictions

### "Out of memory"
For long videos, try processing in audio-only mode first:
```bash
python run.py "URL" --audio-only
```

## 📝 How It Works

```
YouTube URL
    ↓
Download Video (yt-dlp)
    ↓
Extract Audio (ffmpeg)
    ↓
Enhance Audio
├── DeepFilterNet (neural noise suppression)
├── Spectral Gating (frequency noise removal)
└── Normalization (level adjustment)
    ↓
Merge with Video (ffmpeg)
    ↓
Output Enhanced Video
```

## 🗺️ Roadmap

- [x] Basic pipeline (download → enhance → merge)
- [x] YouTube integration
- [x] DeepFilterNet neural enhancement (best quality, GPU auto-detected)
- [x] PyTorch ML enhancement with VAD
- [x] De-reverberation support
- [x] SOTA quality metrics (DNSMOS, PESQ, STOI)
- [x] Benchmark suite
- [x] Speaker diarization (Phase 3)
- [x] Speaker isolation (Phase 3)
- [x] Distance-robust enhancement (Phase 3)
- [x] Per-speaker AGC (Phase 3)
- [x] De-essing / sibilance control (Phase 4)
- [x] Hum removal (Phase 4)
- [x] Click/pop removal (Phase 4)
- [x] Comfort noise (Phase 4)
- [ ] Advanced room correction (Phase 5)
- [ ] GUI interface (future)
- [ ] Real-time preview (future)

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| **QUICKSTART.md** | Get started in 30 seconds |
| **ROADMAP.md** | Current phase and future plans |
| **ITERATION_LOG.md** | Complete history of all changes |
| **docs/QUALITY_METRICS.md** | How we measure audio quality |
| **docs/GITHUB_SETUP.md** | GitHub integration guide |
| **docs/primer.md** | Audio restoration fundamentals |

**For AI Agents:**

| Document | Purpose |
|----------|---------|
| **AGENTS.md** | Project-specific instructions (start here) |
| **docs/SENIOR_ENGINEER_PROMPT.md** | Mindset and behavioral principles |
| **docs/WORKFLOW_ORCHESTRATION.md** | Process, planning, and task management |
| **tasks/todo.md** | Current task tracking |
| **tasks/lessons.md** | Lessons learned from corrections |

## 📄 License

MIT License - Feel free to use and modify!

## 🤝 Contributing

This is an iterative project. Suggestions and improvements welcome!

## 🙏 Acknowledgments

- [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) - Neural noise suppression
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - YouTube downloading
- [ffmpeg](https://ffmpeg.org/) - Audio/video processing
- [SpeechBrain](https://speechbrain.github.io/) - Speech processing toolkit
