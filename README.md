# 🎙️ Audio Restoration Tool for YouTube Conference Videos

Transform poor-quality conference recordings into clear, intelligible audio using AI-powered noise reduction and speech enhancement.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 What This Tool Does

- **Downloads** videos from YouTube
- **Extracts** audio for processing
- **Enhances** audio using:
  - DeepFilterNet (neural noise suppression)
  - Spectral gating (frequency-based noise removal)
  - Audio normalization
- **Recombines** enhanced audio with original video
- **Creates** side-by-side comparison videos

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
  --dereverb            Remove room echo (slower, for short files)
  --diarize             Perform speaker diarization (identify speakers)
  --isolate-speaker     Isolate main speaker (removes audience/others)
  --comparison          Create side-by-side comparison (deprecated)
  --quick               Use only ffmpeg (faster, no ML)
  --noise-reduction N   Set noise reduction 0.0-1.0
  --output-dir PATH     Custom output directory
  --keep-temp           Keep temporary files
  --quiet               Reduce output verbosity
```

## 🔧 Advanced Usage

### Python API

```python
from src.pipeline import AudioRestorationPipeline
from config import TEMP_DIR, OUTPUT_DIR

# Initialize pipeline
pipeline = AudioRestorationPipeline(
    temp_dir=TEMP_DIR,
    output_dir=OUTPUT_DIR,
    use_deepfilternet=True,
    noise_reduction_strength=0.8
)

# Restore a video
result = pipeline.restore(
    url="https://youtu.be/cglDoG0GzyA",
    audio_only=False,
    create_comparison=True
)

if result.success:
    print(f"Enhanced video: {result.enhanced_video}")
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

### 1. DeepFilterNet (Neural Network) - BEST QUALITY
- State-of-the-art speech enhancement
- Significant noise reduction (SNR: 49.0 dB)
- Quality Score: 115.9/100
- Use with: `--enhancer deepfilter`

### 2. PyTorch + VAD (Default)
- Spectral gating with Voice Activity Detection
- Good balance of quality and speed
- Quality Score: 81.0/100
- Use with: `--enhancer torch_advanced` (or default)

### 3. ffmpeg Filters (Quick Mode)
- Fast processing, basic quality
- High/low pass filters
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
- [x] DeepFilterNet neural enhancement
- [x] PyTorch ML enhancement with VAD
- [x] De-reverberation support
- [x] SOTA quality metrics (DNSMOS, PESQ, STOI)
- [x] Benchmark suite
- [x] Speaker diarization (Phase 3)
- [x] Speaker isolation (Phase 3)
- [ ] Distance-robust enhancement (Phase 3 - Next)
- [ ] GUI interface (future)
- [ ] Real-time preview (future)

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| **QUICKSTART.md** | Get started in 30 seconds |
| **AGENTS.md** | Guidelines for LLM agents working on this project |
| **ROADMAP.md** | Current phase and future plans |
| **ITERATION_LOG.md** | Complete history of all changes |
| **docs/QUALITY_METRICS.md** | How we measure audio quality |
| **docs/GITHUB_SETUP.md** | GitHub integration guide |

## 📄 License

MIT License - Feel free to use and modify!

## 🤝 Contributing

This is an iterative project. Suggestions and improvements welcome!

## 🙏 Acknowledgments

- [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) - Neural noise suppression
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - YouTube downloading
- [ffmpeg](https://ffmpeg.org/) - Audio/video processing
- [SpeechBrain](https://speechbrain.github.io/) - Speech processing toolkit
