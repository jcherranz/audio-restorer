# 🚀 Quick Start Guide

## Your First Audio Restoration (30 seconds)

```bash
# 1. Go to project directory
cd audio-restorer

# 2. Activate the virtual environment
source venv/bin/activate

# 3. Run the restoration
python run.py "https://youtu.be/cglDoG0GzyA" --audio-only --quick
```

That's it! Your enhanced audio will be in `output/audio_cglDoG0GzyA_enhanced.wav`

---

## Common Commands

### Audio Only (Fastest)
```bash
python run.py "YOUR_YOUTUBE_URL" --audio-only --quick
```

### Full Video
```bash
python run.py "YOUR_YOUTUBE_URL"
```

### With Side-by-Side Comparison
```bash
python run.py "YOUR_YOUTUBE_URL" --comparison
```

### Adjust Noise Reduction
```bash
# More aggressive noise removal
python run.py "YOUR_YOUTUBE_URL" --noise-reduction 0.95

# Lighter touch (keep more original sound)
python run.py "YOUR_YOUTUBE_URL" --noise-reduction 0.5
```

---

## What Each Flag Does

| Flag | What it does |
|------|--------------|
| `--audio-only` | Download & enhance just the audio (faster, smaller files) |
| `--enhancer deepfilter` | Use best neural enhancer (slower, best quality) |
| `--enhancer torch_advanced` | Use default ML enhancer (balanced) |
| `--dereverb` | Remove room echo (CPU-intensive, shorter files) |
| `--quick` | Use ffmpeg filters only (no ML, faster) |
| `--comparison` | Create side-by-side before/after video |
| `--noise-reduction 0.8` | Set noise removal strength (0.0 to 1.0) |
| `--keep-temp` | Keep temporary files (for debugging) |
| `--help` | Show all options |

---

## Example Workflow

```bash
# 1. Process a conference video (audio only)
python run.py "https://youtu.be/YOUR_VIDEO" --audio-only --quick

# 2. The result is saved to output/ folder
ls output/

# 3. Listen to it!
# On Linux:
vlc output/*_enhanced.wav
# On macOS:
afplay output/*_enhanced.wav
# On Windows: Double-click the file

# 4. Not satisfied? Try with video
python run.py "https://youtu.be/YOUR_VIDEO" --comparison
```

---

## Troubleshooting

### "command not found: python"
```bash
# Try python3 instead
python3 run.py "URL"
```

### "No module named 'xxx'"
```bash
# Reinstall dependencies
source venv/bin/activate
pip install -r requirements.txt
```

### Downloads are slow
- YouTube throttles sometimes
- Try again later, or use a different video

### Audio sounds weird
- Adjust noise reduction: `--noise-reduction 0.5`
- Try without quick mode (but install torch first)

---

## Understanding the Output

After running, you'll see:
```
📁 Output files:
   Audio: /home/jcherranz/audio-restorer/output/audio_XXX_enhanced.wav
```

**What happened:**
1. Downloaded from YouTube ✓
2. Extracted the audio ✓
3. Removed background noise ✓
4. Normalized volume levels ✓
5. Saved enhanced version ✓

---

## Next Steps

1. ✅ You've already run your first restoration!
2. 🎧 Listen to the result in `output/` folder
3. 📖 Read `PROJECT_SUMMARY.md` for more details
4. 🔧 Try different enhancers: `--enhancer deepfilter` for best quality
5. 📊 Run benchmarks: `python tests/benchmark_enhancers.py`

---

**Questions? Check README.md or PROJECT_SUMMARY.md**
