"""
Audio Restoration Configuration
Edit these settings to customize the restoration process
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
TEMP_DIR = BASE_DIR / "temp"
OUTPUT_DIR = BASE_DIR / "output"
MODELS_DIR = BASE_DIR / "models"

# FFmpeg path (use local if available, otherwise system)
FFMPEG_PATH = str(BASE_DIR / "ffmpeg") if (BASE_DIR / "ffmpeg").exists() else "ffmpeg"
FFPROBE_PATH = str(BASE_DIR / "ffprobe") if (BASE_DIR / "ffprobe").exists() else "ffprobe"

# Ensure directories exist
TEMP_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Audio Settings
# NOTE: Using 48kHz throughout for consistency with DeepFilterNet's native rate.
# This avoids quality loss from multiple resampling operations.
AUDIO_SETTINGS = {
    "sample_rate": 48000,        # Matches DeepFilterNet native rate
    "output_sample_rate": 48000,  # Keep 48kHz for output
    "format": "wav",             # Intermediate format
    "output_format": "mp3",      # Final audio format
    "bitrate": "192k",           # Output bitrate
}

# Enhancement Settings
def _detect_gpu():
    """Auto-detect GPU availability for optimal defaults."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

_GPU_AVAILABLE = _detect_gpu()

ENHANCEMENT = {
    # Noise reduction strength (0.0 to 1.0)
    # Higher = more aggressive noise removal, but might affect speech quality
    "noise_reduction_strength": 0.8,

    # Enhancer type to use:
    # - "simple": ffmpeg-based (fastest, basic quality)
    # - "torch": PyTorch-based spectral gating (better quality)
    # - "torch_advanced": PyTorch with VAD (good quality, CPU-friendly)
    # - "deepfilter": Neural network denoising (best quality, GPU recommended)
    # Auto-selects deepfilter if GPU available, otherwise torch_advanced
    "enhancer_type": "deepfilter" if _GPU_AVAILABLE else "torch_advanced",

    # Use GPU acceleration if available (auto-detected)
    "use_gpu": _GPU_AVAILABLE,

    # Fallback to simple enhancer if ML fails
    "fallback_to_simple": True,
    
    # Normalize audio levels
    "normalize": True,
    
    # Target loudness in LUFS (broadcast standard is -16)
    "target_loudness": -16,
    
    # High-pass filter cutoff (Hz) - removes rumble
    "highpass_freq": 100,
    
    # Low-pass filter cutoff (Hz) - removes hiss
    # 12kHz preserves consonant clarity (/s/, /f/, /th/ sounds need 8-12kHz)
    "lowpass_freq": 12000,
}

# De-reverberation Settings (Iteration 5)
DEREVERB = {
    # Enable de-reverberation (removes room echo)
    # Applied AFTER noise reduction for best results
    "enabled": False,  # Optional, off by default

    # WPE filter length (prediction filter order)
    # Higher = more reverb removal but risk of artifacts
    # Recommended: 5-20 for speech
    "taps": 10,

    # Prediction delay in frames
    # Should be at least 1, typically 2-4
    "delay": 3,

    # Number of refinement iterations
    # More = better quality but slower
    # Recommended: 2-5
    "iterations": 3,
}

# Video Settings
VIDEO_SETTINGS = {
    "video_codec": "copy",      # Copy video stream (no re-encode)
    "audio_codec": "aac",       # Audio codec for output
    "audio_bitrate": "192k",
    "container": "mp4",
}

# YouTube Download Settings
YOUTUBE_SETTINGS = {
    "format": "bestvideo[height<=1080]+bestaudio/best",
    "merge_output_format": "mp4",
}

# Logging
VERBOSE = True
KEEP_TEMP_FILES = False  # Set to True to keep intermediate files for debugging
