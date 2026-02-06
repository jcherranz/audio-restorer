"""
Shared Audio Utilities
======================

Common audio I/O and processing functions used across modules.
Eliminates duplication of audio loading, saving, and normalization patterns.
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Tuple

# Anti-clipping peak threshold used across all modules
PEAK_LIMIT = 0.95


def load_mono_audio(path: Path, verbose: bool = False) -> Tuple[np.ndarray, int]:
    """
    Load an audio file as mono float32.

    Args:
        path: Path to audio file
        verbose: Print loading info

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    audio, sr = sf.read(str(path), dtype='float32')

    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
        if verbose:
            print("  Converted to mono")

    if verbose:
        duration = len(audio) / sr
        print(f"  Loaded: {duration:.1f}s at {sr}Hz")

    return audio, sr


def save_audio(audio: np.ndarray, path: Path, sr: int) -> None:
    """
    Save audio array to file.

    Args:
        audio: Audio data as numpy array
        path: Output file path
        sr: Sample rate
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)


def prevent_clipping(audio: np.ndarray, peak_limit: float = PEAK_LIMIT,
                     verbose: bool = False) -> np.ndarray:
    """
    Normalize audio to prevent clipping.

    Args:
        audio: Audio data
        peak_limit: Maximum peak level (default: 0.95)
        verbose: Print when normalization is applied

    Returns:
        Audio with peaks limited to peak_limit
    """
    max_val = np.max(np.abs(audio))
    if max_val > peak_limit:
        audio = audio * (peak_limit / max_val)
        if verbose:
            print("  Normalized to prevent clipping")
    return audio


def stitch_chunks(chunks: list, overlap_samples: int) -> np.ndarray:
    """
    Stitch audio chunks together with crossfade overlap.

    Args:
        chunks: List of numpy arrays (audio chunks)
        overlap_samples: Number of samples to overlap/crossfade

    Returns:
        Single stitched audio array
    """
    if len(chunks) == 1:
        return chunks[0]

    fade_in = np.linspace(0, 1, overlap_samples, dtype=np.float32)
    fade_out = np.linspace(1, 0, overlap_samples, dtype=np.float32)

    total_length = sum(len(c) for c in chunks) - overlap_samples * (len(chunks) - 1)
    result = np.zeros(total_length, dtype=np.float32)

    pos = 0
    for i, chunk in enumerate(chunks):
        if i == 0:
            result[:len(chunk) - overlap_samples] = chunk[:-overlap_samples]
            result[len(chunk) - overlap_samples:len(chunk)] = chunk[-overlap_samples:] * fade_out
            pos = len(chunk) - overlap_samples
        elif i == len(chunks) - 1:
            result[pos:pos + overlap_samples] += chunk[:overlap_samples] * fade_in
            result[pos + overlap_samples:pos + len(chunk)] = chunk[overlap_samples:]
        else:
            result[pos:pos + overlap_samples] += chunk[:overlap_samples] * fade_in
            result[pos + overlap_samples:pos + len(chunk) - overlap_samples] = chunk[overlap_samples:-overlap_samples]
            result[pos + len(chunk) - overlap_samples:pos + len(chunk)] = chunk[-overlap_samples:] * fade_out
            pos += len(chunk) - overlap_samples

    return result
