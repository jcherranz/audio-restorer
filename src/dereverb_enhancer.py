"""
De-reverberation using NARA-WPE Algorithm
==========================================

Removes room echo/reverb using Weighted Prediction Error algorithm.
WPE is a blind dereverberation method that uses long-term linear
prediction to estimate and remove reverb from speech signals.

Requirements:
    pip install nara-wpe

Usage:
    from src.dereverb_enhancer import DereverbEnhancer

    dereverb = DereverbEnhancer()
    dereverb.enhance(input_path, output_path)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import soundfile as sf
from typing import Optional
from .audio_utils import load_mono_audio, save_audio, prevent_clipping, stitch_chunks


class DereverbEnhancer:
    """
    De-reverberation using NARA-WPE algorithm.

    WPE (Weighted Prediction Error) is a blind dereverberation method
    that uses long-term linear prediction to estimate and remove reverb.

    .. deprecated:: Iteration 36
        DNSMOS benchmarking showed WPE is harmful on DeepFilterNet-enhanced audio:
        - Post-enhancement: mean OVRL -1.30 across 5 recordings (all degraded)
        - Pre-enhancement: OVRL -0.10, SIG still hurt
        DeepFilterNet handles reverb implicitly, making separate dereverberation
        redundant and counterproductive. The ``--dereverb`` flag is retained for
        edge cases but ``_run_stage()`` quality checks will auto-skip it.
    """

    NATIVE_SR = 48000  # Match DeepFilterNet for compatibility

    def __init__(self,
                 taps: int = 10,
                 delay: int = 3,
                 iterations: int = 3,
                 verbose: bool = True):
        """
        Initialize de-reverberator.

        Args:
            taps: Filter length (prediction filter order). Higher values
                  can remove more reverb but may introduce artifacts.
                  Recommended: 5-20 for speech.
            delay: Prediction delay in frames. Should be at least 1.
                  Recommended: 2-4 for typical room reverb.
            iterations: Number of refinement iterations. More iterations
                       give better results but slower processing.
                       Recommended: 2-5.
            verbose: Print progress messages.
        """
        self.taps = taps
        self.delay = delay
        self.iterations = iterations
        self.verbose = verbose

    @property
    def name(self) -> str:
        return "NARA-WPE Dereverberation"

    def _process_chunk(self, audio_chunk: np.ndarray, wpe_func, stft_func, istft_func, stft_options: dict) -> np.ndarray:
        """Process a single audio chunk through WPE."""
        # Convert to STFT domain
        # WPE expects shape (channels, frequencies, time)
        Y = stft_func(audio_chunk, **stft_options)
        Y = Y.T[np.newaxis, ...]  # Shape: (1, freq, time)

        # Apply WPE de-reverberation
        Z = wpe_func(
            Y,
            taps=self.taps,
            delay=self.delay,
            iterations=self.iterations,
            statistics_mode='full'
        )

        # Convert back to time domain
        z = istft_func(Z[0].T, **stft_options)

        return z

    def enhance(self, input_path: Path, output_path: Path, target_sr: int = 48000) -> Path:
        """
        Remove reverberation from audio.

        Uses the Weighted Prediction Error (WPE) algorithm to blindly
        estimate and remove room reverberation from the audio signal.

        For long files, processes in chunks to avoid memory issues.

        Args:
            input_path: Path to input audio file
            output_path: Path to save dereverberated audio
            target_sr: Target sample rate (default 48000 for DeepFilter compatibility)

        Returns:
            Path to dereverberated audio file
        """
        try:
            from nara_wpe.wpe import wpe
            from nara_wpe.utils import stft, istft
        except ImportError as e:
            raise ImportError(
                "NARA-WPE not installed. Install with:\n"
                "  pip install nara-wpe\n"
                f"Original error: {e}"
            )

        if self.verbose:
            print(f"\n🔊 De-reverberation: {input_path.name}")

        # Load audio
        audio, sr = load_mono_audio(input_path, verbose=self.verbose)

        # STFT parameters - tuned for speech
        # Size 512 @ 48kHz gives ~10.7ms frames (good for speech)
        # Shift 128 gives 75% overlap (smooth reconstruction)
        stft_options = dict(size=512, shift=128)

        # Process in chunks to avoid memory issues
        # 60 seconds per chunk with 2 second overlap for smooth transitions
        chunk_seconds = 60
        overlap_seconds = 2
        chunk_samples = chunk_seconds * sr
        overlap_samples = overlap_seconds * sr

        total_samples = len(audio)
        num_chunks = max(1, int(np.ceil((total_samples - overlap_samples) / (chunk_samples - overlap_samples))))

        if self.verbose:
            if num_chunks > 1:
                print(f"  Processing in {num_chunks} chunks ({chunk_seconds}s each)...")
            else:
                print(f"  Applying WPE (taps={self.taps}, delay={self.delay}, iter={self.iterations})...")

        dereverb_chunks = []

        for i in range(num_chunks):
            start = i * (chunk_samples - overlap_samples)
            end = min(start + chunk_samples, total_samples)
            chunk = audio[start:end]

            if self.verbose and num_chunks > 1:
                progress = (i + 1) / num_chunks * 100
                print(f"  Chunk {i+1}/{num_chunks} ({progress:.0f}%)...", end='\r')

            # Process this chunk
            dereverb_chunk = self._process_chunk(chunk, wpe, stft, istft, stft_options)

            # Match chunk length
            if len(dereverb_chunk) > len(chunk):
                dereverb_chunk = dereverb_chunk[:len(chunk)]
            elif len(dereverb_chunk) < len(chunk):
                dereverb_chunk = np.pad(dereverb_chunk, (0, len(chunk) - len(dereverb_chunk)))

            dereverb_chunks.append(dereverb_chunk.astype(np.float32))

        if self.verbose and num_chunks > 1:
            print()  # New line after progress

        # Stitch chunks together with crossfade
        if len(dereverb_chunks) == 1:
            z = dereverb_chunks[0]
        else:
            z = stitch_chunks(dereverb_chunks, overlap_samples)

        if self.verbose:
            print("  De-reverberation complete")

        # Match original length
        if len(z) > len(audio):
            z = z[:len(audio)]
        elif len(z) < len(audio):
            z = np.pad(z, (0, len(audio) - len(z)))

        z = prevent_clipping(z, verbose=self.verbose)
        save_audio(z.astype(np.float32), Path(output_path), sr)

        if self.verbose:
            print(f"  ✓ Saved: {output_path}")

        return output_path


if __name__ == "__main__":
    # Test the enhancer
    import tempfile

    print("Testing DereverbEnhancer...")

    # Create test audio with simulated reverb
    sr = 48000
    duration = 3  # seconds
    t = np.linspace(0, duration, sr * duration)

    # Speech-like signal (simple sine waves)
    speech = np.sin(2 * np.pi * 200 * t) * 0.3
    speech += np.sin(2 * np.pi * 400 * t) * 0.2
    speech += np.sin(2 * np.pi * 800 * t) * 0.1

    # Add simple reverb simulation (delayed copies)
    reverb = np.zeros_like(speech)
    delays = [int(0.02 * sr), int(0.05 * sr), int(0.1 * sr)]
    gains = [0.4, 0.2, 0.1]
    for delay, gain in zip(delays, gains):
        reverb[delay:] += speech[:-delay] * gain

    reverberant_audio = speech + reverb

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "test_reverb.wav"
        output_path = Path(tmpdir) / "test_dereverb.wav"

        sf.write(input_path, reverberant_audio.astype(np.float32), sr)

        dereverb = DereverbEnhancer(taps=10, delay=3, iterations=3, verbose=True)
        dereverb.enhance(input_path, output_path)

        # Verify output exists
        assert output_path.exists(), "Output file not created"

        # Load and check
        dereverberated, _ = sf.read(output_path)
        print(f"\n  Input RMS:  {np.sqrt(np.mean(reverberant_audio**2)):.4f}")
        print(f"  Output RMS: {np.sqrt(np.mean(dereverberated**2)):.4f}")

        print("\n✅ DereverbEnhancer test passed!")
