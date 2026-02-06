"""
DeepFilterNet-based Audio Enhancement
=====================================

Uses pre-trained neural network for state-of-the-art noise suppression.
DeepFilterNet is trained on thousands of hours of noisy speech and provides
superior noise reduction compared to spectral gating methods.

Requirements:
    pip install deepfilternet

Usage:
    from src.deepfilter_enhancer import DeepFilterNetEnhancer

    enhancer = DeepFilterNetEnhancer()
    enhancer.enhance(input_path, output_path)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import soundfile as sf
import warnings
from typing import Optional
from .audio_utils import load_mono_audio, save_audio, prevent_clipping, stitch_chunks

# Suppress torchaudio deprecation warning from df library
warnings.filterwarnings("ignore", message=".*torchaudio.backend.common.*")


class DeepFilterNetEnhancer:
    """
    Neural network-based speech enhancement using DeepFilterNet.

    DeepFilterNet3 is a state-of-the-art deep learning model specifically
    designed for speech enhancement. It provides significantly better
    noise reduction than traditional spectral gating methods.

    Features:
    - Pre-trained on thousands of hours of noisy speech
    - Handles various noise types (background chatter, hum, hiss, etc.)
    - GPU acceleration when available
    - Minimal speech distortion

    Note:
        DeepFilterNet operates at 48kHz internally. Audio is automatically
        resampled to/from 48kHz during processing.
    """

    # DeepFilterNet's native sample rate
    NATIVE_SR = 48000

    def __init__(self,
                 noise_reduction_strength: float = 0.8,
                 use_gpu: bool = True,
                 verbose: bool = True,
                 post_filter: bool = False):
        """
        Initialize DeepFilterNet enhancer.

        Args:
            noise_reduction_strength: Controls how much of the enhanced audio
                to mix with original (0.0=original, 1.0=fully enhanced).
                Lower values preserve more speech quality at cost of noise.
            use_gpu: Use GPU acceleration if available
            verbose: Print progress messages
            post_filter: Apply additional post-filtering (can help with
                residual noise but may affect speech quality)
        """
        self.noise_reduction_strength = noise_reduction_strength
        self.use_gpu = use_gpu
        self.verbose = verbose
        self.post_filter = post_filter

        self.model = None
        self.df_state = None
        self._model_sr = None

    def _load_model(self):
        """Load DeepFilterNet model (lazy loading on first use)."""
        if self.model is not None:
            return

        try:
            from df import init_df
            from df.enhance import enhance as df_enhance

            if self.verbose:
                print("  Loading DeepFilterNet model...")

            # Suppress logging during init
            import logging
            df_logger = logging.getLogger("DF")
            old_level = df_logger.level
            if not self.verbose:
                df_logger.setLevel(logging.WARNING)

            self.model, self.df_state, _ = init_df(
                post_filter=self.post_filter,
                log_level="WARNING" if not self.verbose else "INFO"
            )
            self._df_enhance = df_enhance
            self._model_sr = self.df_state.sr()

            df_logger.setLevel(old_level)

            if self.verbose:
                device = "GPU" if next(self.model.parameters()).is_cuda else "CPU"
                print(f"  DeepFilterNet loaded ({device}, {self._model_sr}Hz)")

        except ImportError as e:
            raise ImportError(
                "DeepFilterNet not installed. Install with:\n"
                "  pip install deepfilternet\n"
                f"Original error: {e}"
            )

    @property
    def name(self) -> str:
        return "DeepFilterNet"

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio using librosa."""
        if orig_sr == target_sr:
            return audio
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

    def enhance(self, input_path: Path, output_path: Path, target_sr: int = 48000) -> Path:
        """
        Enhance audio using DeepFilterNet neural network.

        Note: target_sr is accepted for interface compatibility but DeepFilterNet
        always processes at 48kHz internally. Output matches input sample rate.

        The audio is automatically resampled to 48kHz for processing,
        then resampled back to the original sample rate.

        For long files, processes in chunks to avoid GPU memory issues.

        Args:
            input_path: Path to input audio file
            output_path: Path to save enhanced audio
            target_sr: Ignored (kept for interface compatibility)

        Returns:
            Path to enhanced audio file
        """
        # Lazy load model
        self._load_model()

        if self.verbose:
            print(f"\n🧠 DeepFilterNet Enhancement: {input_path.name}")

        # Load audio
        audio, orig_sr = load_mono_audio(input_path, verbose=self.verbose)

        # Store original audio for strength mixing (before processing)
        original_audio = audio.copy()

        # Resample to DeepFilterNet's native rate (48kHz)
        if orig_sr != self.NATIVE_SR:
            if self.verbose:
                print(f"  Resampling: {orig_sr}Hz -> {self.NATIVE_SR}Hz")
            audio = self._resample(audio, orig_sr, self.NATIVE_SR)
            # Also resample original for mixing later
            original_audio = self._resample(original_audio, orig_sr, self.NATIVE_SR)

        # Process in chunks to avoid GPU memory issues
        # 30 seconds per chunk with 1 second overlap for smooth transitions
        chunk_seconds = 30
        overlap_seconds = 1
        chunk_samples = chunk_seconds * self.NATIVE_SR
        overlap_samples = overlap_seconds * self.NATIVE_SR

        total_samples = len(audio)
        num_chunks = max(1, int(np.ceil((total_samples - overlap_samples) / (chunk_samples - overlap_samples))))

        if self.verbose:
            if num_chunks > 1:
                print(f"  Processing in {num_chunks} chunks ({chunk_seconds}s each)...")
            else:
                print("  Applying neural denoising...")

        enhanced_chunks = []

        for i in range(num_chunks):
            start = i * (chunk_samples - overlap_samples)
            end = min(start + chunk_samples, total_samples)
            chunk = audio[start:end]

            if self.verbose and num_chunks > 1:
                progress = (i + 1) / num_chunks * 100
                print(f"  Chunk {i+1}/{num_chunks} ({progress:.0f}%)...", end='\r')

            # Convert to torch tensor
            chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0)

            # Move to GPU if available and requested
            if self.use_gpu and torch.cuda.is_available():
                chunk_tensor = chunk_tensor.cuda()

            # Apply DeepFilterNet
            with torch.no_grad():
                enhanced_chunk = self._df_enhance(
                    self.model,
                    self.df_state,
                    chunk_tensor
                )

            # Move back to CPU and convert to numpy
            enhanced_chunk = enhanced_chunk.squeeze().cpu().numpy()
            enhanced_chunks.append(enhanced_chunk)

            # Clear GPU cache
            if self.use_gpu and torch.cuda.is_available():
                torch.cuda.empty_cache()

        if self.verbose and num_chunks > 1:
            print()  # New line after progress

        # Stitch chunks together with crossfade
        if len(enhanced_chunks) == 1:
            enhanced = enhanced_chunks[0]
        else:
            enhanced = stitch_chunks(enhanced_chunks, overlap_samples)

        if self.verbose:
            print("  Neural denoising complete")

        # Apply noise reduction strength (mix enhanced with original)
        # strength=1.0 means fully enhanced, strength=0.5 means 50% mix
        if self.noise_reduction_strength < 1.0:
            strength = self.noise_reduction_strength
            # Ensure original_audio matches enhanced length (may differ slightly)
            min_len = min(len(enhanced), len(original_audio))
            enhanced = enhanced[:min_len] * strength + original_audio[:min_len] * (1 - strength)
            if self.verbose:
                print(f"  Applied strength mixing: {strength*100:.0f}% enhanced, {(1-strength)*100:.0f}% original")

        # Resample back to original rate if needed
        if orig_sr != self.NATIVE_SR:
            if self.verbose:
                print(f"  Resampling: {self.NATIVE_SR}Hz -> {orig_sr}Hz")
            enhanced = self._resample(enhanced, self.NATIVE_SR, orig_sr)

        enhanced = prevent_clipping(enhanced, verbose=self.verbose)
        save_audio(enhanced, output_path, orig_sr)

        if self.verbose:
            print(f"  ✓ Saved: {output_path}")

        return output_path


if __name__ == "__main__":
    # Test the enhancer
    import tempfile

    print("Testing DeepFilterNetEnhancer...")

    # Create test audio with noise
    sr = 48000
    duration = 3  # seconds
    t = np.linspace(0, duration, sr * duration)

    # Speech-like signal (simple sine waves)
    speech = np.sin(2 * np.pi * 200 * t) * 0.3  # Fundamental
    speech += np.sin(2 * np.pi * 400 * t) * 0.2  # Harmonic
    speech += np.sin(2 * np.pi * 800 * t) * 0.1  # Harmonic

    # Add noise
    noise = np.random.randn(len(t)) * 0.1
    noisy_audio = speech + noise

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "test_noisy.wav"
        output_path = Path(tmpdir) / "test_enhanced.wav"

        sf.write(input_path, noisy_audio.astype(np.float32), sr)

        enhancer = DeepFilterNetEnhancer(verbose=True)
        enhancer.enhance(input_path, output_path)

        # Verify output exists
        assert output_path.exists(), "Output file not created"

        # Load and check
        enhanced, _ = sf.read(output_path)
        print(f"\n  Input RMS:  {np.sqrt(np.mean(noisy_audio**2)):.4f}")
        print(f"  Output RMS: {np.sqrt(np.mean(enhanced**2)):.4f}")

        print("\n✅ DeepFilterNet test passed!")
