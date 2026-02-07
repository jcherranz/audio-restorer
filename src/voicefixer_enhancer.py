"""
VoiceFixer Speech Super-Resolution
====================================

Generative model that reconstructs damaged speech, potentially improving
SIG (speech signal quality) beyond what discriminative models like
DeepFilterNet can achieve.

VoiceFixer uses a neural vocoder to re-synthesize speech, which can restore
bandwidth, fix distortion, and improve perceived quality.

Requirements:
    pip install voicefixer

Usage:
    from src.voicefixer_enhancer import VoiceFixerEnhancer

    enhancer = VoiceFixerEnhancer()
    enhancer.enhance(input_path, output_path)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from .audio_utils import load_mono_audio, save_audio, prevent_clipping


class VoiceFixerEnhancer:
    """
    Speech super-resolution using VoiceFixer generative model.

    VoiceFixer re-synthesizes speech from a degraded input, potentially
    improving SIG beyond the ceiling of discriminative denoisers.

    The model outputs at 44100 Hz; this enhancer resamples back to
    the target sample rate (48000 Hz by default) for pipeline compatibility.
    """

    VOICEFIXER_SR = 44100  # VoiceFixer native output rate

    def __init__(self, mode: int = 0, use_gpu: bool = False, verbose: bool = True):
        """
        Initialize VoiceFixer enhancer.

        Args:
            mode: VoiceFixer mode (0=general, 1=speech, 2=music).
            use_gpu: Whether to use CUDA for inference.
            verbose: Print progress messages.
        """
        self.mode = mode
        self.use_gpu = use_gpu
        self.verbose = verbose
        self._model = None

    @property
    def name(self) -> str:
        return "VoiceFixer"

    def _load_model(self):
        """Lazy-load VoiceFixer model on first use."""
        if self._model is not None:
            return

        try:
            from voicefixer import VoiceFixer
        except ImportError:
            raise ImportError(
                "VoiceFixer not installed. Install with:\n"
                "  pip install voicefixer"
            )

        if self.verbose:
            print("  Loading VoiceFixer model...")
        self._model = VoiceFixer()

    def enhance(self, input_path: Path, output_path: Path, target_sr: int = 48000) -> Path:
        """
        Enhance audio using VoiceFixer speech super-resolution.

        Args:
            input_path: Path to input audio file.
            output_path: Path to save enhanced audio.
            target_sr: Target sample rate (VoiceFixer outputs 44100 Hz,
                       resampled to target_sr for pipeline compatibility).

        Returns:
            Path to enhanced audio file.
        """
        import tempfile
        import soundfile as sf

        input_path = Path(input_path)
        output_path = Path(output_path)

        if self.verbose:
            print(f"\n  VoiceFixer super-resolution: {input_path.name}")

        self._load_model()

        # VoiceFixer works on files directly
        # Use a temp file for output since VoiceFixer writes at 44100 Hz
        with tempfile.TemporaryDirectory() as tmpdir:
            vf_output = Path(tmpdir) / "vf_out.wav"

            self._model.restore(
                input=str(input_path),
                output=str(vf_output),
                cuda=self.use_gpu,
                mode=self.mode,
            )

            # Load VoiceFixer output (44100 Hz)
            enhanced, vf_sr = sf.read(str(vf_output))
            enhanced = enhanced.astype(np.float32)

            # Ensure mono
            if enhanced.ndim > 1:
                enhanced = enhanced.mean(axis=1)

        # Resample to target SR if needed
        if vf_sr != target_sr:
            import librosa
            if self.verbose:
                print(f"  Resampling {vf_sr} Hz -> {target_sr} Hz...")
            enhanced = librosa.resample(enhanced, orig_sr=vf_sr, target_sr=target_sr)

        # Match original length for consistency
        original, orig_sr = load_mono_audio(input_path, verbose=False)
        if orig_sr != target_sr:
            import librosa
            original = librosa.resample(original, orig_sr=orig_sr, target_sr=target_sr)

        if len(enhanced) > len(original):
            enhanced = enhanced[:len(original)]
        elif len(enhanced) < len(original):
            enhanced = np.pad(enhanced, (0, len(original) - len(enhanced)))

        enhanced = prevent_clipping(enhanced, verbose=self.verbose)
        save_audio(enhanced, output_path, target_sr)

        if self.verbose:
            print(f"  Saved: {output_path}")

        return output_path
