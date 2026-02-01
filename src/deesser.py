"""
De-Esser Module
===============

Reduces harsh sibilant sounds (/s/, /sh/, /ch/, /z/) that can be fatiguing
and reduce speech clarity in recordings.

Sibilants typically occur in:
- Male voices: 3-6 kHz
- Female voices: 6-9 kHz
- General problem zone: 4-10 kHz

Implementation uses split-band compression:
1. Extract sibilant frequency band (4-10 kHz)
2. Detect when sibilance exceeds threshold
3. Apply gain reduction during sibilant moments
4. Recombine with unprocessed frequencies

Usage:
    from src.deesser import DeEsser

    deesser = DeEsser()
    deesser.process(input_path, output_path)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import soundfile as sf
from scipy import signal
from typing import Tuple
import warnings
warnings.filterwarnings("ignore")


class DeEsser:
    """
    Split-band de-esser for sibilance control.

    Uses dynamic compression on the sibilant frequency band (4-10 kHz)
    while leaving other frequencies untouched. This reduces harshness
    without affecting overall speech quality.
    """

    def __init__(self,
                 threshold_db: float = -20.0,
                 ratio: float = 4.0,
                 attack_ms: float = 1.0,
                 release_ms: float = 50.0,
                 low_freq: int = 4000,
                 high_freq: int = 10000,
                 verbose: bool = True):
        """
        Initialize de-esser.

        Args:
            threshold_db: Level above which compression starts (dB below peak).
                Lower values = more aggressive de-essing.
            ratio: Compression ratio. 4:1 is typical for de-essing.
                Higher = more reduction.
            attack_ms: How quickly compression engages (ms).
                Fast attack catches sibilants.
            release_ms: How quickly compression releases (ms).
                Moderate release avoids pumping.
            low_freq: Lower bound of sibilant band (Hz).
            high_freq: Upper bound of sibilant band (Hz).
            verbose: Print progress messages.
        """
        self.threshold_db = threshold_db
        self.ratio = ratio
        self.attack_ms = attack_ms
        self.release_ms = release_ms
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.verbose = verbose

    @property
    def name(self) -> str:
        return "DeEsser"

    def _design_bandpass(self, sr: int, low: int, high: int,
                         order: int = 4) -> np.ndarray:
        """Design bandpass filter for sibilant extraction."""
        nyquist = sr / 2
        # Ensure frequencies are within valid range
        low_norm = min(low / nyquist, 0.99)
        high_norm = min(high / nyquist, 0.99)

        if low_norm >= high_norm:
            high_norm = min(low_norm + 0.1, 0.99)

        sos = signal.butter(order, [low_norm, high_norm],
                           btype='bandpass', output='sos')
        return sos

    def _design_bandstop(self, sr: int, low: int, high: int,
                         order: int = 4) -> np.ndarray:
        """Design bandstop filter to remove sibilant band from original."""
        nyquist = sr / 2
        low_norm = min(low / nyquist, 0.99)
        high_norm = min(high / nyquist, 0.99)

        if low_norm >= high_norm:
            high_norm = min(low_norm + 0.1, 0.99)

        sos = signal.butter(order, [low_norm, high_norm],
                           btype='bandstop', output='sos')
        return sos

    def _envelope_follower(self, audio: np.ndarray, sr: int,
                           attack_ms: float, release_ms: float) -> np.ndarray:
        """
        Create amplitude envelope for gain calculation.

        Uses asymmetric smoothing with fast attack and slower release.
        """
        # Convert to samples
        attack_samples = max(1, int(attack_ms / 1000 * sr))
        release_samples = max(1, int(release_ms / 1000 * sr))

        # Calculate attack and release coefficients
        attack_coef = np.exp(-1.0 / attack_samples)
        release_coef = np.exp(-1.0 / release_samples)

        # Rectified signal (absolute value)
        rectified = np.abs(audio)

        # Envelope following
        envelope = np.zeros_like(rectified)
        envelope[0] = rectified[0]

        for i in range(1, len(rectified)):
            if rectified[i] > envelope[i-1]:
                # Attack: fast rise
                envelope[i] = attack_coef * envelope[i-1] + (1 - attack_coef) * rectified[i]
            else:
                # Release: slow fall
                envelope[i] = release_coef * envelope[i-1] + (1 - release_coef) * rectified[i]

        return envelope

    def _calculate_gain_reduction(self, envelope: np.ndarray,
                                  threshold_linear: float,
                                  ratio: float) -> np.ndarray:
        """
        Calculate gain reduction based on envelope and threshold.

        Uses soft-knee compression formula.
        """
        # Convert envelope to dB
        envelope_db = 20 * np.log10(envelope + 1e-10)
        threshold_db = 20 * np.log10(threshold_linear + 1e-10)

        # Calculate gain reduction (in dB)
        # Above threshold: reduce by (1 - 1/ratio) * overshoot
        overshoot_db = np.maximum(0, envelope_db - threshold_db)
        gain_reduction_db = -overshoot_db * (1 - 1/ratio)

        # Convert back to linear
        gain = 10 ** (gain_reduction_db / 20)

        return gain

    def process(self, input_path: Path, output_path: Path,
                target_sr: int = None) -> Path:
        """
        Apply de-essing to audio file.

        Args:
            input_path: Path to input audio file.
            output_path: Path to save processed audio.
            target_sr: Ignored (kept for interface compatibility).

        Returns:
            Path to output file.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if self.verbose:
            print(f"\n🎤 De-Essing: {input_path.name}")

        # Load audio
        audio, sr = sf.read(str(input_path), dtype='float32')

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
            if self.verbose:
                print("  Converted to mono")

        if self.verbose:
            duration = len(audio) / sr
            print(f"  Duration: {duration:.1f}s at {sr}Hz")
            print(f"  Sibilant band: {self.low_freq}-{self.high_freq} Hz")

        # Check if sample rate supports the frequency range
        nyquist = sr / 2
        if self.high_freq >= nyquist:
            if self.verbose:
                print(f"  Adjusting high_freq to {int(nyquist * 0.9)}Hz (below Nyquist)")
            effective_high = int(nyquist * 0.9)
        else:
            effective_high = self.high_freq

        # Design filters
        bandpass_sos = self._design_bandpass(sr, self.low_freq, effective_high)
        bandstop_sos = self._design_bandstop(sr, self.low_freq, effective_high)

        # Extract sibilant band
        sibilant_band = signal.sosfilt(bandpass_sos, audio)

        # Get the rest of the signal (everything except sibilant band)
        rest_of_signal = signal.sosfilt(bandstop_sos, audio)

        # Calculate envelope of sibilant band
        envelope = self._envelope_follower(sibilant_band, sr,
                                          self.attack_ms, self.release_ms)

        # Calculate threshold in linear scale
        # Threshold is relative to signal peak
        peak_level = np.max(np.abs(sibilant_band))
        threshold_linear = peak_level * (10 ** (self.threshold_db / 20))

        # Calculate gain reduction
        gain = self._calculate_gain_reduction(envelope, threshold_linear, self.ratio)

        # Apply gain to sibilant band
        sibilant_reduced = sibilant_band * gain

        # Recombine
        output = rest_of_signal + sibilant_reduced

        # Calculate reduction statistics
        original_sibilant_rms = np.sqrt(np.mean(sibilant_band**2))
        reduced_sibilant_rms = np.sqrt(np.mean(sibilant_reduced**2))
        reduction_db = 20 * np.log10(reduced_sibilant_rms / (original_sibilant_rms + 1e-10))

        if self.verbose:
            avg_gain_reduction = 20 * np.log10(np.mean(gain) + 1e-10)
            print(f"  Threshold: {self.threshold_db:.1f} dB, Ratio: {self.ratio:.1f}:1")
            print(f"  Sibilance reduction: {reduction_db:.1f} dB")
            print(f"  Average gain reduction: {avg_gain_reduction:.1f} dB")

        # Normalize to prevent clipping
        max_val = np.max(np.abs(output))
        if max_val > 0.95:
            output = output * (0.95 / max_val)
            if self.verbose:
                print("  Normalized to prevent clipping")

        # Save output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), output, sr)

        if self.verbose:
            print(f"  ✓ Saved: {output_path}")

        return output_path


def quick_deess(input_path: Path,
                output_path: Path = None,
                threshold_db: float = -20.0,
                verbose: bool = True) -> Path:
    """
    Quick one-liner for de-essing.

    Args:
        input_path: Input audio file.
        output_path: Output file (default: input_deessed.wav).
        threshold_db: De-essing threshold.
        verbose: Print progress.

    Returns:
        Path to output file.
    """
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_deessed.wav"

    deesser = DeEsser(threshold_db=threshold_db, verbose=verbose)
    return deesser.process(input_path, output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="De-ess audio file")
    parser.add_argument("input", type=Path, help="Input audio file")
    parser.add_argument("-o", "--output", type=Path, help="Output audio file")
    parser.add_argument("--threshold", type=float, default=-20.0,
                        help="Threshold in dB (default: -20)")
    parser.add_argument("--ratio", type=float, default=4.0,
                        help="Compression ratio (default: 4.0)")
    parser.add_argument("--low-freq", type=int, default=4000,
                        help="Low frequency bound (default: 4000 Hz)")
    parser.add_argument("--high-freq", type=int, default=10000,
                        help="High frequency bound (default: 10000 Hz)")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")

    args = parser.parse_args()

    output = args.output or args.input.parent / f"{args.input.stem}_deessed.wav"

    deesser = DeEsser(
        threshold_db=args.threshold,
        ratio=args.ratio,
        low_freq=args.low_freq,
        high_freq=args.high_freq,
        verbose=not args.quiet
    )

    deesser.process(args.input, output)
    print(f"\n✅ De-essing complete: {output}")
