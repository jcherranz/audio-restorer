"""
Hum Remover Module
==================

Removes power line hum (50/60 Hz) and its harmonics from audio recordings.

Power line hum is caused by:
- Electromagnetic interference from power lines
- Ground loops in audio equipment
- Fluorescent lighting
- HVAC systems

The hum appears at the fundamental frequency (50 Hz in Europe/Asia, 60 Hz in
Americas) and its harmonics (100/120, 150/180, 200/240 Hz, etc.).

Implementation uses cascaded IIR notch filters to remove the fundamental
and several harmonics without affecting speech frequencies.

Usage:
    from src.hum_remover import HumRemover

    remover = HumRemover()
    remover.process(input_path, output_path)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import soundfile as sf
from scipy import signal
from typing import Tuple, Optional
from .audio_utils import load_mono_audio, save_audio, prevent_clipping


class HumRemover:
    """
    Removes power line hum using cascaded notch filters.

    Applies narrow notch filters at the fundamental frequency and its
    harmonics to remove hum while preserving speech quality.

    .. note:: Iteration 37 benchmark
        On DeepFilterNet-enhanced audio, hum removal measured OVRL -0.25 due to
        false positives on clean audio. The ``--remove-hum`` flag is quality-checked
        by ``_run_stage()`` and will be auto-skipped if DNSMOS degrades.
        Most useful on raw audio with visible 50/60 Hz hum before enhancement.
    """

    def __init__(self,
                 fundamental_freq: float = 60.0,
                 num_harmonics: int = 5,
                 q_factor: float = 30.0,
                 auto_detect: bool = True,
                 verbose: bool = True):
        """
        Initialize hum remover.

        Args:
            fundamental_freq: Base hum frequency (50 or 60 Hz).
                Will be auto-detected if auto_detect=True.
            num_harmonics: Number of harmonics to remove (1=fundamental only,
                5=removes 60, 120, 180, 240, 300 Hz for 60Hz hum).
            q_factor: Notch filter quality factor. Higher = narrower notch.
                30 is typical for hum removal without affecting speech.
            auto_detect: Automatically detect whether hum is 50 or 60 Hz.
            verbose: Print progress messages.
        """
        self.fundamental_freq = fundamental_freq
        self.num_harmonics = num_harmonics
        self.q_factor = q_factor
        self.auto_detect = auto_detect
        self.verbose = verbose

    @property
    def name(self) -> str:
        return "HumRemover"

    def _detect_hum_frequency(self, audio: np.ndarray, sr: int) -> float:
        """
        Auto-detect whether hum is 50 Hz or 60 Hz.

        Compares energy at 50 Hz vs 60 Hz and their first few harmonics.

        Args:
            audio: Audio samples.
            sr: Sample rate.

        Returns:
            Detected fundamental frequency (50 or 60 Hz).
        """
        # Use first 10 seconds for analysis
        analysis_samples = min(len(audio), sr * 10)
        segment = audio[:analysis_samples]

        # Compute FFT
        n_fft = 8192  # Good frequency resolution
        fft = np.abs(np.fft.rfft(segment, n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, 1/sr)

        def get_energy_at_freq(target_freq, tolerance=2):
            """Get energy in a narrow band around target frequency."""
            mask = (freqs >= target_freq - tolerance) & (freqs <= target_freq + tolerance)
            return np.sum(fft[mask])

        # Compare 50 Hz vs 60 Hz (and first 3 harmonics)
        energy_50 = sum(get_energy_at_freq(50 * (i+1)) for i in range(3))
        energy_60 = sum(get_energy_at_freq(60 * (i+1)) for i in range(3))

        detected = 50 if energy_50 > energy_60 else 60

        if self.verbose:
            ratio = energy_50 / (energy_60 + 1e-10)
            print(f"  Hum detection: 50Hz energy={energy_50:.0f}, 60Hz energy={energy_60:.0f}")
            print(f"  Detected: {detected} Hz (ratio: {ratio:.2f})")

        return detected

    def _create_notch_filter(self, freq: float, sr: int, q: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create IIR notch filter coefficients.

        Args:
            freq: Center frequency to remove.
            sr: Sample rate.
            q: Quality factor.

        Returns:
            (b, a) filter coefficients.
        """
        # Ensure frequency is below Nyquist
        if freq >= sr / 2:
            return np.array([1.0]), np.array([1.0])  # Identity filter

        b, a = signal.iirnotch(freq, q, sr)
        return b, a

    def _apply_notch_cascade(self, audio: np.ndarray, sr: int,
                             fundamental: float, num_harmonics: int,
                             q: float) -> np.ndarray:
        """
        Apply cascaded notch filters at fundamental and harmonics.

        Args:
            audio: Audio samples.
            sr: Sample rate.
            fundamental: Fundamental frequency.
            num_harmonics: Number of harmonics to filter.
            q: Quality factor.

        Returns:
            Filtered audio.
        """
        filtered = audio.copy()

        for harmonic in range(1, num_harmonics + 1):
            freq = fundamental * harmonic

            # Skip if above Nyquist
            if freq >= sr / 2:
                break

            b, a = self._create_notch_filter(freq, sr, q)

            # Apply filter (forward-backward for zero phase distortion)
            filtered = signal.filtfilt(b, a, filtered)

            if self.verbose:
                print(f"  Removed: {freq:.0f} Hz (harmonic {harmonic})")

        return filtered

    def _measure_hum_level(self, audio: np.ndarray, sr: int,
                          fundamental: float, num_harmonics: int = 3) -> float:
        """
        Measure hum level in dB.

        Args:
            audio: Audio samples.
            sr: Sample rate.
            fundamental: Fundamental frequency.
            num_harmonics: Number of harmonics to measure.

        Returns:
            Hum level in dB relative to total signal.
        """
        # Compute FFT
        n_fft = 8192
        fft = np.abs(np.fft.rfft(audio, n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, 1/sr)

        # Measure energy at hum frequencies
        hum_energy = 0
        tolerance = 3  # Hz

        for harmonic in range(1, num_harmonics + 1):
            freq = fundamental * harmonic
            if freq >= sr / 2:
                break
            mask = (freqs >= freq - tolerance) & (freqs <= freq + tolerance)
            hum_energy += np.sum(fft[mask] ** 2)

        # Total energy
        total_energy = np.sum(fft ** 2)

        # Hum level in dB
        hum_db = 10 * np.log10(hum_energy / (total_energy + 1e-10) + 1e-10)

        return hum_db

    def process(self, input_path: Path, output_path: Path,
                target_sr: int = None) -> Path:
        """
        Remove hum from audio file.

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
            print(f"\n🔌 Hum Removal: {input_path.name}")

        # Load audio
        audio, sr = load_mono_audio(input_path, verbose=self.verbose)

        # Auto-detect hum frequency if enabled
        if self.auto_detect:
            fundamental = self._detect_hum_frequency(audio, sr)
        else:
            fundamental = self.fundamental_freq

        # Measure hum level before
        hum_before = self._measure_hum_level(audio, sr, fundamental)

        if self.verbose:
            print(f"  Hum level before: {hum_before:.1f} dB")

        # Apply notch filters
        if self.verbose:
            print(f"  Applying notch filters (Q={self.q_factor}):")

        filtered = self._apply_notch_cascade(
            audio, sr, fundamental, self.num_harmonics, self.q_factor
        )

        # Measure hum level after
        hum_after = self._measure_hum_level(filtered, sr, fundamental)
        reduction = hum_before - hum_after

        if self.verbose:
            print(f"  Hum level after: {hum_after:.1f} dB")
            print(f"  Hum reduction: {reduction:.1f} dB")

        filtered = prevent_clipping(filtered, verbose=self.verbose)
        save_audio(filtered, output_path, sr)

        if self.verbose:
            print(f"  ✓ Saved: {output_path}")

        return output_path


def quick_remove_hum(input_path: Path,
                     output_path: Path = None,
                     fundamental: float = None,
                     verbose: bool = True) -> Path:
    """
    Quick one-liner for hum removal.

    Args:
        input_path: Input audio file.
        output_path: Output file (default: input_nohum.wav).
        fundamental: Hum frequency (auto-detected if None).
        verbose: Print progress.

    Returns:
        Path to output file.
    """
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_nohum.wav"

    remover = HumRemover(
        fundamental_freq=fundamental or 60.0,
        auto_detect=(fundamental is None),
        verbose=verbose
    )
    return remover.process(input_path, output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Remove power line hum from audio")
    parser.add_argument("input", type=Path, help="Input audio file")
    parser.add_argument("-o", "--output", type=Path, help="Output audio file")
    parser.add_argument("--freq", type=float, default=60.0,
                        help="Fundamental frequency (default: 60 Hz)")
    parser.add_argument("--harmonics", type=int, default=5,
                        help="Number of harmonics to remove (default: 5)")
    parser.add_argument("--q", type=float, default=30.0,
                        help="Q factor (default: 30)")
    parser.add_argument("--no-auto", action="store_true",
                        help="Disable auto-detection of hum frequency")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")

    args = parser.parse_args()

    output = args.output or args.input.parent / f"{args.input.stem}_nohum.wav"

    remover = HumRemover(
        fundamental_freq=args.freq,
        num_harmonics=args.harmonics,
        q_factor=args.q,
        auto_detect=not args.no_auto,
        verbose=not args.quiet
    )

    remover.process(args.input, output)
    print(f"\n✅ Hum removal complete: {output}")
