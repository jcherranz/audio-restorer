"""
Comfort Noise Generator Module
==============================

Adds subtle, spectrally-shaped noise during silence to avoid the "dead air"
effect that can occur after aggressive noise reduction.

Why comfort noise matters:
- Aggressive noise reduction can create unnatural "dead silence"
- Sudden transitions from noise to silence are jarring
- Comfort noise masks residual processing artifacts
- Creates a more natural listening experience

Implementation generates pink noise (1/f spectrum) and inserts it during
detected silence regions with smooth crossfades.

Usage:
    from src.comfort_noise import ComfortNoiseGenerator

    generator = ComfortNoiseGenerator()
    generator.process(input_path, output_path)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from typing import List, Tuple
from .audio_utils import load_mono_audio, save_audio, prevent_clipping


class ComfortNoiseGenerator:
    """
    Generates and inserts comfort noise during silence regions.

    Uses pink noise (1/f spectrum) which sounds more natural than white
    noise for speech applications.
    """

    def __init__(self,
                 target_level_db: float = -60.0,
                 silence_threshold_db: float = -45.0,
                 min_silence_ms: float = 100.0,
                 crossfade_ms: float = 20.0,
                 spectrum_shape: str = "pink",
                 verbose: bool = True):
        """
        Initialize comfort noise generator.

        Args:
            target_level_db: RMS level of comfort noise (very quiet).
                -60 dB is barely perceptible but fills silence.
            silence_threshold_db: Level below which audio is considered silence.
                Should be set relative to speech level.
            min_silence_ms: Minimum silence duration to add comfort noise.
                Very short silences don't need filling.
            crossfade_ms: Transition time into/out of comfort noise.
                Smooth crossfade prevents audible clicks.
            spectrum_shape: Noise spectrum type ("pink" or "white").
                Pink noise sounds more natural for speech.
            verbose: Print progress messages.
        """
        self.target_level_db = target_level_db
        self.silence_threshold_db = silence_threshold_db
        self.min_silence_ms = min_silence_ms
        self.crossfade_ms = crossfade_ms
        self.spectrum_shape = spectrum_shape
        self.verbose = verbose

    @property
    def name(self) -> str:
        return "ComfortNoiseGenerator"

    def _generate_pink_noise(self, num_samples: int) -> np.ndarray:
        """
        Generate pink (1/f) noise.

        Pink noise has equal energy per octave, making it sound more
        natural than white noise for speech applications.

        Uses Voss-McCartney algorithm for efficient pink noise generation.

        Args:
            num_samples: Number of samples to generate.

        Returns:
            Pink noise samples normalized to unit variance.
        """
        # Voss-McCartney algorithm: sum of multiple octave-band noise sources
        # Each source updates at half the rate of the previous
        num_rows = 16  # Number of octaves

        # Initialize random sources
        rows = np.random.randn(num_rows, num_samples)

        # Create update masks (each row updates at half the rate of previous)
        output = np.zeros(num_samples)

        for i in range(num_rows):
            # Row i updates every 2^i samples
            step = 2 ** i
            mask = np.zeros(num_samples)
            mask[::step] = 1

            # Cumulative sum with resets creates held values
            held = np.zeros(num_samples)
            current = rows[i, 0]
            for j in range(num_samples):
                if j % step == 0:
                    current = rows[i, j]
                held[j] = current

            output += held

        # Normalize to unit variance
        output = output / np.std(output)

        return output

    def _generate_white_noise(self, num_samples: int) -> np.ndarray:
        """
        Generate white noise.

        Args:
            num_samples: Number of samples to generate.

        Returns:
            White noise samples normalized to unit variance.
        """
        return np.random.randn(num_samples)

    def _detect_silence_regions(self, audio: np.ndarray, sr: int) -> List[Tuple[int, int]]:
        """
        Detect silence regions in audio.

        Args:
            audio: Audio samples.
            sr: Sample rate.

        Returns:
            List of (start, end) tuples for silence regions.
        """
        # Calculate frame-based energy
        frame_ms = 20  # 20ms frames
        frame_samples = int(frame_ms / 1000 * sr)
        hop_samples = frame_samples // 2

        # Calculate RMS for each frame
        num_frames = (len(audio) - frame_samples) // hop_samples + 1
        frame_db = np.zeros(num_frames)

        for i in range(num_frames):
            start = i * hop_samples
            end = start + frame_samples
            frame = audio[start:end]
            rms = np.sqrt(np.mean(frame ** 2))
            frame_db[i] = 20 * np.log10(rms + 1e-10)

        # Find frames below threshold
        silence_mask = frame_db < self.silence_threshold_db

        # Convert to sample-based regions
        regions = []
        in_silence = False
        silence_start = 0
        min_frames = int(self.min_silence_ms / frame_ms)

        for i in range(len(silence_mask)):
            if silence_mask[i] and not in_silence:
                in_silence = True
                silence_start = i
            elif not silence_mask[i] and in_silence:
                in_silence = False
                silence_end = i

                # Only keep if long enough
                if silence_end - silence_start >= min_frames:
                    start_sample = silence_start * hop_samples
                    end_sample = min(silence_end * hop_samples + frame_samples, len(audio))
                    regions.append((start_sample, end_sample))

        # Handle silence at end
        if in_silence and len(silence_mask) - silence_start >= min_frames:
            start_sample = silence_start * hop_samples
            end_sample = len(audio)
            regions.append((start_sample, end_sample))

        return regions

    def process(self, input_path: Path, output_path: Path,
                target_sr: int = None) -> Path:
        """
        Add comfort noise to silence regions in audio.

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
            print(f"\n🔇 Comfort Noise: {input_path.name}")

        # Load audio
        audio, sr = load_mono_audio(input_path, verbose=self.verbose)

        if self.verbose:
            print(f"  Noise type: {self.spectrum_shape}")
            print(f"  Target level: {self.target_level_db:.1f} dB")

        # Detect silence regions
        if self.verbose:
            print("  Detecting silence regions...")

        silence_regions = self._detect_silence_regions(audio, sr)

        if self.verbose:
            total_silence = sum(end - start for start, end in silence_regions)
            silence_duration = total_silence / sr
            print(f"  Found {len(silence_regions)} silence region(s) ({silence_duration:.1f}s total)")

        if not silence_regions:
            # No silence detected, just copy the file
            if self.verbose:
                print("  No silence regions found, saving unchanged")
            save_audio(audio, output_path, sr)
            return output_path

        # Generate comfort noise for entire audio length
        if self.verbose:
            print("  Generating comfort noise...")

        if self.spectrum_shape == "pink":
            noise = self._generate_pink_noise(len(audio))
        else:
            noise = self._generate_white_noise(len(audio))

        # Scale noise to target level
        target_amplitude = 10 ** (self.target_level_db / 20)
        noise = noise * target_amplitude

        # Create output
        output = audio.copy()

        # Add comfort noise to silence regions with crossfade
        crossfade_samples = int(self.crossfade_ms / 1000 * sr)

        if self.verbose:
            print("  Inserting comfort noise...")

        for start, end in silence_regions:
            region_length = end - start

            if region_length <= 2 * crossfade_samples:
                # Region too short for crossfade, just add noise
                output[start:end] = audio[start:end] + noise[start:end]
            else:
                # Crossfade in
                fade_in = np.linspace(0, 1, crossfade_samples)
                output[start:start + crossfade_samples] = (
                    audio[start:start + crossfade_samples] +
                    noise[start:start + crossfade_samples] * fade_in
                )

                # Full noise in middle
                mid_start = start + crossfade_samples
                mid_end = end - crossfade_samples
                output[mid_start:mid_end] = audio[mid_start:mid_end] + noise[mid_start:mid_end]

                # Crossfade out
                fade_out = np.linspace(1, 0, crossfade_samples)
                output[end - crossfade_samples:end] = (
                    audio[end - crossfade_samples:end] +
                    noise[end - crossfade_samples:end] * fade_out
                )

        if self.verbose:
            print(f"  Added noise to {len(silence_regions)} region(s)")

        output = prevent_clipping(output, verbose=self.verbose)
        save_audio(output, output_path, sr)

        if self.verbose:
            print(f"  ✓ Saved: {output_path}")

        return output_path


def quick_comfort_noise(input_path: Path,
                        output_path: Path = None,
                        level_db: float = -60.0,
                        verbose: bool = True) -> Path:
    """
    Quick one-liner for adding comfort noise.

    Args:
        input_path: Input audio file.
        output_path: Output file (default: input_comfort.wav).
        level_db: Comfort noise level.
        verbose: Print progress.

    Returns:
        Path to output file.
    """
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_comfort.wav"

    generator = ComfortNoiseGenerator(target_level_db=level_db, verbose=verbose)
    return generator.process(input_path, output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Add comfort noise to silence regions")
    parser.add_argument("input", type=Path, help="Input audio file")
    parser.add_argument("-o", "--output", type=Path, help="Output audio file")
    parser.add_argument("--level", type=float, default=-60.0,
                        help="Comfort noise level in dB (default: -60)")
    parser.add_argument("--threshold", type=float, default=-45.0,
                        help="Silence threshold in dB (default: -45)")
    parser.add_argument("--min-silence", type=float, default=100.0,
                        help="Minimum silence duration in ms (default: 100)")
    parser.add_argument("--noise-type", choices=["pink", "white"], default="pink",
                        help="Noise spectrum type (default: pink)")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")

    args = parser.parse_args()

    output = args.output or args.input.parent / f"{args.input.stem}_comfort.wav"

    generator = ComfortNoiseGenerator(
        target_level_db=args.level,
        silence_threshold_db=args.threshold,
        min_silence_ms=args.min_silence,
        spectrum_shape=args.noise_type,
        verbose=not args.quiet
    )

    generator.process(args.input, output)
    print(f"\n✅ Comfort noise added: {output}")
