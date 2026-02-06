"""
Distance-Robust Speech Enhancement Module
Handles speakers at varying distances from the microphone.

This module estimates relative speaker distance from audio characteristics
and applies adaptive enhancement (gain, frequency correction) per segment
to achieve consistent volume and clarity across all speakers.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import timedelta
from .audio_utils import load_mono_audio, save_audio, prevent_clipping


@dataclass
class DistanceEstimate:
    """Distance estimation for an audio segment."""
    segment_start: float
    segment_end: float
    speaker: str
    relative_distance: float  # 0.0 = close, 1.0 = far
    energy_db: float
    high_freq_ratio: float
    estimated_snr: float
    reverb_ratio: float


@dataclass
class EnhancementResult:
    """Result of distance-robust enhancement."""
    output_path: Path
    segments_processed: int
    average_distance: float
    gain_adjustments: List[float]
    original_loudness_db: float
    output_loudness_db: float


class DistanceEstimator:
    """
    Estimates relative speaker distance from audio characteristics.

    Distance indicators:
    - Energy level (far = quieter)
    - High-frequency content (far = less treble due to air absorption)
    - Direct-to-reverberant ratio (far = more reverb)
    - SNR (far = lower SNR due to room noise)
    """

    def __init__(self, sample_rate: int = 44100, verbose: bool = True):
        self.sample_rate = sample_rate
        self.verbose = verbose

    def estimate_segment_distance(self, audio: np.ndarray,
                                   start: float, end: float,
                                   speaker: str = "UNKNOWN") -> DistanceEstimate:
        """
        Estimate relative distance for a single audio segment.

        Args:
            audio: Full audio array
            start: Segment start time in seconds
            end: Segment end time in seconds
            speaker: Speaker identifier

        Returns:
            DistanceEstimate with distance metrics
        """
        sr = self.sample_rate
        start_sample = int(start * sr)
        end_sample = int(end * sr)

        # Ensure bounds
        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)

        segment = audio[start_sample:end_sample]

        if len(segment) < int(0.1 * sr):  # Too short
            return DistanceEstimate(
                segment_start=start,
                segment_end=end,
                speaker=speaker,
                relative_distance=0.5,
                energy_db=-30.0,
                high_freq_ratio=0.5,
                estimated_snr=20.0,
                reverb_ratio=0.3
            )

        # 1. Energy level (RMS in dB)
        rms = np.sqrt(np.mean(segment**2))
        energy_db = 20 * np.log10(rms + 1e-10)

        # 2. High-frequency ratio (energy above 4kHz / total energy)
        fft = np.fft.rfft(segment)
        magnitude = np.abs(fft)
        freqs = np.fft.rfftfreq(len(segment), 1/sr)

        total_energy = np.sum(magnitude**2)
        high_freq_mask = freqs > 4000
        high_freq_energy = np.sum(magnitude[high_freq_mask]**2)
        high_freq_ratio = high_freq_energy / (total_energy + 1e-10)

        # 3. Estimate SNR using noise floor estimation
        # Use bottom 10% of frame energies as noise estimate
        frame_size = int(0.025 * sr)
        hop_size = int(0.01 * sr)
        frame_energies = []

        for i in range(0, len(segment) - frame_size, hop_size):
            frame = segment[i:i + frame_size]
            frame_energies.append(np.sqrt(np.mean(frame**2)))

        if frame_energies:
            frame_energies = np.array(frame_energies)
            noise_floor = np.percentile(frame_energies, 10)
            signal_level = np.percentile(frame_energies, 90)
            estimated_snr = 20 * np.log10((signal_level + 1e-10) / (noise_floor + 1e-10))
        else:
            estimated_snr = 20.0

        # 4. Reverb ratio (late energy vs early energy)
        # Simple proxy: compare energy decay in the segment
        reverb_ratio = self._estimate_reverb_ratio(segment, sr)

        # Calculate relative distance (0=close, 1=far)
        # Combine all factors with weights
        relative_distance = self._calculate_distance_score(
            energy_db, high_freq_ratio, estimated_snr, reverb_ratio
        )

        return DistanceEstimate(
            segment_start=start,
            segment_end=end,
            speaker=speaker,
            relative_distance=relative_distance,
            energy_db=energy_db,
            high_freq_ratio=high_freq_ratio,
            estimated_snr=estimated_snr,
            reverb_ratio=reverb_ratio
        )

    def _estimate_reverb_ratio(self, segment: np.ndarray, sr: int) -> float:
        """
        Estimate reverberant energy ratio using energy decay.

        Higher ratio indicates more reverb (speaker further from mic).
        """
        # Compute envelope
        envelope = np.abs(signal.hilbert(segment))

        # Smooth envelope
        smooth_samples = int(0.05 * sr)
        envelope_smooth = uniform_filter1d(envelope, smooth_samples)

        # Split into early (first 50ms) and late (rest)
        early_samples = int(0.05 * sr)
        if len(envelope_smooth) <= early_samples * 2:
            return 0.3  # Default for short segments

        early_energy = np.mean(envelope_smooth[:early_samples]**2)
        late_energy = np.mean(envelope_smooth[early_samples:]**2)

        # Ratio of late to early energy
        reverb_ratio = late_energy / (early_energy + 1e-10)

        # Clamp to reasonable range
        return min(1.0, max(0.0, reverb_ratio))

    def _calculate_distance_score(self, energy_db: float,
                                   high_freq_ratio: float,
                                   snr: float,
                                   reverb_ratio: float) -> float:
        """
        Calculate combined distance score from individual metrics.

        Returns: 0.0 = very close, 1.0 = very far
        """
        # Normalize each metric to 0-1 range
        # Energy: -60dB (far) to -10dB (close)
        energy_score = np.clip((energy_db + 60) / 50, 0, 1)
        energy_score = 1 - energy_score  # Invert: low energy = far

        # High frequency: 0.01 (far) to 0.15 (close)
        hf_score = np.clip((high_freq_ratio - 0.01) / 0.14, 0, 1)
        hf_score = 1 - hf_score  # Invert: low HF = far

        # SNR: 5dB (far) to 40dB (close)
        snr_score = np.clip((snr - 5) / 35, 0, 1)
        snr_score = 1 - snr_score  # Invert: low SNR = far

        # Reverb: 0.1 (close) to 0.8 (far)
        reverb_score = np.clip((reverb_ratio - 0.1) / 0.7, 0, 1)
        # No invert: high reverb = far

        # Weighted combination
        weights = {
            'energy': 0.35,
            'high_freq': 0.25,
            'snr': 0.20,
            'reverb': 0.20
        }

        distance = (
            weights['energy'] * energy_score +
            weights['high_freq'] * hf_score +
            weights['snr'] * snr_score +
            weights['reverb'] * reverb_score
        )

        return np.clip(distance, 0, 1)

    def estimate_from_diarization(self, audio: np.ndarray,
                                   diarization: Dict) -> List[DistanceEstimate]:
        """
        Estimate distance for all segments in diarization result.

        Args:
            audio: Full audio array
            diarization: Diarization dict with 'segments' list

        Returns:
            List of DistanceEstimates, one per segment
        """
        segments = diarization.get('segments', [])
        estimates = []

        for seg in segments:
            estimate = self.estimate_segment_distance(
                audio,
                start=seg['start'],
                end=seg['end'],
                speaker=seg.get('speaker', 'UNKNOWN')
            )
            estimates.append(estimate)

        return estimates


class DistanceRobustEnhancer:
    """
    Applies adaptive enhancement based on speaker distance.

    Features:
    - Automatic gain control per segment
    - Frequency correction (boost highs for far speakers)
    - Consistent output loudness across all speakers
    """

    def __init__(self,
                 target_loudness_db: float = -20.0,
                 max_gain_db: float = 20.0,
                 eq_strength: float = 0.5,
                 crossfade_ms: float = 50.0,
                 verbose: bool = True):
        """
        Initialize distance-robust enhancer.

        Args:
            target_loudness_db: Target RMS level in dB
            max_gain_db: Maximum gain adjustment in dB
            eq_strength: Strength of frequency correction (0-1)
            crossfade_ms: Crossfade duration at segment boundaries
            verbose: Print progress
        """
        self.target_loudness_db = target_loudness_db
        self.max_gain_db = max_gain_db
        self.eq_strength = eq_strength
        self.crossfade_ms = crossfade_ms
        self.verbose = verbose

        self.estimator = DistanceEstimator(verbose=verbose)

    def enhance(self, input_path: Path,
                output_path: Path,
                diarization_json: Optional[Path] = None) -> EnhancementResult:
        """
        Apply distance-robust enhancement to audio file.

        Args:
            input_path: Input audio file
            output_path: Output audio file
            diarization_json: Optional diarization file (if None, analyzes whole file)

        Returns:
            EnhancementResult with processing statistics
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if self.verbose:
            print(f"\n🎚️ Distance-Robust Enhancement")
            print(f"   Input: {input_path.name}")

        # Load audio
        audio, sr = load_mono_audio(input_path, verbose=self.verbose)

        self.estimator.sample_rate = sr

        # Calculate original loudness
        original_rms = np.sqrt(np.mean(audio**2))
        original_loudness_db = 20 * np.log10(original_rms + 1e-10)

        # Get segments (from diarization or create single segment)
        if diarization_json and Path(diarization_json).exists():
            with open(diarization_json, 'r') as f:
                diarization = json.load(f)
            estimates = self.estimator.estimate_from_diarization(audio, diarization)
        else:
            # Create segments from audio (split into 5-second chunks)
            duration = len(audio) / sr
            chunk_duration = 5.0
            estimates = []

            for start in np.arange(0, duration, chunk_duration):
                end = min(start + chunk_duration, duration)
                estimate = self.estimator.estimate_segment_distance(
                    audio, start, end, "SPEAKER_00"
                )
                estimates.append(estimate)

        if not estimates:
            # No segments, just copy the file
            save_audio(audio, output_path, sr)
            return EnhancementResult(
                output_path=output_path,
                segments_processed=0,
                average_distance=0.5,
                gain_adjustments=[],
                original_loudness_db=original_loudness_db,
                output_loudness_db=original_loudness_db
            )

        if self.verbose:
            print(f"   Analyzing {len(estimates)} segments...")

        # Calculate gain adjustments for each segment
        gain_adjustments = self._calculate_gain_adjustments(estimates)

        # Apply per-segment enhancement
        enhanced_audio = self._apply_segment_enhancement(
            audio, sr, estimates, gain_adjustments
        )

        # Final loudness normalization
        enhanced_audio = self._normalize_final_loudness(enhanced_audio)

        # Calculate output loudness
        output_rms = np.sqrt(np.mean(enhanced_audio**2))
        output_loudness_db = 20 * np.log10(output_rms + 1e-10)

        # Save output
        save_audio(enhanced_audio, output_path, sr)

        average_distance = np.mean([e.relative_distance for e in estimates])

        if self.verbose:
            print(f"   Average distance: {average_distance:.2f} (0=close, 1=far)")
            print(f"   Gain range: {min(gain_adjustments):.1f} to {max(gain_adjustments):.1f} dB")
            print(f"   Loudness: {original_loudness_db:.1f} → {output_loudness_db:.1f} dB")
            print(f"   Saved: {output_path.name}")

        return EnhancementResult(
            output_path=output_path,
            segments_processed=len(estimates),
            average_distance=average_distance,
            gain_adjustments=gain_adjustments,
            original_loudness_db=original_loudness_db,
            output_loudness_db=output_loudness_db
        )

    def _calculate_gain_adjustments(self,
                                     estimates: List[DistanceEstimate]) -> List[float]:
        """
        Calculate gain adjustment for each segment based on distance.

        Far segments get more gain, close segments get less.
        """
        # Find the median energy as reference
        energies = [e.energy_db for e in estimates]
        median_energy = np.median(energies)

        gain_adjustments = []
        for estimate in estimates:
            # Base gain: bring to target loudness
            base_gain = self.target_loudness_db - estimate.energy_db

            # Distance-based adjustment
            # Far speakers (distance ~1) get extra boost
            # Close speakers (distance ~0) get less
            distance_adjustment = estimate.relative_distance * 6.0  # Up to +6dB for far speakers

            # Combine
            total_gain = base_gain + distance_adjustment

            # Clamp to max
            total_gain = np.clip(total_gain, -self.max_gain_db, self.max_gain_db)

            gain_adjustments.append(total_gain)

        return gain_adjustments

    def _apply_segment_enhancement(self, audio: np.ndarray, sr: int,
                                    estimates: List[DistanceEstimate],
                                    gain_adjustments: List[float]) -> np.ndarray:
        """
        Apply enhancement to each segment with crossfades.
        """
        enhanced = audio.copy()
        crossfade_samples = int(self.crossfade_ms / 1000 * sr)

        for estimate, gain_db in zip(estimates, gain_adjustments):
            start_sample = int(estimate.segment_start * sr)
            end_sample = int(estimate.segment_end * sr)

            # Bounds check
            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)

            if end_sample <= start_sample:
                continue

            # Extract segment
            segment = audio[start_sample:end_sample].copy()

            # Apply gain
            gain_linear = 10 ** (gain_db / 20)
            segment = segment * gain_linear

            # Apply frequency correction for far speakers
            if estimate.relative_distance > 0.4 and self.eq_strength > 0:
                segment = self._apply_high_frequency_boost(
                    segment, sr, estimate.relative_distance
                )

            # Apply with crossfade
            if crossfade_samples > 0 and len(segment) > 2 * crossfade_samples:
                # Fade in
                fade_in = np.linspace(0, 1, crossfade_samples)
                segment[:crossfade_samples] *= fade_in

                # Fade out
                fade_out = np.linspace(1, 0, crossfade_samples)
                segment[-crossfade_samples:] *= fade_out

                # Blend with existing (keep existing at fades)
                enhanced[start_sample:start_sample + crossfade_samples] *= (1 - fade_in)
                enhanced[end_sample - crossfade_samples:end_sample] *= (1 - fade_out)

            # Apply segment
            enhanced[start_sample:end_sample] = segment

        return enhanced

    def _apply_high_frequency_boost(self, segment: np.ndarray, sr: int,
                                     distance: float) -> np.ndarray:
        """
        Apply high-frequency boost to compensate for air absorption.

        Far speakers lose high frequencies, so we boost them back.
        """
        # Boost amount scales with distance and EQ strength
        boost_db = distance * self.eq_strength * 6.0  # Up to +6dB at 10kHz

        if boost_db < 0.5:
            return segment  # Not worth processing

        # Design a gentle high-shelf filter
        # Boost frequencies above 3kHz
        try:
            # High-shelf filter using biquad
            fc = 3000  # Cutoff frequency
            Q = 0.707  # Butterworth Q

            # Normalized frequency
            w0 = 2 * np.pi * fc / sr
            A = 10 ** (boost_db / 40)  # Amplitude

            # High-shelf coefficients
            alpha = np.sin(w0) / (2 * Q)
            cos_w0 = np.cos(w0)

            b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
            b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
            b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
            a0 = (A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
            a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
            a2 = (A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha

            # Normalize
            b = np.array([b0/a0, b1/a0, b2/a0])
            a = np.array([1.0, a1/a0, a2/a0])

            # Apply filter
            filtered = signal.lfilter(b, a, segment)

            return filtered.astype(np.float32)

        except Exception:
            return segment  # Return original on error

    def _normalize_final_loudness(self, audio: np.ndarray) -> np.ndarray:
        """
        Final loudness normalization to target level.
        """
        rms = np.sqrt(np.mean(audio**2))
        current_db = 20 * np.log10(rms + 1e-10)

        gain_db = self.target_loudness_db - current_db
        gain_db = np.clip(gain_db, -20, 20)  # Limit adjustment

        gain_linear = 10 ** (gain_db / 20)
        normalized = audio * gain_linear

        normalized = prevent_clipping(normalized)
        return normalized


def quick_distance_enhance(input_path: Path,
                           output_path: Optional[Path] = None,
                           diarization_json: Optional[Path] = None,
                           verbose: bool = True) -> EnhancementResult:
    """
    Quick one-liner for distance-robust enhancement.

    Args:
        input_path: Input audio file
        output_path: Output file (default: input_distance_enhanced.wav)
        diarization_json: Optional diarization results
        verbose: Print progress

    Returns:
        EnhancementResult
    """
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_distance_enhanced.wav"

    enhancer = DistanceRobustEnhancer(verbose=verbose)
    return enhancer.enhance(input_path, output_path, diarization_json)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Distance-robust speech enhancement"
    )
    parser.add_argument("input", type=Path, help="Input audio file")
    parser.add_argument("-o", "--output", type=Path, help="Output audio file")
    parser.add_argument("-d", "--diarization", type=Path,
                        help="Diarization JSON file")
    parser.add_argument("--target-db", type=float, default=-20.0,
                        help="Target loudness in dB (default: -20)")
    parser.add_argument("--eq-strength", type=float, default=0.5,
                        help="EQ correction strength 0-1 (default: 0.5)")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")

    args = parser.parse_args()

    output = args.output or args.input.parent / f"{args.input.stem}_distance_enhanced.wav"

    enhancer = DistanceRobustEnhancer(
        target_loudness_db=args.target_db,
        eq_strength=args.eq_strength,
        verbose=not args.quiet
    )

    result = enhancer.enhance(args.input, output, args.diarization)

    print(f"\n✅ Enhancement complete!")
    print(f"   Segments: {result.segments_processed}")
    print(f"   Avg distance: {result.average_distance:.2f}")
    print(f"   Output: {result.output_path}")
