"""
Per-Speaker Automatic Gain Control (AGC) Module
Ensures consistent volume across different speakers in multi-speaker recordings.

This module uses diarization results to:
1. Calculate average loudness per speaker
2. Normalize each speaker to a consistent target level
3. Apply smooth gain transitions at speaker boundaries

Different from distance-robust enhancement:
- Distance-robust: adjusts gain based on estimated distance (per segment)
- Per-speaker AGC: ensures each speaker has consistent loudness across ALL their segments
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import soundfile as sf
from scipy import signal
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")


@dataclass
class SpeakerLoudness:
    """Loudness statistics for a single speaker."""
    speaker_id: str
    mean_rms: float
    mean_db: float
    peak_db: float
    total_duration: float
    num_segments: int
    gain_adjustment_db: float = 0.0


@dataclass
class AGCResult:
    """Result of automatic gain control processing."""
    output_path: Path
    num_speakers: int
    speaker_stats: List[SpeakerLoudness]
    target_loudness_db: float
    max_gain_applied_db: float
    min_gain_applied_db: float


class SpeakerAGC:
    """
    Automatic Gain Control that normalizes each speaker to a consistent level.

    Unlike simple normalization (which affects the whole file equally) or
    distance-robust enhancement (which adjusts per-segment based on distance),
    this module:

    1. Identifies all speakers using diarization
    2. Calculates each speaker's average loudness
    3. Determines gain adjustment needed for each speaker
    4. Applies speaker-specific gain with smooth transitions

    This ensures that in a multi-speaker recording:
    - A quiet speaker becomes as loud as others
    - A loud speaker is brought down to match
    - Each speaker sounds consistent throughout their segments
    """

    def __init__(self,
                 target_loudness_db: float = -20.0,
                 max_gain_db: float = 15.0,
                 min_gain_db: float = -10.0,
                 crossfade_ms: float = 30.0,
                 verbose: bool = True):
        """
        Initialize per-speaker AGC.

        Args:
            target_loudness_db: Target RMS level in dB for all speakers
            max_gain_db: Maximum gain boost allowed (prevents noise amplification)
            min_gain_db: Maximum gain reduction allowed (prevents over-attenuation)
            crossfade_ms: Crossfade duration at speaker boundaries
            verbose: Print progress messages
        """
        self.target_loudness_db = target_loudness_db
        self.max_gain_db = max_gain_db
        self.min_gain_db = min_gain_db
        self.crossfade_ms = crossfade_ms
        self.verbose = verbose

    def analyze_speaker_loudness(self, audio: np.ndarray, sr: int,
                                  diarization: Dict) -> List[SpeakerLoudness]:
        """
        Analyze loudness statistics for each speaker.

        Args:
            audio: Audio samples
            sr: Sample rate
            diarization: Diarization dict with 'segments' list

        Returns:
            List of SpeakerLoudness objects, one per speaker
        """
        segments = diarization.get('segments', [])

        if not segments:
            # No diarization, treat as single speaker
            rms = np.sqrt(np.mean(audio**2))
            mean_db = 20 * np.log10(rms + 1e-10)
            peak_db = 20 * np.log10(np.max(np.abs(audio)) + 1e-10)

            return [SpeakerLoudness(
                speaker_id="SPEAKER_00",
                mean_rms=rms,
                mean_db=mean_db,
                peak_db=peak_db,
                total_duration=len(audio) / sr,
                num_segments=1,
                gain_adjustment_db=self.target_loudness_db - mean_db
            )]

        # Group segments by speaker
        speaker_segments: Dict[str, List[Dict]] = {}
        for seg in segments:
            speaker = seg.get('speaker', 'UNKNOWN')
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append(seg)

        # Calculate loudness per speaker
        speaker_stats = []
        for speaker_id, segs in speaker_segments.items():
            rms_values = []
            peaks = []
            total_duration = 0.0

            for seg in segs:
                start_sample = int(seg['start'] * sr)
                end_sample = int(seg['end'] * sr)

                # Bounds check
                start_sample = max(0, start_sample)
                end_sample = min(len(audio), end_sample)

                if end_sample <= start_sample:
                    continue

                segment_audio = audio[start_sample:end_sample]

                # Calculate RMS for this segment
                seg_rms = np.sqrt(np.mean(segment_audio**2))
                rms_values.append(seg_rms)

                # Track peak
                seg_peak = np.max(np.abs(segment_audio))
                peaks.append(seg_peak)

                # Track duration
                total_duration += (end_sample - start_sample) / sr

            if not rms_values:
                continue

            # Calculate mean loudness (weighted by segment duration would be better,
            # but simple mean is usually good enough)
            mean_rms = np.mean(rms_values)
            mean_db = 20 * np.log10(mean_rms + 1e-10)
            peak_db = 20 * np.log10(max(peaks) + 1e-10)

            # Calculate required gain adjustment
            gain_db = self.target_loudness_db - mean_db
            gain_db = np.clip(gain_db, self.min_gain_db, self.max_gain_db)

            speaker_stats.append(SpeakerLoudness(
                speaker_id=speaker_id,
                mean_rms=mean_rms,
                mean_db=mean_db,
                peak_db=peak_db,
                total_duration=total_duration,
                num_segments=len(segs),
                gain_adjustment_db=gain_db
            ))

        # Sort by total duration (main speaker first)
        speaker_stats.sort(key=lambda x: x.total_duration, reverse=True)

        return speaker_stats

    def apply_agc(self, audio: np.ndarray, sr: int,
                  diarization: Dict,
                  speaker_stats: List[SpeakerLoudness]) -> np.ndarray:
        """
        Apply per-speaker gain adjustments.

        Args:
            audio: Audio samples
            sr: Sample rate
            diarization: Diarization dict
            speaker_stats: Pre-calculated speaker loudness stats

        Returns:
            Gain-adjusted audio
        """
        segments = diarization.get('segments', [])

        if not segments or not speaker_stats:
            # No diarization, apply simple normalization
            if speaker_stats:
                gain_db = speaker_stats[0].gain_adjustment_db
                gain_linear = 10 ** (gain_db / 20)
                return audio * gain_linear
            return audio

        # Create lookup for speaker gains
        speaker_gains = {s.speaker_id: s.gain_adjustment_db for s in speaker_stats}

        # Create output array
        output = audio.copy()

        # Create a gain envelope for smooth transitions
        crossfade_samples = int(self.crossfade_ms / 1000 * sr)

        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda x: x['start'])

        for seg in sorted_segments:
            speaker = seg.get('speaker', 'UNKNOWN')
            gain_db = speaker_gains.get(speaker, 0.0)
            gain_linear = 10 ** (gain_db / 20)

            start_sample = int(seg['start'] * sr)
            end_sample = int(seg['end'] * sr)

            # Bounds check
            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)

            if end_sample <= start_sample:
                continue

            segment_length = end_sample - start_sample

            # Apply gain with crossfade at boundaries
            if segment_length > 2 * crossfade_samples:
                # Fade in
                fade_in = np.linspace(1.0, gain_linear, crossfade_samples)
                output[start_sample:start_sample + crossfade_samples] *= fade_in

                # Middle section (full gain)
                output[start_sample + crossfade_samples:end_sample - crossfade_samples] *= gain_linear

                # Fade out
                fade_out = np.linspace(gain_linear, 1.0, crossfade_samples)
                output[end_sample - crossfade_samples:end_sample] *= fade_out
            else:
                # Short segment: apply uniform gain
                output[start_sample:end_sample] *= gain_linear

        # Prevent clipping
        peak = np.max(np.abs(output))
        if peak > 0.95:
            output = output * (0.95 / peak)

        return output

    def process(self, input_path: Path,
                output_path: Path,
                diarization_json: Optional[Path] = None) -> AGCResult:
        """
        Apply per-speaker AGC to an audio file.

        Args:
            input_path: Input audio file
            output_path: Output audio file
            diarization_json: Path to diarization JSON (if None, runs diarization)

        Returns:
            AGCResult with processing statistics
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if self.verbose:
            print(f"\n🎚️ Per-Speaker Automatic Gain Control")
            print(f"   Input: {input_path.name}")

        # Load audio
        audio, sr = sf.read(str(input_path), dtype='float32')
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        duration = len(audio) / sr

        if self.verbose:
            print(f"   Duration: {timedelta(seconds=int(duration))}")

        # Load or run diarization
        if diarization_json and Path(diarization_json).exists():
            with open(diarization_json, 'r') as f:
                diarization = json.load(f)
            if self.verbose:
                print(f"   Using existing diarization: {diarization_json.name}")
        else:
            # Run diarization
            if self.verbose:
                print("   Running speaker diarization...")
            from .diarization import SpeakerDiarizer
            diarizer = SpeakerDiarizer(verbose=False)
            result = diarizer.diarize(input_path)
            diarization = result.to_dict()

        # Analyze speaker loudness
        if self.verbose:
            print("   Analyzing speaker loudness...")
        speaker_stats = self.analyze_speaker_loudness(audio, sr, diarization)

        if self.verbose:
            print(f"   Found {len(speaker_stats)} speaker(s):")
            for stat in speaker_stats:
                print(f"      {stat.speaker_id}: {stat.mean_db:.1f} dB → "
                      f"gain {stat.gain_adjustment_db:+.1f} dB "
                      f"({stat.num_segments} segments, {stat.total_duration:.1f}s)")

        # Apply AGC
        if self.verbose:
            print("   Applying per-speaker gain control...")
        output_audio = self.apply_agc(audio, sr, diarization, speaker_stats)

        # Save output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), output_audio, sr)

        # Calculate result statistics
        gains = [s.gain_adjustment_db for s in speaker_stats]
        max_gain = max(gains) if gains else 0.0
        min_gain = min(gains) if gains else 0.0

        if self.verbose:
            print(f"   Gain range: {min_gain:+.1f} to {max_gain:+.1f} dB")
            print(f"   Target loudness: {self.target_loudness_db:.1f} dB")
            print(f"   Saved: {output_path.name}")

        return AGCResult(
            output_path=output_path,
            num_speakers=len(speaker_stats),
            speaker_stats=speaker_stats,
            target_loudness_db=self.target_loudness_db,
            max_gain_applied_db=max_gain,
            min_gain_applied_db=min_gain
        )


def quick_agc(input_path: Path,
              output_path: Optional[Path] = None,
              diarization_json: Optional[Path] = None,
              verbose: bool = True) -> AGCResult:
    """
    Quick one-liner for per-speaker AGC.

    Args:
        input_path: Input audio file
        output_path: Output file (default: input_agc.wav)
        diarization_json: Optional diarization results
        verbose: Print progress

    Returns:
        AGCResult
    """
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_agc.wav"

    agc = SpeakerAGC(verbose=verbose)
    return agc.process(input_path, output_path, diarization_json)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Per-speaker automatic gain control"
    )
    parser.add_argument("input", type=Path, help="Input audio file")
    parser.add_argument("-o", "--output", type=Path, help="Output audio file")
    parser.add_argument("-d", "--diarization", type=Path,
                        help="Diarization JSON file")
    parser.add_argument("--target-db", type=float, default=-20.0,
                        help="Target loudness in dB (default: -20)")
    parser.add_argument("--max-gain", type=float, default=15.0,
                        help="Maximum gain boost in dB (default: 15)")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")

    args = parser.parse_args()

    output = args.output or args.input.parent / f"{args.input.stem}_agc.wav"

    agc = SpeakerAGC(
        target_loudness_db=args.target_db,
        max_gain_db=args.max_gain,
        verbose=not args.quiet
    )

    result = agc.process(args.input, output, args.diarization)

    print(f"\n✅ AGC complete!")
    print(f"   Speakers: {result.num_speakers}")
    print(f"   Gain range: {result.min_gain_applied_db:+.1f} to {result.max_gain_applied_db:+.1f} dB")
    print(f"   Output: {result.output_path}")
