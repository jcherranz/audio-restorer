#!/usr/bin/env python3
"""
Audio Quality Measurement Tool (Quick Analysis)
===============================================

Analyzes audio files and reports quality metrics.
This tool provides FAST, SIMPLIFIED metrics for quick iteration.

For ACADEMIC-GRADE METRICS (DNSMOS, PESQ, STOI, SI-SDR), use:
    python tests/sota_benchmark.py <audio_file>

Usage:
    python measure_quality.py <audio_file>
    python measure_quality.py --compare <before.wav> <after.wav>

Metrics measured (simplified):
    - SNR (Signal-to-Noise Ratio) - estimated from energy percentiles
    - Background noise level - bottom 10% frame energy
    - Dynamic range - peak to RMS ratio
    - Loudness (LUFS) - simplified K-weighting estimate (not ITU-R BS.1770)
    - Clarity score - custom heuristic (HF ratio + noise factor)
    - Overall quality score - weighted combination

NOTE: For publication-quality metrics or academic comparison, use sota_benchmark.py
which implements proper DNSMOS (Microsoft), PESQ (ITU-T P.862), and STOI.
"""

import sys
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class QualityMetrics:
    """Container for audio quality metrics"""
    snr_db: float
    noise_level_db: float
    dynamic_range_db: float
    peak_db: float
    rms_db: float
    loudness_lufs: float  # Estimated
    clarity_score: float
    quality_score: float
    duration_s: float
    
    def to_dict(self) -> Dict:
        return {
            'snr_db': self.snr_db,
            'noise_level_db': self.noise_level_db,
            'dynamic_range_db': self.dynamic_range_db,
            'peak_db': self.peak_db,
            'rms_db': self.rms_db,
            'loudness_lufs': self.loudness_lufs,
            'clarity_score': self.clarity_score,
            'quality_score': self.quality_score,
            'duration_s': self.duration_s,
        }


def estimate_noise_level(audio: np.ndarray, sr: int) -> Tuple[float, np.ndarray]:
    """
    Estimate background noise level using voice activity detection.
    
    Returns:
        noise_level_db: Estimated noise floor in dB
        noise_mask: Boolean mask of noise segments
    """
    # Calculate RMS in short windows
    frame_length = int(0.025 * sr)  # 25ms windows
    hop_length = int(0.010 * sr)    # 10ms hop
    
    # Compute RMS energy
    rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]
    
    # Find quiet segments (likely noise)
    # Use bottom 10% of frames as noise estimate
    sorted_rms = np.sort(rms)
    noise_threshold_idx = max(1, int(len(rms) * 0.1))
    noise_rms = np.mean(sorted_rms[:noise_threshold_idx])
    
    # Create noise mask
    noise_threshold = np.percentile(rms, 20)  # Bottom 20% is noise
    noise_mask = rms < noise_threshold
    
    # Convert to dB
    noise_level_db = 20 * np.log10(noise_rms + 1e-10)
    
    return noise_level_db, noise_mask


def estimate_signal_level(audio: np.ndarray, sr: int, noise_mask: np.ndarray) -> float:
    """
    Estimate signal level using voiced segments.
    
    Returns:
        signal_level_db: Estimated signal level in dB
    """
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)
    
    # Compute RMS
    rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]
    
    # Use high-energy frames as signal
    signal_threshold = np.percentile(rms, 80)  # Top 20% is signal
    signal_frames = rms[rms > signal_threshold]
    
    if len(signal_frames) == 0:
        signal_rms = np.mean(rms)
    else:
        signal_rms = np.mean(signal_frames)
    
    signal_level_db = 20 * np.log10(signal_rms + 1e-10)
    
    return signal_level_db


def calculate_snr(audio: np.ndarray, sr: int) -> float:
    """Calculate Signal-to-Noise Ratio in dB"""
    noise_level_db, noise_mask = estimate_noise_level(audio, sr)
    signal_level_db = estimate_signal_level(audio, sr, noise_mask)
    
    snr = signal_level_db - noise_level_db
    return snr, noise_level_db, signal_level_db


def calculate_dynamic_range(audio: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculate dynamic range metrics.
    
    Returns:
        peak_db: Peak level in dB
        rms_db: RMS level in dB
        dynamic_range_db: Difference between peak and RMS
    """
    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio**2))
    
    peak_db = 20 * np.log10(peak + 1e-10)
    rms_db = 20 * np.log10(rms + 1e-10)
    dynamic_range = peak_db - rms_db
    
    return peak_db, rms_db, dynamic_range


def estimate_loudness(audio: np.ndarray, sr: int) -> float:
    """
    Estimate loudness in LUFS (simplified version).
    
    Note: This is a simplified estimate. For accurate LUFS, use pyloudnorm.
    """
    # Simple estimate based on RMS with speech weighting
    # LUFS is approximately -0.691 + 10*log10(mean(z^2))
    # where z is the signal filtered with K-weighting
    
    # Simplified: assume K-weighting roughly equals high-pass at 100Hz
    from scipy import signal
    sos = signal.butter(4, 100, 'hp', fs=sr, output='sos')
    filtered = signal.sosfilt(sos, audio)
    
    # Calculate gated loudness (simplified)
    frame_length = int(0.4 * sr)  # 400ms windows
    hop_length = int(0.1 * sr)    # 100ms hop
    
    frames = librosa.util.frame(filtered, frame_length=frame_length, hop_length=hop_length)
    
    # Calculate mean square for each frame
    ms = np.mean(frames**2, axis=0)
    
    # Relative gate at -70 LUFS equivalent (very quiet threshold)
    threshold = 10 ** ((-70 + 0.691) / 10)
    ms_above_threshold = ms[ms > threshold]
    
    if len(ms_above_threshold) == 0:
        return -70.0
    
    # Mean of frames above threshold
    loudness = -0.691 + 10 * np.log10(np.mean(ms_above_threshold))
    
    return loudness


def calculate_clarity(audio: np.ndarray, sr: int, noise_level_db: float) -> float:
    """
    Calculate clarity score (0-1).
    
    Based on:
    - High-frequency content (4-8 kHz for consonants)
    - Low noise floor
    - Dynamic content
    """
    # Spectral analysis
    stft = np.abs(librosa.stft(audio))
    freqs = librosa.fft_frequencies(sr=sr)
    
    # Calculate energy in different bands
    # Speech clarity heavily depends on 2-8 kHz range
    clarity_band = (freqs >= 2000) & (freqs <= 8000)
    total_energy = np.sum(stft)
    clarity_energy = np.sum(stft[clarity_band, :])
    
    # High-frequency ratio
    hf_ratio = clarity_energy / (total_energy + 1e-10)
    
    # Noise factor (how much noise affects clarity)
    noise_linear = 10 ** (noise_level_db / 20)
    signal_linear = np.sqrt(np.mean(audio**2))
    noise_factor = 1.0 - (noise_linear / (signal_linear + 1e-10))
    noise_factor = np.clip(noise_factor, 0, 1)
    
    # Dynamic factor (some dynamic range is good for clarity)
    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio**2))
    crest_factor = peak / (rms + 1e-10)
    dynamic_factor = np.clip((crest_factor - 1) / 9, 0, 1)  # Normalize 1-10 to 0-1
    
    # Combined clarity score
    clarity = (hf_ratio * 0.4 + noise_factor * 0.4 + dynamic_factor * 0.2)
    clarity = np.clip(clarity, 0, 1)
    
    return clarity


def calculate_overall_score(metrics: QualityMetrics) -> float:
    """
    Calculate overall quality score (0-100).
    
    Weights:
    - SNR: 30%
    - Clarity: 25%
    - Noise level: 20%
    - Dynamic range: 15%
    - Loudness: 10%
    """
    score = 0.0
    
    # SNR (30% weight) - target 30 dB
    snr_score = min(metrics.snr_db / 30, 1.0) * 30
    score += snr_score
    
    # Clarity (25% weight)
    clarity_score = metrics.clarity_score * 25
    score += clarity_score
    
    # Noise level (20% weight) - target -50 dB, bad at -30 dB
    noise_range = -30 - (-50)  # 20 dB range
    noise_score = min(max(0, (-30 - metrics.noise_level_db) / noise_range), 1.0) * 20
    score += noise_score
    
    # Dynamic range (15% weight) - optimal 20 dB
    dr = metrics.dynamic_range_db
    if 15 <= dr <= 25:
        dr_score = 15
    else:
        dr_deviation = abs(dr - 20)
        dr_score = max(0, 15 * (1 - dr_deviation / 20))
    score += dr_score
    
    # Loudness (10% weight) - target -16 LUFS
    loudness_diff = abs(metrics.loudness_lufs - (-16))
    loudness_score = max(0, 10 - loudness_diff / 2)
    score += loudness_score
    
    return score


def measure_all(audio_path: Path) -> QualityMetrics:
    """
    Measure all quality metrics for an audio file.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        QualityMetrics object with all measurements
    """
    print(f"Analyzing: {audio_path}")

    # Load audio at native sample rate (preserves full bandwidth)
    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = len(audio) / sr

    print(f"  Duration: {duration:.1f}s ({duration/60:.1f} min)")
    print(f"  Sample rate: {sr} Hz")
    
    # Calculate metrics
    print("  Calculating SNR...")
    snr, noise_level, signal_level = calculate_snr(audio, sr)
    
    print("  Calculating dynamic range...")
    peak_db, rms_db, dynamic_range = calculate_dynamic_range(audio)
    
    print("  Estimating loudness...")
    loudness = estimate_loudness(audio, sr)
    
    print("  Calculating clarity...")
    clarity = calculate_clarity(audio, sr, noise_level)
    
    # Create metrics object
    metrics = QualityMetrics(
        snr_db=snr,
        noise_level_db=noise_level,
        dynamic_range_db=dynamic_range,
        peak_db=peak_db,
        rms_db=rms_db,
        loudness_lufs=loudness,
        clarity_score=clarity,
        quality_score=0.0,  # Will calculate after
        duration_s=duration
    )
    
    # Calculate overall score
    metrics.quality_score = calculate_overall_score(metrics)
    
    return metrics


def print_report(metrics: QualityMetrics, title: str = "AUDIO QUALITY REPORT"):
    """Print formatted quality report"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print(f"  Signal-to-Noise Ratio:  {metrics.snr_db:6.1f} dB   (target: >25)")
    print(f"  Noise Level:            {metrics.noise_level_db:6.1f} dB   (target: <-50)")
    print(f"  Dynamic Range:          {metrics.dynamic_range_db:6.1f} dB   (target: 15-25)")
    print(f"  Peak Level:             {metrics.peak_db:6.1f} dB")
    print(f"  RMS Level:              {metrics.rms_db:6.1f} dB")
    print(f"  Loudness:               {metrics.loudness_lufs:6.1f} LUFS (target: -16)")
    print(f"  Clarity Score:          {metrics.clarity_score:6.2f} / 1.0 (target: >0.7)")
    print("-" * 60)
    print(f"  OVERALL QUALITY SCORE:  {metrics.quality_score:5.1f} / 100")
    print("=" * 60)
    
    # Quality assessment
    if metrics.quality_score >= 90:
        quality = "EXCELLENT"
    elif metrics.quality_score >= 80:
        quality = "VERY GOOD"
    elif metrics.quality_score >= 70:
        quality = "GOOD"
    elif metrics.quality_score >= 60:
        quality = "ACCEPTABLE"
    else:
        quality = "NEEDS IMPROVEMENT"
    
    print(f"  Assessment: {quality}")
    print("=" * 60)


def compare_files(before_path: Path, after_path: Path):
    """Compare two audio files and show improvement"""
    print("\n" + "=" * 60)
    print("  BEFORE/AFTER COMPARISON")
    print("=" * 60)
    
    print(f"\nBEFORE: {before_path}")
    before_metrics = measure_all(before_path)
    
    print(f"\nAFTER: {after_path}")
    after_metrics = measure_all(after_path)
    
    # Print comparison table
    print("\n" + "=" * 60)
    print("  COMPARISON TABLE")
    print("=" * 60)
    print(f"  {'Metric':<25} {'Before':>10} {'After':>10} {'Change':>10}")
    print("-" * 60)
    
    metrics_to_compare = [
        ('SNR (dB)', before_metrics.snr_db, after_metrics.snr_db),
        ('Noise Level (dB)', before_metrics.noise_level_db, after_metrics.noise_level_db),
        ('Dynamic Range (dB)', before_metrics.dynamic_range_db, after_metrics.dynamic_range_db),
        ('Loudness (LUFS)', before_metrics.loudness_lufs, after_metrics.loudness_lufs),
        ('Clarity Score', before_metrics.clarity_score, after_metrics.clarity_score),
        ('Quality Score', before_metrics.quality_score, after_metrics.quality_score),
    ]
    
    for name, before, after in metrics_to_compare:
        change = after - before
        change_str = f"+{change:.1f}" if change >= 0 else f"{change:.1f}"
        print(f"  {name:<25} {before:>10.1f} {after:>10.1f} {change_str:>10}")
    
    print("=" * 60)
    
    # Overall improvement
    score_improvement = after_metrics.quality_score - before_metrics.quality_score
    print(f"\n  Overall Quality Improvement: {score_improvement:+.1f} points")
    
    if score_improvement > 0:
        print("  ✅ Processing improved audio quality")
    elif score_improvement < 0:
        print("  ⚠️  Processing degraded audio quality")
    else:
        print("  ➡️  No significant change")
    
    print("=" * 60)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Measure audio quality metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python measure_quality.py audio.wav
  python measure_quality.py --compare original.wav enhanced.wav
        """
    )
    
    parser.add_argument("file", nargs="?", help="Audio file to analyze")
    parser.add_argument("--compare", action="store_true", 
                       help="Compare two files (before/after)")
    parser.add_argument("before", nargs="?", help="Before processing file")
    parser.add_argument("after", nargs="?", help="After processing file")
    
    args = parser.parse_args()
    
    if args.compare:
        if not args.before or not args.after:
            print("Error: --compare requires two files: before and after")
            print("Usage: python measure_quality.py --compare before.wav after.wav")
            sys.exit(1)
        compare_files(Path(args.before), Path(args.after))
    elif args.file:
        metrics = measure_all(Path(args.file))
        print_report(metrics)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
