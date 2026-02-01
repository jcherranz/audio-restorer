#!/usr/bin/env python3
"""
Comprehensive Enhancer Benchmark Suite
======================================

Systematically compares all available enhancers against a reference audio file.
Generates quality metrics and comparison reports.

Usage:
    # Full benchmark (all enhancers)
    python tests/benchmark_enhancers.py

    # Specific enhancers only
    python tests/benchmark_enhancers.py --enhancers deepfilter torch_advanced

    # Use existing audio file instead of downloading
    python tests/benchmark_enhancers.py --input path/to/audio.wav

    # Quick mode (shorter audio sample)
    python tests/benchmark_enhancers.py --quick

Output:
    - benchmarks/benchmark_YYYYMMDD_HHMMSS.json (raw metrics)
    - benchmarks/benchmark_YYYYMMDD_HHMMSS.txt (human-readable report)
    - benchmarks/audio_<enhancer>_enhanced.wav (enhanced files)
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.measure_quality import measure_all, QualityMetrics


@dataclass
class EnhancerResult:
    """Results for a single enhancer run."""
    enhancer: str
    success: bool
    processing_time_s: float
    error_message: str
    metrics: Optional[Dict]
    sota_metrics: Optional[Dict]
    output_file: Optional[str]


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""
    timestamp: str
    reference_video: str
    input_file: str
    input_duration_s: float
    enhancers_tested: List[str]
    results: List[Dict]
    best_enhancer: str
    best_quality_score: float


# Default reference video (consistent across all tests)
REFERENCE_VIDEO = "https://youtu.be/cglDoG0GzyA"

# Available enhancers to test
AVAILABLE_ENHANCERS = ["simple", "torch", "torch_advanced", "deepfilter"]
# NOTE: 'resemble' enhancer removed due to dependency issues (PyTorch 2.1.1 required)


def download_reference_audio(output_dir: Path, video_url: str, verbose: bool = True) -> Path:
    """
    Download and extract audio from reference video.

    Uses cached version if available.
    """
    from src.downloader import YouTubeDownloader

    # Extract video ID from URL
    if "youtu.be" in video_url:
        video_id = video_url.split("/")[-1].split("?")[0]
    else:
        video_id = video_url.split("v=")[-1].split("&")[0]

    # Check for cached audio
    cached_audio = output_dir / f"reference_{video_id}.wav"
    if cached_audio.exists():
        if verbose:
            print(f"Using cached reference audio: {cached_audio}")
        return cached_audio

    if verbose:
        print(f"Downloading reference video: {video_url}")

    # Download
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    downloader = YouTubeDownloader(temp_dir, verbose=verbose)
    audio_path = downloader.download_audio_only(video_url)

    # Move to benchmark dir with consistent name
    import shutil
    shutil.move(str(audio_path), str(cached_audio))

    if verbose:
        print(f"Saved reference audio: {cached_audio}")

    return cached_audio


def run_enhancer(
    enhancer_type: str,
    input_path: Path,
    output_dir: Path,
    include_sota: bool = False,
    verbose: bool = True
) -> EnhancerResult:
    """
    Run a single enhancer and collect metrics.
    """
    output_path = output_dir / f"enhanced_{enhancer_type}.wav"

    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing: {enhancer_type}")
        print('='*60)

    start_time = time.time()

    try:
        # Import and create enhancer
        if enhancer_type == "simple":
            from src.audio_enhancer import SimpleEnhancer
            enhancer = SimpleEnhancer(verbose=verbose)
        elif enhancer_type == "torch":
            from src.ml_enhancer import TorchEnhancer
            enhancer = TorchEnhancer(verbose=verbose)
        elif enhancer_type == "torch_advanced":
            from src.ml_enhancer import AdvancedMLEnhancer
            enhancer = AdvancedMLEnhancer(verbose=verbose, use_vad=True)
        elif enhancer_type == "deepfilter":
            from src.deepfilter_enhancer import DeepFilterNetEnhancer
            enhancer = DeepFilterNetEnhancer(verbose=verbose)
        else:
            raise ValueError(f"Unknown enhancer: {enhancer_type}")

        # Run enhancement
        enhancer.enhance(input_path, output_path)

        processing_time = time.time() - start_time

        # Measure quality
        if verbose:
            print(f"\nMeasuring quality metrics...")
        metrics = measure_all(output_path)

        # Calculate SOTA metrics if requested
        sota_metrics = None
        if include_sota:
            try:
                if verbose:
                    print(f"Calculating SOTA metrics (DNSMOS)...")
                from src.sota_metrics import SOTAMetricsCalculator
                calculator = SOTAMetricsCalculator(verbose=False)
                sota_result = calculator.calculate(output_path)
                sota_metrics = sota_result.to_dict()
            except Exception as e:
                if verbose:
                    print(f"  Warning: SOTA metrics failed: {e}")
                sota_metrics = None

        return EnhancerResult(
            enhancer=enhancer_type,
            success=True,
            processing_time_s=round(processing_time, 2),
            error_message="",
            metrics=metrics.to_dict(),
            sota_metrics=sota_metrics,
            output_file=str(output_path)
        )

    except Exception as e:
        processing_time = time.time() - start_time
        if verbose:
            print(f"  ERROR: {e}")
        return EnhancerResult(
            enhancer=enhancer_type,
            success=False,
            processing_time_s=round(processing_time, 2),
            error_message=str(e),
            metrics=None,
            sota_metrics=None,
            output_file=None
        )


def run_benchmark(
    enhancers: List[str],
    input_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    video_url: str = REFERENCE_VIDEO,
    quick: bool = False,
    include_sota: bool = False,
    verbose: bool = True
) -> BenchmarkReport:
    """
    Run full benchmark suite.
    """
    # Setup directories
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get input audio
    if input_path is None:
        input_path = download_reference_audio(output_dir, video_url, verbose)

    # Get input duration
    import soundfile as sf
    audio, sr = sf.read(input_path)
    input_duration = len(audio) / sr

    if verbose:
        print(f"\nInput: {input_path.name}")
        print(f"Duration: {input_duration:.1f}s ({input_duration/60:.1f} min)")
        print(f"Enhancers to test: {', '.join(enhancers)}")
        if include_sota:
            print("SOTA metrics (DNSMOS): Enabled")

    # Run each enhancer
    results = []
    for enhancer in enhancers:
        result = run_enhancer(enhancer, input_path, output_dir, include_sota, verbose)
        results.append(result)

    # Find best enhancer
    successful_results = [r for r in results if r.success and r.metrics]
    if successful_results:
        best = max(successful_results, key=lambda r: r.metrics.get('quality_score', 0))
        best_enhancer = best.enhancer
        best_quality_score = best.metrics.get('quality_score', 0)
    else:
        best_enhancer = "none"
        best_quality_score = 0

    report = BenchmarkReport(
        timestamp=timestamp,
        reference_video=video_url,
        input_file=str(input_path),
        input_duration_s=round(input_duration, 2),
        enhancers_tested=enhancers,
        results=[asdict(r) for r in results],
        best_enhancer=best_enhancer,
        best_quality_score=round(best_quality_score, 2)
    )

    return report


def print_comparison_table(report: BenchmarkReport):
    """Print a comparison table of all enhancers."""
    # Check if any results have SOTA metrics
    has_sota = any(r.get('sota_metrics') for r in report.results)

    print(f"\n{'='*90}")
    print("BENCHMARK COMPARISON TABLE")
    print('='*90)

    # Header - adjust for SOTA metrics if present
    if has_sota:
        print(f"\n{'Enhancer':<15} {'Status':<8} {'Time':<8} {'Quality':<9} {'SNR':<9} {'DNSMOS':<8} {'Clarity':<8}")
        print("-" * 90)
    else:
        print(f"\n{'Enhancer':<15} {'Status':<10} {'Time':<10} {'Quality':<10} {'SNR':<10} {'Clarity':<10}")
        print("-" * 75)

    for result in report.results:
        enhancer = result['enhancer']
        status = "PASS" if result['success'] else "FAIL"
        time_s = f"{result['processing_time_s']:.1f}s"

        if result['success'] and result['metrics']:
            quality = f"{result['metrics']['quality_score']:.1f}"
            snr = f"{result['metrics']['snr_db']:.1f} dB"
            clarity = f"{result['metrics']['clarity_score']:.2f}"
        else:
            quality = "N/A"
            snr = "N/A"
            clarity = "N/A"

        # Get DNSMOS if available
        dnsmos = "N/A"
        if result.get('sota_metrics') and result['sota_metrics'].get('dnsmos_ovrl'):
            dnsmos = f"{result['sota_metrics']['dnsmos_ovrl']:.2f}"

        # Mark best enhancer
        marker = " *" if enhancer == report.best_enhancer else ""

        if has_sota:
            print(f"{enhancer:<15} {status:<8} {time_s:<8} {quality:<9} {snr:<9} {dnsmos:<8} {clarity:<8}{marker}")
        else:
            print(f"{enhancer:<15} {status:<10} {time_s:<10} {quality:<10} {snr:<10} {clarity:<10}{marker}")

    print("-" * (90 if has_sota else 75))
    print(f"\n* Best enhancer: {report.best_enhancer} (Quality Score: {report.best_quality_score})")


def save_report(report: BenchmarkReport, output_dir: Path):
    """Save benchmark report to files."""
    json_path = output_dir / f"benchmark_{report.timestamp}.json"
    txt_path = output_dir / f"benchmark_{report.timestamp}.txt"

    # Save JSON
    with open(json_path, 'w') as f:
        json.dump(asdict(report), f, indent=2)

    # Save text report
    with open(txt_path, 'w') as f:
        f.write("AUDIO ENHANCER BENCHMARK REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {report.timestamp}\n")
        f.write(f"Reference: {report.reference_video}\n")
        f.write(f"Input: {report.input_file}\n")
        f.write(f"Duration: {report.input_duration_s}s\n\n")

        f.write("RESULTS\n")
        f.write("-" * 60 + "\n\n")

        for result in report.results:
            f.write(f"Enhancer: {result['enhancer']}\n")
            f.write(f"  Status: {'SUCCESS' if result['success'] else 'FAILED'}\n")
            f.write(f"  Time: {result['processing_time_s']:.1f}s\n")

            if result['success'] and result['metrics']:
                f.write(f"  Quality Score: {result['metrics']['quality_score']:.1f}\n")
                f.write(f"  SNR: {result['metrics']['snr_db']:.1f} dB\n")
                f.write(f"  Noise Level: {result['metrics']['noise_level_db']:.1f} dB\n")
                f.write(f"  Clarity: {result['metrics']['clarity_score']:.2f}\n")
                f.write(f"  Dynamic Range: {result['metrics']['dynamic_range_db']:.1f} dB\n")
                # Add SOTA metrics if available
                if result.get('sota_metrics'):
                    sota = result['sota_metrics']
                    if sota.get('dnsmos_ovrl'):
                        f.write(f"  DNSMOS OVRL: {sota['dnsmos_ovrl']:.2f}\n")
                        f.write(f"  DNSMOS SIG: {sota['dnsmos_sig']:.2f}\n")
                        f.write(f"  DNSMOS BAK: {sota['dnsmos_bak']:.2f}\n")
            else:
                f.write(f"  Error: {result['error_message']}\n")
            f.write("\n")

        f.write("-" * 60 + "\n")
        f.write(f"\nBest Enhancer: {report.best_enhancer}\n")
        f.write(f"Best Quality Score: {report.best_quality_score}\n")

    return json_path, txt_path


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive enhancer benchmark suite"
    )
    parser.add_argument(
        "--enhancers",
        nargs="+",
        default=AVAILABLE_ENHANCERS,
        choices=AVAILABLE_ENHANCERS,
        help=f"Enhancers to test (default: all available)"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Input audio file (default: download reference video)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "benchmarks",
        help="Output directory for results"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode (shorter test sample)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output"
    )
    parser.add_argument(
        "--sota",
        action="store_true",
        help="Include SOTA metrics (DNSMOS) - slower but more comprehensive"
    )

    args = parser.parse_args()
    verbose = not args.quiet

    if verbose:
        print("\n" + "=" * 60)
        print("AUDIO ENHANCER BENCHMARK SUITE")
        print("=" * 60)

    # Run benchmark
    report = run_benchmark(
        enhancers=args.enhancers,
        input_path=args.input,
        output_dir=args.output_dir,
        quick=args.quick,
        include_sota=args.sota,
        verbose=verbose
    )

    # Print comparison
    print_comparison_table(report)

    # Save reports
    json_path, txt_path = save_report(report, args.output_dir)

    if verbose:
        print(f"\nReports saved:")
        print(f"  JSON: {json_path}")
        print(f"  Text: {txt_path}")

    # Return exit code based on success
    all_success = all(r['success'] for r in report.results)
    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
