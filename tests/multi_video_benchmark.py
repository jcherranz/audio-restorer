#!/usr/bin/env python3
"""
Multi-Video Benchmark Suite
===========================

Tests audio enhancement across multiple videos to validate quality improvements
are generalizable, not overfitted to a single reference video.

Usage:
    # Test with 3 videos (default)
    python tests/multi_video_benchmark.py

    # Test with specific number of videos
    python tests/multi_video_benchmark.py --videos 5

    # Include SOTA metrics (slower but more comprehensive)
    python tests/multi_video_benchmark.py --sota

    # Test specific enhancers
    python tests/multi_video_benchmark.py --enhancers deepfilter torch_advanced

    # Quick mode (first 60 seconds only)
    python tests/multi_video_benchmark.py --quick

Output:
    - benchmarks/multi_video_YYYYMMDD_HHMMSS/
        - summary.json (aggregate statistics)
        - summary.txt (human-readable report)
        - video_<id>/metrics.json (per-video results)
"""

import argparse
import json
import sys
import time
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict, field
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.measure_quality import measure_all


@dataclass
class VideoResult:
    """Results for a single video."""
    video_id: str
    video_type: str
    success: bool
    processing_time_s: float
    error_message: str = ""
    metrics: Optional[Dict] = None
    sota_metrics: Optional[Dict] = None


@dataclass
class EnhancerStats:
    """Aggregate statistics for an enhancer across all videos."""
    enhancer: str
    videos_tested: int
    videos_passed: int
    avg_quality_score: float
    std_quality_score: float
    avg_snr: float
    std_snr: float
    avg_clarity: float
    std_clarity: float
    avg_processing_time_s: float
    # SOTA metrics (if available)
    avg_dnsmos_ovrl: Optional[float] = None
    avg_dnsmos_sig: Optional[float] = None
    avg_dnsmos_bak: Optional[float] = None


@dataclass
class MultiVideoBenchmarkReport:
    """Complete multi-video benchmark report."""
    timestamp: str
    num_videos: int
    enhancers_tested: List[str]
    video_results: Dict[str, List[Dict]]  # enhancer -> list of VideoResult
    enhancer_stats: List[Dict]  # List of EnhancerStats
    best_enhancer: str
    best_avg_quality: float


def load_reference_videos(max_videos: int = 5) -> List[Dict]:
    """Load video list from reference_videos.json."""
    videos_file = Path(__file__).parent / "reference_videos.json"

    if not videos_file.exists():
        # Fallback to primary reference video
        return [{
            "id": "cglDoG0GzyA",
            "url": "https://youtu.be/cglDoG0GzyA",
            "type": "conference",
            "notes": "Primary reference video"
        }]

    with open(videos_file) as f:
        data = json.load(f)

    videos = data.get("videos", [])
    return videos[:max_videos]


def download_video_audio(video: Dict, output_dir: Path,
                         quick: bool = False, verbose: bool = True) -> Optional[Path]:
    """Download and extract audio from a video."""
    from src.downloader import YouTubeDownloader

    video_id = video["id"]
    cached_audio = output_dir / f"reference_{video_id}.wav"

    # Check cache
    if cached_audio.exists():
        if verbose:
            print(f"  Using cached: {cached_audio.name}")
        return cached_audio

    if verbose:
        print(f"  Downloading: {video['url']}")

    try:
        downloader = YouTubeDownloader(output_dir=output_dir, verbose=False)
        result = downloader.download(video["url"], audio_only=True)

        if result.success and result.audio_path:
            # Rename to standard format
            shutil.move(str(result.audio_path), str(cached_audio))
            return cached_audio
        else:
            if verbose:
                print(f"  Download failed: {result.error_message}")
            return None
    except Exception as e:
        if verbose:
            print(f"  Error: {e}")
        return None


def run_enhancer(audio_path: Path, enhancer_type: str,
                 output_dir: Path, verbose: bool = True) -> VideoResult:
    """Run an enhancer on a single audio file."""
    from src.pipeline import AudioRestorationPipeline

    video_id = audio_path.stem.replace("reference_", "")
    output_path = output_dir / f"enhanced_{enhancer_type}.wav"

    start_time = time.time()

    try:
        # Create pipeline with specified enhancer
        pipeline = AudioRestorationPipeline(
            temp_dir=output_dir / "temp",
            output_dir=output_dir,
            enhancer_type=enhancer_type,
            keep_temp_files=False,
            verbose=False
        )

        # Get the enhancer and run directly on audio
        enhancer = pipeline._create_enhancer()
        enhancer.enhance(audio_path, output_path)

        processing_time = time.time() - start_time

        # Measure quality
        metrics = measure_all(output_path)

        return VideoResult(
            video_id=video_id,
            video_type="unknown",
            success=True,
            processing_time_s=processing_time,
            metrics=metrics.to_dict() if hasattr(metrics, 'to_dict') else asdict(metrics)
        )

    except Exception as e:
        return VideoResult(
            video_id=video_id,
            video_type="unknown",
            success=False,
            processing_time_s=time.time() - start_time,
            error_message=str(e)
        )


def calculate_sota_metrics(audio_path: Path) -> Optional[Dict]:
    """Calculate SOTA metrics (DNSMOS) for an audio file."""
    try:
        from tests.sota_benchmark import SOTAMetricsCalculator
        calculator = SOTAMetricsCalculator(verbose=False)
        metrics = calculator.calculate_all(audio_path)
        return metrics
    except Exception as e:
        return {"error": str(e)}


def calculate_enhancer_stats(results: List[VideoResult], enhancer: str) -> EnhancerStats:
    """Calculate aggregate statistics for an enhancer."""
    successful = [r for r in results if r.success and r.metrics]

    if not successful:
        return EnhancerStats(
            enhancer=enhancer,
            videos_tested=len(results),
            videos_passed=0,
            avg_quality_score=0.0,
            std_quality_score=0.0,
            avg_snr=0.0,
            std_snr=0.0,
            avg_clarity=0.0,
            std_clarity=0.0,
            avg_processing_time_s=0.0
        )

    quality_scores = [r.metrics.get("quality_score", 0) for r in successful]
    snrs = [r.metrics.get("snr_db", 0) for r in successful]
    clarities = [r.metrics.get("clarity_score", 0) for r in successful]
    times = [r.processing_time_s for r in successful]

    stats = EnhancerStats(
        enhancer=enhancer,
        videos_tested=len(results),
        videos_passed=len(successful),
        avg_quality_score=float(np.mean(quality_scores)),
        std_quality_score=float(np.std(quality_scores)),
        avg_snr=float(np.mean(snrs)),
        std_snr=float(np.std(snrs)),
        avg_clarity=float(np.mean(clarities)),
        std_clarity=float(np.std(clarities)),
        avg_processing_time_s=float(np.mean(times))
    )

    # Add SOTA metrics if available
    dnsmos_ovrl = []
    dnsmos_sig = []
    dnsmos_bak = []

    for r in successful:
        if r.sota_metrics and "dnsmos" in r.sota_metrics:
            dnsmos = r.sota_metrics["dnsmos"]
            if "ovrl" in dnsmos:
                dnsmos_ovrl.append(dnsmos["ovrl"])
            if "sig" in dnsmos:
                dnsmos_sig.append(dnsmos["sig"])
            if "bak" in dnsmos:
                dnsmos_bak.append(dnsmos["bak"])

    if dnsmos_ovrl:
        stats.avg_dnsmos_ovrl = float(np.mean(dnsmos_ovrl))
    if dnsmos_sig:
        stats.avg_dnsmos_sig = float(np.mean(dnsmos_sig))
    if dnsmos_bak:
        stats.avg_dnsmos_bak = float(np.mean(dnsmos_bak))

    return stats


def print_summary_table(stats_list: List[EnhancerStats]):
    """Print a comparison table of all enhancers."""
    print("\n" + "=" * 80)
    print("MULTI-VIDEO BENCHMARK SUMMARY")
    print("=" * 80)

    # Header
    print(f"{'Enhancer':<15} {'Pass':<6} {'Quality':<12} {'SNR':<12} {'Clarity':<12} {'Time':<8}")
    print("-" * 80)

    # Find best quality for highlighting
    best_quality = max(s.avg_quality_score for s in stats_list)

    for stats in sorted(stats_list, key=lambda x: x.avg_quality_score, reverse=True):
        marker = "*" if stats.avg_quality_score == best_quality else " "
        print(f"{stats.enhancer:<14}{marker} "
              f"{stats.videos_passed}/{stats.videos_tested:<4} "
              f"{stats.avg_quality_score:>5.1f} ± {stats.std_quality_score:<4.1f} "
              f"{stats.avg_snr:>5.1f} ± {stats.std_snr:<4.1f} "
              f"{stats.avg_clarity:>5.2f} ± {stats.std_clarity:<4.2f} "
              f"{stats.avg_processing_time_s:>6.1f}s")

    print("-" * 80)
    print("* = Best quality score")

    # DNSMOS table if available
    has_dnsmos = any(s.avg_dnsmos_ovrl is not None for s in stats_list)
    if has_dnsmos:
        print("\nDNSMOS Scores (Average):")
        print(f"{'Enhancer':<15} {'OVRL':<8} {'SIG':<8} {'BAK':<8}")
        print("-" * 40)
        for stats in sorted(stats_list, key=lambda x: x.avg_dnsmos_ovrl or 0, reverse=True):
            ovrl = f"{stats.avg_dnsmos_ovrl:.2f}" if stats.avg_dnsmos_ovrl else "N/A"
            sig = f"{stats.avg_dnsmos_sig:.2f}" if stats.avg_dnsmos_sig else "N/A"
            bak = f"{stats.avg_dnsmos_bak:.2f}" if stats.avg_dnsmos_bak else "N/A"
            print(f"{stats.enhancer:<15} {ovrl:<8} {sig:<8} {bak:<8}")


def run_multi_video_benchmark(
    num_videos: int = 3,
    enhancers: List[str] = None,
    include_sota: bool = False,
    quick: bool = False,
    verbose: bool = True
) -> MultiVideoBenchmarkReport:
    """
    Run benchmark across multiple videos.

    Args:
        num_videos: Number of videos to test
        enhancers: List of enhancers to test (default: all)
        include_sota: Include DNSMOS metrics
        quick: Use shorter audio segments
        verbose: Print progress

    Returns:
        MultiVideoBenchmarkReport with all results
    """
    if enhancers is None:
        enhancers = ["simple", "torch_advanced", "deepfilter"]

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("benchmarks") / f"multi_video_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'='*60}")
        print(f"MULTI-VIDEO BENCHMARK")
        print(f"{'='*60}")
        print(f"Videos: {num_videos}")
        print(f"Enhancers: {', '.join(enhancers)}")
        print(f"SOTA metrics: {'Yes' if include_sota else 'No'}")
        print(f"Output: {output_dir}")
        print(f"{'='*60}\n")

    # Load videos
    videos = load_reference_videos(num_videos)
    if verbose:
        print(f"Loaded {len(videos)} reference videos")

    # Results storage
    all_results: Dict[str, List[VideoResult]] = {e: [] for e in enhancers}

    # Process each video
    for i, video in enumerate(videos, 1):
        video_id = video["id"]
        video_dir = output_dir / f"video_{video_id}"
        video_dir.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"\n[{i}/{len(videos)}] Video: {video_id} ({video.get('type', 'unknown')})")

        # Download audio
        audio_path = download_video_audio(video, video_dir, quick=quick, verbose=verbose)

        if audio_path is None:
            if verbose:
                print(f"  Skipping (download failed)")
            continue

        # Test each enhancer
        for enhancer in enhancers:
            if verbose:
                print(f"  Testing: {enhancer}...", end=" ", flush=True)

            enhancer_dir = video_dir / enhancer
            enhancer_dir.mkdir(exist_ok=True)

            result = run_enhancer(audio_path, enhancer, enhancer_dir, verbose=False)
            result.video_type = video.get("type", "unknown")

            # Add SOTA metrics if requested
            if include_sota and result.success:
                enhanced_path = enhancer_dir / f"enhanced_{enhancer}.wav"
                if enhanced_path.exists():
                    result.sota_metrics = calculate_sota_metrics(enhanced_path)

            all_results[enhancer].append(result)

            if verbose:
                if result.success:
                    qs = result.metrics.get("quality_score", 0)
                    print(f"Quality: {qs:.1f}")
                else:
                    print(f"FAILED: {result.error_message[:50]}")

    # Calculate statistics
    stats_list = []
    for enhancer in enhancers:
        stats = calculate_enhancer_stats(all_results[enhancer], enhancer)
        stats_list.append(stats)

    # Find best enhancer
    best_stats = max(stats_list, key=lambda x: x.avg_quality_score)

    # Create report
    report = MultiVideoBenchmarkReport(
        timestamp=timestamp,
        num_videos=len(videos),
        enhancers_tested=enhancers,
        video_results={e: [asdict(r) for r in results] for e, results in all_results.items()},
        enhancer_stats=[asdict(s) for s in stats_list],
        best_enhancer=best_stats.enhancer,
        best_avg_quality=best_stats.avg_quality_score
    )

    # Save report
    report_path = output_dir / "summary.json"
    with open(report_path, 'w') as f:
        json.dump(asdict(report), f, indent=2)

    # Print summary
    if verbose:
        print_summary_table(stats_list)
        print(f"\nReport saved: {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Multi-video benchmark for audio enhancers"
    )
    parser.add_argument(
        "--videos", type=int, default=3,
        help="Number of videos to test (default: 3)"
    )
    parser.add_argument(
        "--enhancers", nargs="+",
        default=["simple", "torch_advanced", "deepfilter"],
        help="Enhancers to test"
    )
    parser.add_argument(
        "--sota", action="store_true",
        help="Include SOTA metrics (DNSMOS)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode (shorter audio)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Minimal output"
    )

    args = parser.parse_args()

    report = run_multi_video_benchmark(
        num_videos=args.videos,
        enhancers=args.enhancers,
        include_sota=args.sota,
        quick=args.quick,
        verbose=not args.quiet
    )

    # Exit with error if no videos passed
    total_passed = sum(
        sum(1 for r in results if r["success"])
        for results in report.video_results.values()
    )

    if total_passed == 0:
        print("\nERROR: No videos were successfully processed")
        sys.exit(1)

    print(f"\nBest enhancer: {report.best_enhancer} (avg quality: {report.best_avg_quality:.1f})")


if __name__ == "__main__":
    main()
