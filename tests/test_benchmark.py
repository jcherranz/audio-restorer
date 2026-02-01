#!/usr/bin/env python3
"""
Benchmark Suite
===============

Runs the pipeline with different configurations and benchmarks quality.

Usage:
    python test_benchmark.py
    python test_benchmark.py --full

Results are saved to benchmarks/
"""

import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import OUTPUT_DIR


REFERENCE_VIDEO = "https://youtu.be/cglDoG0GzyA"
BENCHMARK_DIR = Path(__file__).parent.parent / "benchmarks"


def run_benchmark(config_name: str, extra_args: list) -> dict:
    """
    Run pipeline with specific configuration and measure results.
    
    Returns:
        dict with timing, file info, and quality metrics
    """
    print(f"\n{'=' * 60}")
    print(f"Benchmark: {config_name}")
    print(f"{'=' * 60}")
    
    output_name = f"bench_{config_name}"
    
    # Run pipeline
    start_time = time.time()
    
    cmd = [
        sys.executable, "run.py",
        REFERENCE_VIDEO,
        "--audio-only",
        "--keep-temp",
        "-o", output_name
    ] + extra_args
    
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=600
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode != 0:
            print(f"❌ Pipeline failed: {result.stderr}")
            return None
        
        # Find output file
        output_files = list(OUTPUT_DIR.glob(f"{output_name}_enhanced.wav"))
        if not output_files:
            print("❌ Output file not found")
            return None
        
        output_file = output_files[0]
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        
        # Measure quality
        quality_result = subprocess.run(
            [sys.executable, "tests/measure_quality.py", str(output_file)],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Parse quality metrics from output
        quality_metrics = {}
        if quality_result.returncode == 0:
            for line in quality_result.stdout.split('\n'):
                if 'Signal-to-Noise Ratio:' in line:
                    quality_metrics['snr_db'] = float(line.split(':')[1].split()[0])
                elif 'Noise Level:' in line:
                    quality_metrics['noise_level_db'] = float(line.split(':')[1].split()[0])
                elif 'Dynamic Range:' in line:
                    quality_metrics['dynamic_range_db'] = float(line.split(':')[1].split()[0])
                elif 'Loudness:' in line:
                    quality_metrics['loudness_lufs'] = float(line.split(':')[1].split()[0])
                elif 'Clarity Score:' in line:
                    quality_metrics['clarity_score'] = float(line.split(':')[1].split()[0])
                elif 'OVERALL QUALITY SCORE:' in line:
                    quality_metrics['quality_score'] = float(line.split(':')[1].split()[0])
        
        benchmark_result = {
            'config': config_name,
            'timestamp': datetime.now().isoformat(),
            'elapsed_time': elapsed,
            'file_size_mb': file_size_mb,
            'quality_metrics': quality_metrics,
            'args': extra_args
        }
        
        print(f"✅ Completed in {elapsed:.1f}s")
        print(f"   Quality Score: {quality_metrics.get('quality_score', 'N/A')}")
        
        return benchmark_result
        
    except subprocess.TimeoutExpired:
        print(f"❌ Timeout after 600s")
        return None
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None


def run_all_benchmarks(full: bool = False):
    """Run benchmark suite"""
    print("\n" + "=" * 60)
    print("AUDIO QUALITY BENCHMARK SUITE")
    print("=" * 60)
    print(f"Reference Video: {REFERENCE_VIDEO}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    
    configs = [
        ("quick_nr_0.5", ["--quick", "--noise-reduction", "0.5"]),
        ("quick_nr_0.8", ["--quick", "--noise-reduction", "0.8"]),
        ("quick_nr_0.95", ["--quick", "--noise-reduction", "0.95"]),
    ]
    
    if full:
        # Add more configurations for full benchmark
        configs.extend([
            ("spectral_nr_0.7", ["--noise-reduction", "0.7"]),
            ("spectral_nr_0.9", ["--noise-reduction", "0.9"]),
        ])
    
    results = []
    
    for config_name, extra_args in configs:
        result = run_benchmark(config_name, extra_args)
        if result:
            results.append(result)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = BENCHMARK_DIR / f"benchmark_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    if results:
        print(f"\n{'Config':<20} {'Time':<10} {'Quality':<10} {'SNR':<10}")
        print("-" * 60)
        
        for r in results:
            qm = r['quality_metrics']
            print(f"{r['config']:<20} "
                  f"{r['elapsed_time']:<10.1f} "
                  f"{qm.get('quality_score', 0):<10.1f} "
                  f"{qm.get('snr_db', 0):<10.1f}")
        
        # Find best
        best = max(results, key=lambda x: x['quality_metrics'].get('quality_score', 0))
        print("-" * 60)
        print(f"Best Config: {best['config']}")
        print(f"Quality Score: {best['quality_metrics'].get('quality_score', 0):.1f}")
        
    print(f"\nResults saved to: {results_file}")
    print("=" * 60)
    
    return results


def cleanup():
    """Remove benchmark files"""
    print("\nCleaning up benchmark files...")
    count = 0
    for f in OUTPUT_DIR.glob("bench_*"):
        try:
            f.unlink()
            count += 1
        except:
            pass
    print(f"✓ Removed {count} benchmark files")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run quality benchmarks")
    parser.add_argument("--full", action="store_true",
                       help="Run full benchmark suite (slower)")
    parser.add_argument("--keep", action="store_true",
                       help="Keep benchmark files")
    
    args = parser.parse_args()
    
    results = run_all_benchmarks(full=args.full)
    
    if not args.keep:
        cleanup()
