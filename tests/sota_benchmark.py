#!/usr/bin/env python3
"""
SOTA Benchmark Script
=====================
Run comprehensive SOTA metrics on enhanced audio.

This script calculates industry-standard speech quality metrics
for enhanced audio files:

- DNSMOS: Neural MOS predictor (no reference needed)
  - SIG: Speech signal quality (1-5)
  - BAK: Background noise quality (1-5)
  - OVRL: Overall quality (1-5)

- PESQ: Perceptual quality (1-4.5, needs reference)
- STOI: Speech intelligibility (0-1, needs reference)
- SI-SDR: Signal distortion ratio (dB, needs reference)

Usage:
    # DNSMOS only (no reference needed)
    python tests/sota_benchmark.py output/enhanced.wav

    # All metrics (with reference)
    python tests/sota_benchmark.py output/enhanced.wav --reference original.wav

    # Compare multiple files
    python tests/sota_benchmark.py output/*.wav

Examples:
    # Test DeepFilter output
    python tests/sota_benchmark.py output/audio_cglDoG0GzyA_enhanced.wav

    # Compare enhancers
    python tests/sota_benchmark.py output/deepfilter.wav output/torch_advanced.wav
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sota_metrics import SOTAMetricsCalculator, SOTAMetrics


def format_score(value, max_val, precision=2):
    """Format a score with color indication."""
    if value is None:
        return "N/A"

    formatted = f"{value:.{precision}f}"

    # Add quality indicator
    ratio = value / max_val
    if ratio >= 0.8:
        indicator = "(excellent)"
    elif ratio >= 0.6:
        indicator = "(good)"
    elif ratio >= 0.4:
        indicator = "(fair)"
    else:
        indicator = "(poor)"

    return f"{formatted} {indicator}"


def print_results(file_path: Path, metrics: SOTAMetrics):
    """Print formatted results for a single file."""
    print(f"\n{'='*60}")
    print(f"File: {file_path.name}")
    print('='*60)

    # DNSMOS (always available)
    if metrics.dnsmos_ovrl is not None:
        print("\nDNSMOS Scores (1-5 scale, higher is better):")
        print(f"  Overall (OVRL): {format_score(metrics.dnsmos_ovrl, 5.0)}")
        print(f"  Speech (SIG):   {format_score(metrics.dnsmos_sig, 5.0)}")
        print(f"  Background:     {format_score(metrics.dnsmos_bak, 5.0)}")
    else:
        print("\nDNSMOS: Not available")
        print("  Install with: pip install onnxruntime")

    # Reference-based metrics
    if metrics.pesq is not None or metrics.stoi is not None:
        print("\nReference-Based Metrics:")

        if metrics.pesq is not None:
            print(f"  PESQ (1.0-4.5):  {format_score(metrics.pesq, 4.5)}")
        else:
            print("  PESQ: Not available (pip install pesq)")

        if metrics.stoi is not None:
            print(f"  STOI (0.0-1.0):  {format_score(metrics.stoi, 1.0, 3)}")
        else:
            print("  STOI: Not available (pip install pystoi)")

        if metrics.si_sdr is not None:
            # SI-SDR doesn't have a fixed max, so just show the value
            if metrics.si_sdr > 20:
                quality = "(excellent)"
            elif metrics.si_sdr > 10:
                quality = "(good)"
            elif metrics.si_sdr > 0:
                quality = "(fair)"
            else:
                quality = "(poor)"
            print(f"  SI-SDR (dB):     {metrics.si_sdr:.1f} {quality}")


def print_comparison_table(results: list):
    """Print a comparison table for multiple files."""
    if len(results) < 2:
        return

    print(f"\n{'='*80}")
    print("COMPARISON TABLE")
    print('='*80)

    # Header
    print(f"{'File':<30} {'DNSMOS':<10} {'SIG':<8} {'BAK':<8} {'PESQ':<8} {'STOI':<8}")
    print('-'*80)

    for file_path, metrics in results:
        name = file_path.stem[:28]

        dnsmos = f"{metrics.dnsmos_ovrl:.2f}" if metrics.dnsmos_ovrl else "N/A"
        sig = f"{metrics.dnsmos_sig:.2f}" if metrics.dnsmos_sig else "N/A"
        bak = f"{metrics.dnsmos_bak:.2f}" if metrics.dnsmos_bak else "N/A"
        pesq = f"{metrics.pesq:.2f}" if metrics.pesq else "N/A"
        stoi = f"{metrics.stoi:.3f}" if metrics.stoi else "N/A"

        print(f"{name:<30} {dnsmos:<10} {sig:<8} {bak:<8} {pesq:<8} {stoi:<8}")

    # Find best
    if all(m.dnsmos_ovrl for _, m in results):
        best_idx = max(range(len(results)), key=lambda i: results[i][1].dnsmos_ovrl)
        print(f"\nBest DNSMOS: {results[best_idx][0].name}")


def main():
    parser = argparse.ArgumentParser(
        description='Calculate SOTA quality metrics for enhanced audio',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tests/sota_benchmark.py output/enhanced.wav
    python tests/sota_benchmark.py output/enhanced.wav --reference clean.wav
    python tests/sota_benchmark.py output/*.wav
        """
    )
    parser.add_argument(
        'enhanced',
        type=Path,
        nargs='+',
        help='Enhanced audio file(s) to analyze'
    )
    parser.add_argument(
        '--reference',
        type=Path,
        help='Clean reference audio (optional, enables PESQ/STOI/SI-SDR)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Only show final summary'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )

    args = parser.parse_args()

    # Validate inputs
    for f in args.enhanced:
        if not f.exists():
            print(f"Error: File not found: {f}")
            sys.exit(1)

    if args.reference and not args.reference.exists():
        print(f"Error: Reference file not found: {args.reference}")
        sys.exit(1)

    # Initialize calculator
    calc = SOTAMetricsCalculator(verbose=not args.quiet)

    # Process files
    results = []

    for enhanced_path in args.enhanced:
        metrics = calc.calculate(enhanced_path, args.reference)
        results.append((enhanced_path, metrics))

        if not args.quiet and not args.json:
            print_results(enhanced_path, metrics)

    # Output
    if args.json:
        import json
        output = {
            'files': [
                {
                    'path': str(p),
                    'metrics': m.to_dict()
                }
                for p, m in results
            ]
        }
        print(json.dumps(output, indent=2))
    else:
        # Print comparison table if multiple files
        if len(results) > 1:
            print_comparison_table(results)

        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print('='*60)

        if args.reference:
            print(f"Reference: {args.reference.name}")
        else:
            print("Reference: None (only DNSMOS available)")

        print(f"Files analyzed: {len(results)}")

        # Average DNSMOS if available
        dnsmos_scores = [m.dnsmos_ovrl for _, m in results if m.dnsmos_ovrl]
        if dnsmos_scores:
            avg = sum(dnsmos_scores) / len(dnsmos_scores)
            print(f"Average DNSMOS: {avg:.2f}")

        # Quality interpretation
        if dnsmos_scores:
            avg = sum(dnsmos_scores) / len(dnsmos_scores)
            if avg >= 4.0:
                print("\nQuality: EXCELLENT - Broadcast quality")
            elif avg >= 3.5:
                print("\nQuality: GOOD - Clear and natural")
            elif avg >= 3.0:
                print("\nQuality: FAIR - Acceptable quality")
            else:
                print("\nQuality: POOR - Needs improvement")


if __name__ == '__main__':
    main()
