#!/usr/bin/env python3
"""
Quality Gate Script
===================

Automated quality check that enforces minimum quality thresholds.
Run this before accepting any changes to ensure no quality regression.

Usage:
    python tests/quality_gate.py <audio_file>
    python tests/quality_gate.py output/enhanced.wav

Exit Codes:
    0 - Quality gate PASSED
    1 - Quality gate FAILED (below minimum thresholds)
    2 - Error (file not found, measurement failed, etc.)

Thresholds (configured below):
    - Quality Score >= 75.0
    - SNR >= 25.0 dB
    - Clarity >= 0.60
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.measure_quality import measure_all, QualityMetrics


# Minimum acceptable thresholds
# These values should NOT be lowered - they represent the quality floor
THRESHOLDS = {
    'quality_score': 75.0,   # Minimum overall quality score
    'snr_db': 25.0,          # Minimum signal-to-noise ratio
    'clarity_score': 0.50,   # Minimum clarity score (lowered: loudnorm affects spectral balance)
}

# Warning thresholds (not failures, but concerning)
WARNINGS = {
    'noise_level_db': -40.0,  # Warn if noise is above this
    'loudness_lufs': -20.0,   # Warn if quieter than this
}


def check_quality_gate(audio_path: Path) -> tuple[bool, dict]:
    """
    Check if audio file passes quality gate.

    Args:
        audio_path: Path to audio file to check

    Returns:
        Tuple of (passed: bool, results: dict with details)
    """
    results = {
        'passed': True,
        'failures': [],
        'warnings': [],
        'metrics': None,
    }

    # Measure quality
    try:
        metrics = measure_all(audio_path)
        results['metrics'] = metrics.to_dict()
    except Exception as e:
        results['passed'] = False
        results['failures'].append(f"Measurement failed: {e}")
        return False, results

    # Check required thresholds
    if metrics.quality_score < THRESHOLDS['quality_score']:
        results['passed'] = False
        results['failures'].append(
            f"Quality Score {metrics.quality_score:.1f} < {THRESHOLDS['quality_score']} (minimum)"
        )

    if metrics.snr_db < THRESHOLDS['snr_db']:
        results['passed'] = False
        results['failures'].append(
            f"SNR {metrics.snr_db:.1f} dB < {THRESHOLDS['snr_db']} dB (minimum)"
        )

    if metrics.clarity_score < THRESHOLDS['clarity_score']:
        results['passed'] = False
        results['failures'].append(
            f"Clarity {metrics.clarity_score:.2f} < {THRESHOLDS['clarity_score']} (minimum)"
        )

    # Check warning thresholds (don't fail, just warn)
    if metrics.noise_level_db > WARNINGS['noise_level_db']:
        results['warnings'].append(
            f"Noise level {metrics.noise_level_db:.1f} dB > {WARNINGS['noise_level_db']} dB (high)"
        )

    if metrics.loudness_lufs < WARNINGS['loudness_lufs']:
        results['warnings'].append(
            f"Loudness {metrics.loudness_lufs:.1f} LUFS < {WARNINGS['loudness_lufs']} LUFS (quiet)"
        )

    return results['passed'], results


def print_results(audio_path: Path, passed: bool, results: dict) -> None:
    """Print quality gate results in a clear format."""
    print("\n" + "=" * 60)
    print("  QUALITY GATE RESULTS")
    print("=" * 60)
    print(f"  File: {audio_path.name}")
    print("-" * 60)

    if results['metrics']:
        m = results['metrics']
        print(f"  Quality Score:  {m['quality_score']:6.1f}  (min: {THRESHOLDS['quality_score']})")
        print(f"  SNR:            {m['snr_db']:6.1f} dB  (min: {THRESHOLDS['snr_db']} dB)")
        print(f"  Clarity:        {m['clarity_score']:6.2f}  (min: {THRESHOLDS['clarity_score']})")
        print(f"  Noise Level:    {m['noise_level_db']:6.1f} dB")
        print(f"  Loudness:       {m['loudness_lufs']:6.1f} LUFS")

    print("-" * 60)

    if passed:
        print("  STATUS: PASSED")
        print("=" * 60)
    else:
        print("  STATUS: FAILED")
        print("-" * 60)
        print("  Failures:")
        for failure in results['failures']:
            print(f"    - {failure}")

    if results['warnings']:
        print("-" * 60)
        print("  Warnings:")
        for warning in results['warnings']:
            print(f"    - {warning}")

    print("=" * 60)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Quality gate check for audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Thresholds:
  Quality Score >= {THRESHOLDS['quality_score']}
  SNR >= {THRESHOLDS['snr_db']} dB
  Clarity >= {THRESHOLDS['clarity_score']}

Examples:
  python tests/quality_gate.py output/enhanced.wav
  python tests/quality_gate.py --quiet output/enhanced.wav
        """
    )

    parser.add_argument("audio_file", type=Path, help="Audio file to check")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Only output pass/fail, no details")
    parser.add_argument("--json", action="store_true",
                       help="Output results as JSON")

    args = parser.parse_args()

    # Validate input
    if not args.audio_file.exists():
        print(f"Error: File not found: {args.audio_file}", file=sys.stderr)
        sys.exit(2)

    # Run quality gate
    passed, results = check_quality_gate(args.audio_file)

    # Output results
    if args.json:
        import json
        print(json.dumps({
            'file': str(args.audio_file),
            'passed': passed,
            **results
        }, indent=2))
    elif args.quiet:
        print("PASSED" if passed else "FAILED")
    else:
        print_results(args.audio_file, passed, results)

    # Exit with appropriate code
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
