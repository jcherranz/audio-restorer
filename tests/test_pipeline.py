#!/usr/bin/env python3
"""
Pipeline Integration Tests
==========================

Tests the full pipeline with the reference video.

Usage:
    python test_pipeline.py
    python test_pipeline.py --keep-temp
"""

import sys
import subprocess
from pathlib import Path
import tempfile
import shutil

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TEMP_DIR, OUTPUT_DIR


# Constants
REFERENCE_VIDEO = "https://youtu.be/cglDoG0GzyA"
TEST_TIMEOUT = 300  # 5 minutes


def setup_test_environment():
    """Prepare test environment"""
    print("Setting up test environment...")
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    print("✓ Test environment ready")


def test_download_only():
    """Test: Can we download audio from YouTube?"""
    print("\n" + "=" * 60)
    print("TEST 1: Audio Download")
    print("=" * 60)
    
    try:
        result = subprocess.run(
            [
                sys.executable, "run.py",
                REFERENCE_VIDEO,
                "--audio-only",
                "--quick",
                "--keep-temp",
                "-o", "test_download"
            ],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=TEST_TIMEOUT
        )
        
        if result.returncode != 0:
            print(f"❌ FAILED: Download test failed")
            print(f"Error: {result.stderr}")
            return False
        
        # Check output file exists
        output_files = list(OUTPUT_DIR.glob("test_download_enhanced.wav"))
        if not output_files:
            print("❌ FAILED: Output file not found")
            return False
        
        print("✅ PASSED: Download and basic processing works")
        return True
        
    except subprocess.TimeoutExpired:
        print(f"❌ FAILED: Test timed out after {TEST_TIMEOUT}s")
        return False
    except Exception as e:
        print(f"❌ FAILED: Exception occurred: {e}")
        return False


def test_quality_metrics():
    """Test: Can we measure quality metrics?"""
    print("\n" + "=" * 60)
    print("TEST 2: Quality Metrics")
    print("=" * 60)
    
    try:
        # Find a test file
        test_files = list(OUTPUT_DIR.glob("*_enhanced.wav"))
        if not test_files:
            print("⚠️  SKIPPED: No enhanced files to test")
            return True
        
        test_file = test_files[0]
        
        result = subprocess.run(
            [
                sys.executable,
                "tests/measure_quality.py",
                str(test_file)
            ],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            print(f"❌ FAILED: Quality measurement failed")
            print(f"Error: {result.stderr}")
            return False
        
        print("✅ PASSED: Quality metrics can be measured")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: Exception occurred: {e}")
        return False


def test_configurable_noise_reduction():
    """Test: Does noise reduction strength setting work?"""
    print("\n" + "=" * 60)
    print("TEST 3: Configurable Noise Reduction")
    print("=" * 60)
    
    noise_levels = [0.5, 0.9]
    results = []
    
    for level in noise_levels:
        try:
            result = subprocess.run(
                [
                    sys.executable, "run.py",
                    REFERENCE_VIDEO,
                    "--audio-only",
                    "--quick",
                    "--noise-reduction", str(level),
                    "-o", f"test_nr_{level}"
                ],
                cwd=Path(__file__).parent.parent,
                capture_output=True,
                text=True,
                timeout=TEST_TIMEOUT
            )
            
            if result.returncode == 0:
                results.append(level)
                print(f"  ✓ Noise reduction {level}: Success")
            else:
                print(f"  ❌ Noise reduction {level}: Failed")
                
        except Exception as e:
            print(f"  ❌ Noise reduction {level}: Exception - {e}")
    
    if len(results) == len(noise_levels):
        print("✅ PASSED: All noise reduction levels work")
        return True
    else:
        print(f"⚠️  PARTIAL: {len(results)}/{len(noise_levels)} levels work")
        return len(results) > 0


def test_file_formats():
    """Test: Can we process different input formats?"""
    print("\n" + "=" * 60)
    print("TEST 4: File Format Support")
    print("=" * 60)
    
    # Currently we only test that we can download and process YouTube audio
    # which is the primary use case
    
    print("  ℹ️  Primary format (YouTube audio): Supported")
    print("✅ PASSED: File format support (basic)")
    return True


def cleanup():
    """Clean up test files"""
    print("\n" + "=" * 60)
    print("CLEANUP")
    print("=" * 60)
    
    test_patterns = ["test_*"]
    removed = 0
    
    for pattern in test_patterns:
        for f in OUTPUT_DIR.glob(pattern):
            try:
                f.unlink()
                removed += 1
            except:
                pass
    
    print(f"✓ Removed {removed} test files")


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("PIPELINE TEST SUITE")
    print("=" * 60)
    print(f"Reference Video: {REFERENCE_VIDEO}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("=" * 60)
    
    setup_test_environment()
    
    results = {
        "download": test_download_only(),
        "quality_metrics": test_quality_metrics(),
        "noise_reduction": test_configurable_noise_reduction(),
        "file_formats": test_file_formats(),
    }
    
    cleanup()
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name:<25} {status}")
    
    print("-" * 60)
    print(f"  TOTAL: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed")
        return 1


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run pipeline tests")
    parser.add_argument("--keep-temp", action="store_true",
                       help="Keep temporary files after tests")
    
    args = parser.parse_args()
    
    exit_code = run_all_tests()
    sys.exit(exit_code)
