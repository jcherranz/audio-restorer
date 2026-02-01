#!/usr/bin/env python3
"""
Audio Restoration Tool for YouTube Conference Videos
=====================================================

A simple tool to enhance audio quality in conference recordings by:
- Reducing background noise
- Enhancing speech clarity  
- Normalizing audio levels

Usage:
    python run.py <youtube_url> [options]
    
Examples:
    python run.py "https://youtu.be/cglDoG0GzyA"
    python run.py "https://youtu.be/cglDoG0GzyA" --audio-only
    python run.py "https://youtu.be/cglDoG0GzyA" --comparison

For help:
    python run.py --help
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import TEMP_DIR, OUTPUT_DIR, ENHANCEMENT
from src.pipeline import AudioRestorationPipeline


def create_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Restore and enhance audio from YouTube conference videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - enhance video
  python run.py "https://youtu.be/cglDoG0GzyA"
  
  # Audio only (faster, smaller files)
  python run.py "https://youtu.be/cglDoG0GzyA" --audio-only
  
  # Create comparison video (shows before/after side by side)
  python run.py "https://youtu.be/cglDoG0GzyA" --comparison
  
  # Quick mode (no ML, just ffmpeg filters)
  python run.py "https://youtu.be/cglDoG0GzyA" --quick
  
  # Keep temporary files for debugging
  python run.py "https://youtu.be/cglDoG0GzyA" --keep-temp
        """
    )
    
    parser.add_argument(
        "url",
        help="YouTube URL to process"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Custom output filename (without extension)"
    )
    
    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="Download and enhance audio only (no video)"
    )
    
    parser.add_argument(
        "--comparison",
        action="store_true",
        help="Create side-by-side comparison video"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: use only ffmpeg filters (faster, no ML models)"
    )
    
    parser.add_argument(
        "--enhancer",
        type=str,
        choices=["simple", "torch", "torch_advanced", "deepfilter"],
        default=None,
        help="Enhancer type: simple (ffmpeg), torch (ML), torch_advanced (ML+VAD), deepfilter (neural network - best quality)"
    )
    
    parser.add_argument(
        "--noise-reduction",
        type=float,
        default=ENHANCEMENT["noise_reduction_strength"],
        help=f"Noise reduction strength 0.0-1.0 (default: {ENHANCEMENT['noise_reduction_strength']})"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary files for debugging"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )

    parser.add_argument(
        "--dereverb",
        action="store_true",
        help="Enable de-reverberation (removes room echo) - applied after denoising"
    )

    return parser


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate URL
    if not args.url or "youtube.com" not in args.url and "youtu.be" not in args.url:
        print("❌ Error: Please provide a valid YouTube URL")
        parser.print_help()
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║           🎙️ AUDIO RESTORATION TOOL v1.0                     ║
║           YouTube Conference Audio Enhancer                  ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Determine enhancer type
    if args.enhancer:
        enhancer_type = args.enhancer
    elif args.quick:
        enhancer_type = "simple"
    else:
        enhancer_type = ENHANCEMENT.get("enhancer_type", "torch_advanced")
    
    # Initialize pipeline
    pipeline = AudioRestorationPipeline(
        temp_dir=TEMP_DIR,
        output_dir=output_dir,
        enhancer_type=enhancer_type,
        noise_reduction_strength=args.noise_reduction,
        fallback_to_simple=ENHANCEMENT.get("fallback_to_simple", True),
        keep_temp_files=args.keep_temp,
        dereverb=args.dereverb,
        verbose=not args.quiet
    )
    
    # Run restoration
    result = pipeline.restore(
        url=args.url,
        output_name=args.output,
        audio_only=args.audio_only,
        create_comparison=args.comparison
    )
    
    # Exit with appropriate code
    if result.success:
        print("\n🎉 All done! Enjoy your enhanced audio/video.")
        sys.exit(0)
    else:
        print(f"\n💥 Restoration failed: {result.error_message}")
        sys.exit(1)


if __name__ == "__main__":
    main()
