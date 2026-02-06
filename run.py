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

from config import TEMP_DIR, OUTPUT_DIR, ENHANCEMENT, PRESETS
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
        help="Enhancer type: simple (ffmpeg), torch (ML), torch_advanced (ML+VAD), deepfilter (neural network)"
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

    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Perform speaker diarization (identifies who is speaking when)"
    )

    parser.add_argument(
        "--isolate-speaker",
        action="store_true",
        help="Isolate main speaker - removes audience and other speakers"
    )

    parser.add_argument(
        "--distance-robust",
        action="store_true",
        help="Apply distance-robust enhancement (adaptive gain/EQ per speaker distance)"
    )

    parser.add_argument(
        "--speaker-agc",
        action="store_true",
        help="Apply per-speaker automatic gain control (normalize each speaker's volume)"
    )

    parser.add_argument(
        "--deess",
        action="store_true",
        help="Apply de-essing to reduce harsh sibilant sounds (/s/, /sh/)"
    )

    parser.add_argument(
        "--remove-hum",
        action="store_true",
        help="Remove power line hum (50/60 Hz) and harmonics"
    )

    parser.add_argument(
        "--remove-clicks",
        action="store_true",
        help="Remove clicks and pops (transient artifacts)"
    )

    parser.add_argument(
        "--comfort-noise",
        action="store_true",
        help="Add comfort noise to silence regions (prevents dead air)"
    )

    preset_names = list(PRESETS.keys())
    preset_help = "Use a curated preset: " + ", ".join(
        f"{k} ({v['description']})" for k, v in PRESETS.items()
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=preset_names,
        default=None,
        help=preset_help
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
    
    # Apply preset defaults (individual flags override preset)
    preset_flags = {}
    if args.preset:
        preset_flags = PRESETS[args.preset].copy()
        preset_flags.pop("description", None)
        if not args.quiet:
            print(f"  Using preset: {args.preset} ({PRESETS[args.preset]['description']})")

    # Determine enhancer type
    if args.enhancer:
        enhancer_type = args.enhancer
    elif args.quick:
        enhancer_type = "simple"
    elif "enhancer_type" in preset_flags:
        enhancer_type = preset_flags.pop("enhancer_type")
    else:
        enhancer_type = ENHANCEMENT.get("enhancer_type", "torch_advanced")

    # Merge preset flags with CLI flags (CLI wins)
    def flag_or_preset(flag_name):
        cli_val = getattr(args, flag_name, False)
        return cli_val or preset_flags.get(flag_name, False)

    # Initialize pipeline
    pipeline = AudioRestorationPipeline(
        temp_dir=TEMP_DIR,
        output_dir=output_dir,
        enhancer_type=enhancer_type,
        noise_reduction_strength=args.noise_reduction,
        fallback_to_simple=ENHANCEMENT.get("fallback_to_simple", True),
        keep_temp_files=args.keep_temp,
        dereverb=flag_or_preset("dereverb"),
        diarize=flag_or_preset("diarize"),
        isolate_speaker=flag_or_preset("isolate_speaker"),
        distance_robust=flag_or_preset("distance_robust"),
        speaker_agc=flag_or_preset("speaker_agc"),
        deess=flag_or_preset("deess"),
        remove_hum=flag_or_preset("remove_hum"),
        remove_clicks=flag_or_preset("remove_clicks"),
        comfort_noise=flag_or_preset("comfort_noise"),
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
