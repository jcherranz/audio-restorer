#!/usr/bin/env python3
"""
Audio Comparison Tool
Creates a side-by-side comparison of original and enhanced audio
"""

import sys
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from config import FFMPEG_PATH


def create_waveform_comparison(original_path: Path, enhanced_path: Path, output_path: Path):
    """Create a visual waveform comparison"""
    
    print("Creating waveform comparison...")
    
    # Load audio (first 30 seconds for visualization)
    orig_audio, orig_sr = sf.read(original_path, start=0, frames=30 * 48000)
    enh_audio, enh_sr = sf.read(enhanced_path, start=0, frames=30 * 16000)
    
    # Convert to mono if stereo
    if len(orig_audio.shape) > 1:
        orig_audio = np.mean(orig_audio, axis=1)
    if len(enh_audio.shape) > 1:
        enh_audio = np.mean(enh_audio, axis=1)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 6))
    
    # Time axes
    orig_time = np.linspace(0, len(orig_audio) / orig_sr, len(orig_audio))
    enh_time = np.linspace(0, len(enh_audio) / enh_sr, len(enh_audio))
    
    # Plot original
    ax1.plot(orig_time, orig_audio, color='#3498db', linewidth=0.5, alpha=0.8)
    ax1.set_title('Original Audio', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Amplitude')
    ax1.set_xlim(0, 30)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # Plot enhanced
    ax2.plot(enh_time, enh_audio, color='#2ecc71', linewidth=0.5, alpha=0.8)
    ax2.set_title('Enhanced Audio', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Amplitude')
    ax2.set_xlim(0, 30)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Waveform comparison saved: {output_path}")
    
    # Calculate and display stats
    print("\n=== Audio Statistics ===")
    print(f"Original - RMS: {np.sqrt(np.mean(orig_audio**2)):.4f}, Peak: {np.max(np.abs(orig_audio)):.4f}")
    print(f"Enhanced - RMS: {np.sqrt(np.mean(enh_audio**2)):.4f}, Peak: {np.max(np.abs(enh_audio)):.4f}")


def create_split_audio_sample(original_path: Path, enhanced_path: Path, output_path: Path, duration: int = 10):
    """
    Create a split audio file: left channel = original, right channel = enhanced
    This allows A/B comparison with headphones (left ear = original, right ear = enhanced)
    """
    print(f"\nCreating A/B comparison audio (first {duration}s)...")
    
    # Load segments
    orig_audio, orig_sr = sf.read(original_path, start=0, frames=duration * 48000)
    enh_audio, enh_sr = sf.read(enhanced_path, start=0, frames=duration * 16000)
    
    # Convert to mono
    if len(orig_audio.shape) > 1:
        orig_audio = np.mean(orig_audio, axis=1)
    if len(enh_audio.shape) > 1:
        enh_audio = np.mean(enh_audio, axis=1)
    
    # Resample enhanced to match original sample rate if needed
    if enh_sr != orig_sr:
        import librosa
        enh_audio = librosa.resample(enh_audio, orig_sr=enh_sr, target_sr=orig_sr)
    
    # Trim to same length
    min_len = min(len(orig_audio), len(enh_audio))
    orig_audio = orig_audio[:min_len]
    enh_audio = enh_audio[:min_len]
    
    # Create stereo: left = original, right = enhanced
    stereo = np.column_stack((orig_audio, enh_audio))
    
    sf.write(output_path, stereo, orig_sr)
    print(f"✓ A/B comparison audio saved: {output_path}")
    print("  Listen with headphones: LEFT ear = original, RIGHT ear = enhanced")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare original and enhanced audio")
    parser.add_argument("original", help="Path to original audio file")
    parser.add_argument("enhanced", help="Path to enhanced audio file")
    parser.add_argument("-o", "--output-dir", default="output", help="Output directory")
    parser.add_argument("--waveform", action="store_true", help="Create waveform comparison image")
    parser.add_argument("--ab-sample", action="store_true", help="Create A/B comparison audio")
    parser.add_argument("--duration", type=int, default=10, help="Duration for A/B sample (seconds)")
    parser.add_argument("--all", action="store_true", help="Create all comparisons")
    
    args = parser.parse_args()
    
    orig_path = Path(args.original)
    enh_path = Path(args.enhanced)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if not orig_path.exists():
        print(f"Error: Original file not found: {orig_path}")
        sys.exit(1)
    if not enh_path.exists():
        print(f"Error: Enhanced file not found: {enh_path}")
        sys.exit(1)
    
    print(f"Comparing:\n  Original: {orig_path}\n  Enhanced: {enh_path}")
    
    if args.waveform or args.all:
        create_waveform_comparison(
            orig_path, 
            enh_path, 
            output_dir / "waveform_comparison.png"
        )
    
    if args.ab_sample or args.all:
        create_split_audio_sample(
            orig_path,
            enh_path,
            output_dir / "ab_comparison.wav",
            args.duration
        )
    
    print("\n✅ Comparison complete!")
