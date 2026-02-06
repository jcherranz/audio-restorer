"""
Audio Enhancement Module
Simple ffmpeg-based audio enhancement for basic noise reduction.
"""

import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


class SimpleEnhancer:
    """
    Simple enhancer using only ffmpeg (no Python ML libraries)
    Good for basic noise reduction when ML models aren't available
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def enhance(self, audio_path: Path, output_path: Path, **kwargs) -> Path:
        """Simple ffmpeg-based audio cleanup"""
        print(f"\n🔧 Applying ffmpeg filters: {audio_path.name}")

        # ffmpeg filter chain for speech enhancement
        # - highpass: Remove low-frequency rumble
        # - lowpass: Remove high-frequency hiss
        # - compand: Compress dynamic range (evens out loud/quiet parts)
        # - loudnorm: Normalize loudness to broadcast standard

        filter_chain = (
            "highpass=f=100,"           # Remove rumble below 100Hz
            "lowpass=f=12000,"          # Remove hiss above 12kHz (preserves consonant clarity)
            "compand=attacks=.05:decays=.5:points=-90/-90 -70/-70 -15/-10 0/-10:soft-knee=6:volume=-12,"  # Compress dynamics
            "loudnorm=I=-16:TP=-1.5:LRA=11"  # Normalize to -16 LUFS (broadcast standard)
        )

        from config import FFMPEG_PATH
        cmd = [
            FFMPEG_PATH, '-y', '-i', str(audio_path),
            '-af', filter_chain,
            '-ar', '48000',  # Match project-wide sample rate
            '-ac', '1',     # Mono (consistent with all other enhancers)
            str(output_path)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✓ Enhancement complete: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"❌ ffmpeg enhancement failed: {e.stderr}")
            raise
