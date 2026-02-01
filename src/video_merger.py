"""
Video Merger Module
Combines enhanced audio with original video
"""

import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from typing import Optional


class VideoMerger:
    """Merges enhanced audio back into video"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def merge(self, 
              video_path: Path, 
              audio_path: Path, 
              output_path: Path,
              video_codec: str = "copy",
              audio_codec: str = "aac",
              audio_bitrate: str = "192k") -> Path:
        """
        Merge enhanced audio with video
        
        Args:
            video_path: Path to original video
            audio_path: Path to enhanced audio
            output_path: Path for output video
            video_codec: Video codec ("copy" to avoid re-encoding)
            audio_codec: Audio codec for output
            audio_bitrate: Audio bitrate
            
        Returns:
            Path to output video
        """
        print(f"\n🎬 Merging enhanced audio with video...")
        print(f"   Video: {video_path.name}")
        print(f"   Audio: {audio_path.name}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        from config import FFMPEG_PATH
        cmd = [
            FFMPEG_PATH, '-y',
            '-i', str(video_path),      # Video input
            '-i', str(audio_path),      # Audio input
            '-c:v', video_codec,        # Video codec (copy = no re-encode)
            '-c:a', audio_codec,        # Audio codec
            '-b:a', audio_bitrate,      # Audio bitrate
            '-map', '0:v:0',            # Take video from first input
            '-map', '1:a:0',            # Take audio from second input
            '-shortest',                # Match shortest duration
            str(output_path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✓ Merged video saved: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"❌ Merge failed: {e.stderr}")
            raise
    
    def create_side_by_side(self,
                           original_video: Path,
                           enhanced_video: Path,
                           output_path: Path) -> Path:
        """
        Create a side-by-side comparison video
        Useful for comparing before/after
        """
        print(f"\n🎬 Creating side-by-side comparison...")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # ffmpeg filter to place videos side by side with labels
        filter_complex = (
            "[0:v]scale=960:540,setsar=1[orig];"
            "[1:v]scale=960:540,setsar=1[enh];"
            "[orig][enh]hstack=inputs=2[v];"
            "[v]drawtext=text='Original':x=20:y=20:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5[vo];"
            "[vo]drawtext=text='Enhanced':x=980:y=20:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5[outv]"
        )
        
        from config import FFMPEG_PATH
        cmd = [
            FFMPEG_PATH, '-y',
            '-i', str(original_video),
            '-i', str(enhanced_video),
            '-filter_complex', filter_complex,
            '-map', '[outv]',
            '-map', '1:a:0',  # Use enhanced audio
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '128k',
            str(output_path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✓ Comparison video saved: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"❌ Comparison creation failed: {e.stderr}")
            raise


if __name__ == "__main__":
    print("VideoMerger module - requires test files to run")
    print("This module is used by the main pipeline")
