"""
YouTube Video Downloader Module
Handles downloading videos and extracting audio
"""

import os
import re
from pathlib import Path
from typing import Optional, Tuple
import yt_dlp
from tqdm import tqdm


class YouTubeDownloader:
    """Downloads videos from YouTube and extracts audio"""
    
    def __init__(self, temp_dir: Path, verbose: bool = True):
        self.temp_dir = temp_dir
        self.verbose = verbose
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def _progress_hook(self, d):
        """Display download progress"""
        if d['status'] == 'downloading':
            percent = d.get('_percent_str', '0%')
            speed = d.get('_speed_str', 'N/A')
            eta = d.get('_eta_str', 'N/A')
            if self.verbose:
                print(f"\r  Downloading: {percent} | Speed: {speed} | ETA: {eta}", end='', flush=True)
        elif d['status'] == 'finished':
            print(f"\n  ✓ Download complete: {d['filename']}")
    
    def extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from various YouTube URL formats"""
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',
            r'(?:embed\/)([0-9A-Za-z_-]{11})',
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def download_video(self, url: str, filename: Optional[str] = None) -> Path:
        """
        Download video from YouTube
        
        Args:
            url: YouTube URL
            filename: Optional custom filename (without extension)
            
        Returns:
            Path to downloaded video file
        """
        video_id = self.extract_video_id(url)
        if not video_id:
            raise ValueError(f"Could not extract video ID from URL: {url}")
        
        if not filename:
            filename = f"video_{video_id}"
        
        output_path = self.temp_dir / f"{filename}.%(ext)s"
        
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from config import FFMPEG_PATH
        
        ydl_opts = {
            'format': 'bestvideo[height<=1080]+bestaudio/best',
            'merge_output_format': 'mp4',
            'outtmpl': str(output_path),
            'progress_hooks': [self._progress_hook],
            'quiet': not self.verbose,
            'no_warnings': not self.verbose,
            'ffmpeg_location': FFMPEG_PATH,
        }
        
        print(f"📥 Downloading video: {url}")
        print(f"   Video ID: {video_id}")
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            actual_filename = ydl.prepare_filename(info)
            # Handle merged format
            if info.get('ext') != 'mp4':
                actual_filename = actual_filename.rsplit('.', 1)[0] + '.mp4'
        
        video_path = Path(actual_filename)
        if not video_path.exists():
            # Try to find the file
            possible_files = list(self.temp_dir.glob(f"{filename}*"))
            if possible_files:
                video_path = possible_files[0]
        
        print(f"✓ Video saved to: {video_path}")
        return video_path
    
    def extract_audio(self, video_path: Path, output_name: Optional[str] = None) -> Path:
        """
        Extract audio from video file using ffmpeg
        
        Args:
            video_path: Path to video file
            output_name: Optional output filename (without extension)
            
        Returns:
            Path to extracted audio file
        """
        if not output_name:
            output_name = video_path.stem + "_audio"
        
        audio_path = self.temp_dir / f"{output_name}.wav"
        
        print(f"\n🔊 Extracting audio from video...")
        
        # Use ffmpeg to extract audio
        import subprocess
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from config import FFMPEG_PATH
        cmd = [
            FFMPEG_PATH, '-y', '-i', str(video_path),
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit little-endian
            '-ar', '16000',  # 16kHz (good for speech)
            '-ac', '1',  # Mono
            str(audio_path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✓ Audio extracted to: {audio_path}")
            return audio_path
        except subprocess.CalledProcessError as e:
            print(f"❌ Error extracting audio: {e.stderr}")
            raise
    
    def download_audio_only(self, url: str, filename: Optional[str] = None) -> Path:
        """
        Download audio directly (faster, no video download needed)
        
        Args:
            url: YouTube URL
            filename: Optional custom filename
            
        Returns:
            Path to downloaded audio file
        """
        video_id = self.extract_video_id(url)
        if not video_id:
            raise ValueError(f"Could not extract video ID from URL: {url}")
        
        if not filename:
            filename = f"audio_{video_id}"
        
        output_path = self.temp_dir / f"{filename}.wav"
        
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from config import FFMPEG_PATH
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'outtmpl': str(self.temp_dir / filename),
            'progress_hooks': [self._progress_hook],
            'quiet': not self.verbose,
            'ffmpeg_location': FFMPEG_PATH,
        }
        
        print(f"📥 Downloading audio: {url}")
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=True)
        
        # yt-dlp adds extension, so find the actual file
        audio_path = self.temp_dir / f"{filename}.wav"
        if not audio_path.exists():
            # Try to find any matching file
            possible_files = list(self.temp_dir.glob(f"{filename}*"))
            for f in possible_files:
                if f.suffix in ['.wav', '.m4a', '.webm', '.mp3']:
                    audio_path = f
                    break
        
        print(f"\n✓ Audio saved to: {audio_path}")
        return audio_path


if __name__ == "__main__":
    # Test the downloader
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import TEMP_DIR, VERBOSE
    
    test_url = "https://www.youtube.com/watch?v=cglDoG0GzyA"
    downloader = YouTubeDownloader(TEMP_DIR, VERBOSE)
    
    # Test video download
    video_path = downloader.download_video(test_url)
    print(f"\nVideo path: {video_path}")
    
    # Test audio extraction
    audio_path = downloader.extract_audio(video_path)
    print(f"Audio path: {audio_path}")
