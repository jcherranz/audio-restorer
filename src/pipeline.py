"""
Audio Restoration Pipeline
Main orchestrator that coordinates the entire process
"""

import shutil
import sys
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass
from datetime import datetime

# Add parent to path for config
sys.path.insert(0, str(Path(__file__).parent.parent))

from .downloader import YouTubeDownloader
from .audio_enhancer import SimpleEnhancer  # AudioEnhancer is legacy, not used


@dataclass
class RestorationResult:
    """Result of the restoration process"""
    original_video: Path
    enhanced_video: Path
    original_audio: Path
    enhanced_audio: Path
    comparison_video: Optional[Path] = None
    processing_time: float = 0.0
    success: bool = False
    error_message: str = ""


class AudioRestorationPipeline:
    """
    Main pipeline for restoring audio in conference videos
    
    Usage:
        pipeline = AudioRestorationPipeline()
        result = pipeline.restore("https://youtube.com/...")
    """
    
    def __init__(self,
                 temp_dir: Path,
                 output_dir: Path,
                 sample_rate: int = 48000,  # Match DeepFilterNet native rate
                 enhancer_type: str = "torch_advanced",
                 noise_reduction_strength: float = 0.8,
                 use_gpu: bool = False,
                 fallback_to_simple: bool = True,
                 keep_temp_files: bool = False,
                 dereverb: bool = False,
                 diarize: bool = False,
                 isolate_speaker: bool = False,
                 distance_robust: bool = False,
                 verbose: bool = True):
        """
        Initialize the pipeline.

        Args:
            temp_dir: Directory for temporary files
            output_dir: Directory for output files
            sample_rate: Target sample rate for processing
            enhancer_type: "simple", "torch", "torch_advanced", or "deepfilter"
            noise_reduction_strength: 0.0 to 1.0
            use_gpu: Whether to use GPU (if available)
            fallback_to_simple: Fall back to simple enhancer on failure
            keep_temp_files: Keep temp files for debugging
            dereverb: Apply de-reverberation after enhancement
            diarize: Perform speaker diarization (identify speakers)
            isolate_speaker: Isolate main speaker (remove others)
            distance_robust: Apply distance-robust enhancement (adaptive gain/EQ)
            verbose: Print progress messages
        """
        from config import ENHANCEMENT, DEREVERB
        self._enhancement_config = ENHANCEMENT
        
        self.temp_dir = Path(temp_dir)
        self.output_dir = Path(output_dir)
        self.keep_temp_files = keep_temp_files
        self.verbose = verbose
        self.sample_rate = sample_rate
        self.fallback_to_simple = fallback_to_simple
        
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize downloader
        self.downloader = YouTubeDownloader(self.temp_dir, verbose)
        
        # Initialize enhancer based on type
        self.enhancer_type = enhancer_type
        self.enhancer = self._create_enhancer(
            enhancer_type=enhancer_type,
            noise_reduction_strength=noise_reduction_strength,
            use_gpu=use_gpu,
            verbose=verbose
        )
        
        # Fallback enhancer
        self.simple_enhancer = SimpleEnhancer(verbose)

        # De-reverberation (optional, applied after primary enhancement)
        self.dereverb_enabled = dereverb or DEREVERB.get("enabled", False)
        self.dereverb_enhancer = None
        if self.dereverb_enabled:
            try:
                from .dereverb_enhancer import DereverbEnhancer
                self.dereverb_enhancer = DereverbEnhancer(
                    taps=DEREVERB.get("taps", 10),
                    delay=DEREVERB.get("delay", 3),
                    iterations=DEREVERB.get("iterations", 3),
                    verbose=verbose
                )
                if verbose:
                    print("De-reverberation enabled (NARA-WPE)")
            except ImportError as e:
                if verbose:
                    print(f"⚠ Could not load de-reverberator: {e}")
                self.dereverb_enabled = False

        # Speaker diarization (optional, identifies who is speaking)
        self.diarize_enabled = diarize
        self.diarizer = None
        if self.diarize_enabled:
            try:
                from .diarization import SpeakerDiarizer
                self.diarizer = SpeakerDiarizer(
                    use_gpu=use_gpu,
                    verbose=verbose
                )
                if verbose:
                    print("Speaker diarization enabled")
            except ImportError as e:
                if verbose:
                    print(f"⚠ Could not load diarization: {e}")
                self.diarize_enabled = False

        # Speaker isolation (optional, removes other speakers/audience)
        self.isolate_enabled = isolate_speaker
        self.isolator = None
        if self.isolate_enabled:
            try:
                from .speaker_isolation import SpeakerIsolator
                self.isolator = SpeakerIsolator(verbose=verbose)
                if verbose:
                    print("Speaker isolation enabled")
            except ImportError as e:
                if verbose:
                    print(f"⚠ Could not load speaker isolation: {e}")
                self.isolate_enabled = False

        # Distance-robust enhancement (optional, adaptive gain/EQ per speaker)
        self.distance_robust_enabled = distance_robust
        self.distance_enhancer = None
        if self.distance_robust_enabled:
            try:
                from .distance_enhancer import DistanceRobustEnhancer
                self.distance_enhancer = DistanceRobustEnhancer(verbose=verbose)
                if verbose:
                    print("Distance-robust enhancement enabled")
            except ImportError as e:
                if verbose:
                    print(f"⚠ Could not load distance enhancer: {e}")
                self.distance_robust_enabled = False
    
    def _create_enhancer(self, 
                         enhancer_type: str,
                         noise_reduction_strength: float,
                         use_gpu: bool,
                         verbose: bool):
        """Create the appropriate enhancer based on type"""
        
        if enhancer_type == "simple":
            if verbose:
                print("Using SimpleEnhancer (ffmpeg-based)")
            return SimpleEnhancer(verbose)

        if enhancer_type == "deepfilter":
            try:
                from .deepfilter_enhancer import DeepFilterNetEnhancer
                if verbose:
                    print("Using DeepFilterNet (neural denoising)")
                return DeepFilterNetEnhancer(
                    noise_reduction_strength=noise_reduction_strength,
                    use_gpu=use_gpu,
                    verbose=verbose
                )
            except ImportError as e:
                if verbose:
                    print(f"⚠ Could not load DeepFilterNet: {e}")
                    print("  Falling back to AdvancedMLEnhancer")
                # Fall through to torch_advanced
                enhancer_type = "torch_advanced"

        if enhancer_type in ("torch", "torch_advanced"):
            try:
                from .ml_enhancer import TorchEnhancer, AdvancedMLEnhancer
                
                if enhancer_type == "torch_advanced":
                    if verbose:
                        print("Using AdvancedMLEnhancer (PyTorch with VAD)")
                    return AdvancedMLEnhancer(
                        noise_reduction_strength=noise_reduction_strength,
                        use_gpu=use_gpu,
                        verbose=verbose,
                        use_vad=True
                    )
                else:
                    if verbose:
                        print("Using TorchEnhancer (PyTorch-based)")
                    return TorchEnhancer(
                        noise_reduction_strength=noise_reduction_strength,
                        use_gpu=use_gpu,
                        verbose=verbose
                    )
            except ImportError as e:
                if verbose:
                    print(f"⚠ Could not load ML enhancer: {e}")
                    print("  Falling back to SimpleEnhancer")
                return SimpleEnhancer(verbose)
        
        else:
            raise ValueError(f"Unknown enhancer type: {enhancer_type}")
    
    def _normalize_loudness(self, audio_path: Path, target_lufs: float = -16.0) -> None:
        """
        Apply EBU R128 loudness normalization.

        Uses ffmpeg's loudnorm filter to normalize audio to a target loudness level.
        This ensures consistent playback volume across different audio files.

        Args:
            audio_path: Path to audio file (modified in-place)
            target_lufs: Target integrated loudness in LUFS (default: -16, podcast standard)
        """
        import subprocess
        import tempfile

        if self.verbose:
            print(f"\n🔊 Normalizing loudness to {target_lufs} LUFS...")

        # Create temp file for normalized output
        temp_output = audio_path.parent / f"{audio_path.stem}_normalized.wav"

        try:
            # Run ffmpeg loudnorm filter
            # I=-16: Target integrated loudness
            # TP=-1.5: Target true peak (prevent clipping)
            # LRA=11: Target loudness range (natural dynamics)
            cmd = [
                'ffmpeg', '-y',
                '-i', str(audio_path),
                '-af', f'loudnorm=I={target_lufs}:TP=-1.5:LRA=11',
                '-ar', '48000',  # Keep 48kHz sample rate
                str(temp_output)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                if self.verbose:
                    print(f"  ⚠ Loudness normalization warning: {result.stderr}")
                return

            # Replace original with normalized version
            import shutil
            shutil.move(str(temp_output), str(audio_path))

            if self.verbose:
                print(f"  ✓ Loudness normalized to {target_lufs} LUFS")

        except Exception as e:
            if self.verbose:
                print(f"  ⚠ Loudness normalization failed: {e}")
            # Clean up temp file if it exists
            if temp_output.exists():
                temp_output.unlink()

    def restore(self,
                url: str,
                output_name: Optional[str] = None,
                audio_only: bool = False,
                create_comparison: bool = False) -> RestorationResult:
        """
        Restore audio from a YouTube video
        
        Args:
            url: YouTube URL
            output_name: Custom output name (without extension)
            audio_only: If True, only output enhanced audio (no video)
            create_comparison: If True, create side-by-side comparison video
            
        Returns:
            RestorationResult with paths to all outputs
        """
        import time
        from .video_merger import VideoMerger
        
        start_time = time.time()
        
        result = RestorationResult(
            original_video=Path(),
            enhanced_video=Path(),
            original_audio=Path(),
            enhanced_audio=Path(),
            success=False
        )
        
        try:
            print("=" * 60)
            print("🎙️ AUDIO RESTORATION PIPELINE")
            print("=" * 60)
            print(f"Source: {url}")
            print(f"Enhancer: {self.enhancer_type}")
            print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60)
            
            # Step 1: Download
            if audio_only:
                print("\n📥 STEP 1: Downloading audio only...")
                audio_path = self.downloader.download_audio_only(url, output_name)
                video_path = None
            else:
                print("\n📥 STEP 1: Downloading video...")
                video_path = self.downloader.download_video(url, output_name)
                
                # Extract audio from video
                audio_path = self.downloader.extract_audio(video_path)
            
            result.original_audio = audio_path
            result.original_video = video_path or audio_path
            
            # Step 2: Enhance audio
            print("\n" + "=" * 60)
            enhanced_audio_path = self.output_dir / f"{audio_path.stem}_enhanced.wav"
            
            try:
                # Try primary enhancer
                self.enhancer.enhance(audio_path, enhanced_audio_path, target_sr=self.sample_rate)
            except Exception as e:
                print(f"\n⚠ Primary enhancer failed: {e}")

                if self.fallback_to_simple:
                    print("  Falling back to simple ffmpeg enhancement...")
                    self.simple_enhancer.enhance(audio_path, enhanced_audio_path)
                else:
                    raise

            # Apply de-reverberation if enabled (after noise reduction)
            if self.dereverb_enabled and self.dereverb_enhancer:
                try:
                    # De-reverb works on the enhanced audio (in-place)
                    dereverb_output = self.output_dir / f"{audio_path.stem}_dereverb.wav"
                    self.dereverb_enhancer.enhance(enhanced_audio_path, dereverb_output)
                    # Replace enhanced audio with dereverberated version
                    import shutil
                    shutil.move(str(dereverb_output), str(enhanced_audio_path))
                except Exception as e:
                    print(f"\n⚠ De-reverberation failed: {e}")
                    print("  Continuing with enhanced audio (no dereverb)")

            # Apply loudness normalization (EBU R128)
            target_lufs = self._enhancement_config.get("target_loudness", -16)
            self._normalize_loudness(enhanced_audio_path, target_lufs=target_lufs)

            result.enhanced_audio = enhanced_audio_path
            
            # Step 3: Merge (if video was downloaded)
            if not audio_only and video_path:
                print("\n" + "=" * 60)
                merger = VideoMerger(self.verbose)
                enhanced_video_path = self.output_dir / f"{video_path.stem}_enhanced.mp4"
                merger.merge(video_path, enhanced_audio_path, enhanced_video_path)
                result.enhanced_video = enhanced_video_path
                
                # Step 4: Create comparison video (optional)
                if create_comparison:
                    print("\n" + "=" * 60)
                    comparison_path = self.output_dir / f"{video_path.stem}_comparison.mp4"
                    merger.create_side_by_side(video_path, enhanced_video_path, comparison_path)
                    result.comparison_video = comparison_path
            
            # Step 5: Speaker diarization (optional)
            diarization_output = None
            if self.diarize_enabled and self.diarizer:
                try:
                    print("\n" + "=" * 60)
                    diarization_output = self.output_dir / f"{audio_path.stem}_diarization.json"
                    diarization_result = self.diarizer.diarize(
                        enhanced_audio_path,
                        output_json=diarization_output
                    )
                    self.diarizer.print_summary()
                except Exception as e:
                    if self.verbose:
                        print(f"\n⚠ Diarization failed: {e}")
                        print("  Continuing without speaker analysis")
            
            # Step 6: Speaker isolation (optional)
            if self.isolate_enabled and self.isolator:
                try:
                    print("\n" + "=" * 60)
                    isolated_output = self.output_dir / f"{audio_path.stem}_isolated.wav"

                    # If we have diarization, use it; otherwise run diarization first
                    if diarization_output and diarization_output.exists():
                        isolation_result = self.isolator.isolate_main_speaker(
                            enhanced_audio_path,
                            diarization_output,
                            isolated_output
                        )
                    else:
                        isolation_result = self.isolator.isolate_with_diarization(
                            enhanced_audio_path,
                            isolated_output
                        )

                    # Replace enhanced audio with isolated version
                    import shutil
                    shutil.move(str(isolated_output), str(enhanced_audio_path))

                    if self.verbose:
                        print(f"\n   Isolated main speaker: {isolation_result.main_speaker}")
                        print(f"   Retained: {isolation_result.retention_percentage:.1f}% of audio")
                        print(f"   Removed: {timedelta(seconds=int(isolation_result.removed_duration))} of interference")

                except Exception as e:
                    if self.verbose:
                        print(f"\n⚠ Speaker isolation failed: {e}")
                        print("  Continuing with all speakers")

            # Step 7: Distance-robust enhancement (optional)
            if self.distance_robust_enabled and self.distance_enhancer:
                try:
                    print("\n" + "=" * 60)
                    distance_output = self.output_dir / f"{audio_path.stem}_distance_enhanced.wav"

                    # Use diarization if available for per-speaker analysis
                    distance_result = self.distance_enhancer.enhance(
                        enhanced_audio_path,
                        distance_output,
                        diarization_json=diarization_output if diarization_output and diarization_output.exists() else None
                    )

                    # Replace enhanced audio with distance-enhanced version
                    import shutil
                    shutil.move(str(distance_output), str(enhanced_audio_path))

                    if self.verbose:
                        print(f"\n   Segments analyzed: {distance_result.segments_processed}")
                        print(f"   Average distance: {distance_result.average_distance:.2f} (0=close, 1=far)")
                        if distance_result.gain_adjustments:
                            print(f"   Gain range: {min(distance_result.gain_adjustments):.1f} to {max(distance_result.gain_adjustments):.1f} dB")

                except Exception as e:
                    if self.verbose:
                        print(f"\n⚠ Distance-robust enhancement failed: {e}")
                        print("  Continuing without distance compensation")

            # Success!
            result.success = True
            result.processing_time = time.time() - start_time
            
            print("\n" + "=" * 60)
            print("✅ RESTORATION COMPLETE!")
            print("=" * 60)
            print(f"⏱️  Processing time: {result.processing_time:.1f} seconds")
            print(f"\n📁 Output files:")
            if result.enhanced_video.exists():
                print(f"   Video: {result.enhanced_video}")
            print(f"   Audio: {result.enhanced_audio}")
            if result.comparison_video:
                print(f"   Comparison: {result.comparison_video}")
            if self.diarize_enabled:
                print(f"   Diarization: {diarization_output}")
            if self.distance_robust_enabled:
                print(f"   Distance-robust: Applied (adaptive gain/EQ)")
            print("=" * 60)
            
            # Cleanup
            if not self.keep_temp_files:
                self._cleanup()
            
        except Exception as e:
            result.error_message = str(e)
            result.processing_time = time.time() - start_time
            print("\n" + "=" * 60)
            print("❌ RESTORATION FAILED!")
            print("=" * 60)
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            print("=" * 60)
        
        return result
    
    def restore_batch(self, urls: List[str], **kwargs) -> List[RestorationResult]:
        """Restore multiple videos"""
        results = []
        for i, url in enumerate(urls, 1):
            print(f"\n\nProcessing video {i}/{len(urls)}")
            result = self.restore(url, **kwargs)
            results.append(result)
        return results
    
    def _cleanup(self):
        """Remove temporary files"""
        if self.verbose:
            print("\n🧹 Cleaning up temporary files...")
        try:
            shutil.rmtree(self.temp_dir)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            if self.verbose:
                print("  ✓ Cleanup complete")
        except Exception as e:
            if self.verbose:
                print(f"  ⚠ Cleanup warning: {e}")


def quick_restore(url: str, output_dir: Optional[Path] = None) -> RestorationResult:
    """
    Quick one-liner to restore a video
    
    Args:
        url: YouTube URL
        output_dir: Where to save output (default: ./output)
        
    Returns:
        RestorationResult
    """
    from config import TEMP_DIR, OUTPUT_DIR, ENHANCEMENT
    
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    pipeline = AudioRestorationPipeline(
        temp_dir=TEMP_DIR,
        output_dir=output_dir,
        enhancer_type=ENHANCEMENT.get("enhancer_type", "torch_advanced"),
        noise_reduction_strength=ENHANCEMENT.get("noise_reduction_strength", 0.8),
        verbose=True
    )
    
    return pipeline.restore(url)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from config import TEMP_DIR, OUTPUT_DIR, ENHANCEMENT
    
    # Example usage
    test_url = "https://www.youtube.com/watch?v=cglDoG0GzyA"
    
    pipeline = AudioRestorationPipeline(
        temp_dir=TEMP_DIR,
        output_dir=OUTPUT_DIR,
        enhancer_type=ENHANCEMENT.get("enhancer_type", "torch_advanced"),
        verbose=True
    )
    
    result = pipeline.restore(test_url)
    
    if result.success:
        print("\n✅ Success!")
    else:
        print(f"\n❌ Failed: {result.error_message}")
