"""
Audio Restoration Pipeline
Main orchestrator that coordinates the entire process
"""

import shutil
import sys
from pathlib import Path
from typing import Optional, List  # List used in restore_batch
from dataclasses import dataclass
from datetime import datetime, timedelta

# Add parent to path for config
sys.path.insert(0, str(Path(__file__).parent.parent))

from .downloader import YouTubeDownloader
from .audio_enhancer import SimpleEnhancer


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
                 speaker_agc: bool = False,
                 deess: bool = False,
                 remove_hum: bool = False,
                 remove_clicks: bool = False,
                 comfort_noise: bool = False,
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
            speaker_agc: Apply per-speaker automatic gain control
            deess: Apply de-essing to reduce sibilant sounds (/s/, /sh/)
            remove_hum: Remove power line hum (50/60 Hz) and harmonics
            remove_clicks: Remove clicks and pops (transient artifacts)
            comfort_noise: Add comfort noise to silence regions
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

        # Optional modules — loaded on demand with graceful fallback
        self.dereverb_enabled, self.dereverb_enhancer = self._init_module(
            dereverb or DEREVERB.get("enabled", False),
            ".dereverb_enhancer", "DereverbEnhancer",
            "De-reverberation (NARA-WPE)",
            taps=DEREVERB.get("taps", 10),
            delay=DEREVERB.get("delay", 3),
            iterations=DEREVERB.get("iterations", 3),
            verbose=verbose
        )

        self.diarize_enabled, self.diarizer = self._init_module(
            diarize, ".diarization", "SpeakerDiarizer",
            "Speaker diarization", verbose=verbose
        )

        self.isolate_enabled, self.isolator = self._init_module(
            isolate_speaker, ".speaker_isolation", "SpeakerIsolator",
            "Speaker isolation", verbose=verbose
        )

        self.distance_robust_enabled, self.distance_enhancer = self._init_module(
            distance_robust, ".distance_enhancer", "DistanceRobustEnhancer",
            "Distance-robust enhancement", verbose=verbose
        )

        self.speaker_agc_enabled, self.speaker_agc = self._init_module(
            speaker_agc, ".speaker_agc", "SpeakerAGC",
            "Per-speaker AGC", verbose=verbose
        )

        self.deess_enabled, self.deesser = self._init_module(
            deess, ".deesser", "DeEsser",
            "De-essing", verbose=verbose
        )

        self.remove_hum_enabled, self.hum_remover = self._init_module(
            remove_hum, ".hum_remover", "HumRemover",
            "Hum removal (auto-detect 50/60 Hz)", verbose=verbose
        )

        self.remove_clicks_enabled, self.click_remover = self._init_module(
            remove_clicks, ".click_remover", "ClickRemover",
            "Click removal", verbose=verbose
        )

        self.comfort_noise_enabled, self.comfort_noise_generator = self._init_module(
            comfort_noise, ".comfort_noise", "ComfortNoiseGenerator",
            "Comfort noise (pink noise at -60dB)", verbose=verbose
        )

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

    def _init_module(self, enabled, module_path, class_name, label, **kwargs):
        """Try to load an optional processing module.

        Returns:
            (enabled, instance) — enabled is False if import failed.
        """
        if not enabled:
            return False, None
        try:
            import importlib
            mod = importlib.import_module(module_path, package="src")
            cls = getattr(mod, class_name)
            instance = cls(**kwargs)
            if self.verbose:
                print(f"{label} enabled")
            return True, instance
        except ImportError as e:
            if self.verbose:
                print(f"⚠ Could not load {label}: {e}")
            return False, None

    def _run_stage(self, name, processor, audio_path, suffix, method="process", **kwargs):
        """Run an optional processing stage with error handling.

        Writes to a temp file, then replaces the input on success.
        Returns True if the stage succeeded, False otherwise.
        """
        try:
            print("\n" + "=" * 60)
            stage_output = self.output_dir / f"{audio_path.stem}{suffix}.wav"
            getattr(processor, method)(audio_path, stage_output, **kwargs)
            shutil.move(str(stage_output), str(audio_path))
            return True
        except Exception as e:
            if self.verbose:
                print(f"\n⚠ {name} failed: {e}")
                print(f"  Continuing without {name.lower()}")
            return False

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

            # Pre-processing: Hum removal (if enabled, before main enhancement)
            if self.remove_hum_enabled and self.hum_remover:
                try:
                    print("\n" + "=" * 60)
                    hum_free_path = self.output_dir / f"{audio_path.stem}_nohum.wav"
                    self.hum_remover.process(audio_path, hum_free_path)
                    # Use hum-free audio for subsequent processing
                    audio_path = hum_free_path
                except Exception as e:
                    if self.verbose:
                        print(f"\n⚠ Hum removal failed: {e}")
                        print("  Continuing with original audio")

            # Pre-processing: Click removal (if enabled, before main enhancement)
            if self.remove_clicks_enabled and self.click_remover:
                try:
                    print("\n" + "=" * 60)
                    declicked_path = self.output_dir / f"{audio_path.stem}_declicked.wav"
                    self.click_remover.process(audio_path, declicked_path)
                    # Use declicked audio for subsequent processing
                    audio_path = declicked_path
                except Exception as e:
                    if self.verbose:
                        print(f"\n⚠ Click removal failed: {e}")
                        print("  Continuing with previous audio")

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
                self._run_stage("De-reverberation", self.dereverb_enhancer,
                                enhanced_audio_path, "_dereverb", method="enhance")

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
            diar_json = diarization_output if diarization_output and diarization_output.exists() else None

            if self.distance_robust_enabled and self.distance_enhancer:
                self._run_stage("Distance-robust enhancement", self.distance_enhancer,
                                enhanced_audio_path, "_distance_enhanced",
                                method="enhance", diarization_json=diar_json)

            # Step 8: Per-speaker AGC (optional)
            if self.speaker_agc_enabled and self.speaker_agc:
                self._run_stage("Per-speaker AGC", self.speaker_agc,
                                enhanced_audio_path, "_agc",
                                diarization_json=diar_json)

            # Step 9: De-essing (optional)
            if self.deess_enabled and self.deesser:
                self._run_stage("De-essing", self.deesser,
                                enhanced_audio_path, "_deessed")

            # Step 10: Comfort noise (optional)
            if self.comfort_noise_enabled and self.comfort_noise_generator:
                self._run_stage("Comfort noise", self.comfort_noise_generator,
                                enhanced_audio_path, "_comfort")

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
            if self.speaker_agc_enabled:
                print(f"   Speaker AGC: Applied (per-speaker normalization)")
            if self.deess_enabled:
                print(f"   De-essing: Applied (sibilance reduction)")
            if self.remove_hum_enabled:
                print(f"   Hum removal: Applied (50/60Hz notch filter)")
            if self.remove_clicks_enabled:
                print(f"   Click removal: Applied (transient detection)")
            if self.comfort_noise_enabled:
                print(f"   Comfort noise: Applied (pink noise at -60dB)")
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
