"""
Audio Enhancement Module
Handles noise reduction, speech enhancement, and audio cleanup
"""

import os
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import tempfile
from typing import Optional, Tuple
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm


class AudioEnhancer:
    """
    DEPRECATED: Legacy enhancer class, not used by the pipeline.

    This class was the original audio enhancer implementation.
    The pipeline now uses specialized enhancers:
    - SimpleEnhancer (ffmpeg-based)
    - TorchEnhancer / AdvancedMLEnhancer (ml_enhancer.py)
    - DeepFilterNetEnhancer (deepfilter_enhancer.py)

    Kept for backwards compatibility but may be removed in future versions.
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 use_deepfilternet: bool = True,
                 use_spectral_gating: bool = True,
                 noise_reduction_strength: float = 0.8,
                 normalize: bool = True,
                 verbose: bool = True):
        
        self.sample_rate = sample_rate
        self.use_deepfilternet = use_deepfilternet
        self.use_spectral_gating = use_spectral_gating
        self.noise_reduction_strength = noise_reduction_strength
        self.normalize = normalize
        self.verbose = verbose
        
        self._df_model = None
        
    def _load_deepfilternet(self):
        """Lazy load DeepFilterNet model"""
        if self._df_model is None and self.use_deepfilternet:
            try:
                from df import enhance, init_df
                if self.verbose:
                    print("  Loading DeepFilterNet model...")
                self._df_model = init_df()
                if self.verbose:
                    print("  ✓ DeepFilterNet loaded")
            except ImportError:
                print("  ⚠ DeepFilterNet not available, install with: pip install deepfilternet")
                self.use_deepfilternet = False
            except Exception as e:
                print(f"  ⚠ Could not load DeepFilterNet: {e}")
                self.use_deepfilternet = False
    
    def load_audio(self, audio_path: Path) -> Tuple[np.ndarray, int]:
        """Load audio file and return samples + sample rate"""
        if self.verbose:
            print(f"  Loading audio: {audio_path}")
        
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        if self.verbose:
            duration = len(audio) / sr
            print(f"  ✓ Loaded: {duration:.1f}s at {sr}Hz")
        
        return audio, sr
    
    def save_audio(self, audio: np.ndarray, output_path: Path, sr: Optional[int] = None):
        """Save audio to file"""
        if sr is None:
            sr = self.sample_rate
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, audio, sr)
        
        if self.verbose:
            print(f"  ✓ Saved: {output_path}")
    
    def apply_spectral_gating(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply spectral gating for noise reduction
        Uses noisereduce library (simple but effective)
        """
        try:
            import noisereduce as nr
            
            if self.verbose:
                print("  Applying spectral gating noise reduction...")
            
            # Estimate noise from first second (or less if audio is short)
            noise_sample_duration = min(int(sr), len(audio))
            noise_clip = audio[:noise_sample_duration]
            
            # Apply noise reduction
            reduced_noise = nr.reduce_noise(
                y=audio,
                y_noise=noise_clip,
                sr=sr,
                prop_decrease=self.noise_reduction_strength,
                stationary=True,  # Conference noise is often stationary
            )
            
            if self.verbose:
                print("  ✓ Spectral gating applied")
            
            return reduced_noise
            
        except ImportError:
            print("  ⚠ noisereduce not installed, skipping spectral gating")
            print("     Install with: pip install noisereduce")
            return audio
        except Exception as e:
            print(f"  ⚠ Spectral gating failed: {e}")
            return audio
    
    def apply_deepfilternet(self, audio_path: Path, output_path: Path) -> Path:
        """
        Apply DeepFilterNet for advanced noise suppression
        This is a neural network-based approach
        """
        self._load_deepfilternet()
        
        if not self.use_deepfilternet:
            return audio_path
        
        try:
            from df import enhance, init_df
            
            if self.verbose:
                print("  Applying DeepFilterNet enhancement...")
            
            # DeepFilterNet works on files directly
            model, df_state, suffix = self._df_model
            
            # Enhance the audio
            enhance(
                model,
                df_state,
                str(audio_path),
                str(output_path),
            )
            
            if self.verbose:
                print(f"  ✓ DeepFilterNet enhancement complete")
            
            return output_path
            
        except Exception as e:
            print(f"  ⚠ DeepFilterNet enhancement failed: {e}")
            return audio_path
    
    def normalize_audio(self, audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        """Normalize audio to target dB level"""
        if not self.normalize:
            return audio
        
        # Calculate current RMS
        rms = np.sqrt(np.mean(audio**2))
        if rms == 0:
            return audio
        
        current_db = 20 * np.log10(rms)
        gain_db = target_db - current_db
        gain_linear = 10 ** (gain_db / 20)
        
        normalized = audio * gain_linear
        
        # Prevent clipping
        max_val = np.max(np.abs(normalized))
        if max_val > 1.0:
            normalized = normalized / max_val * 0.95
        
        if self.verbose:
            print(f"  ✓ Normalized: {current_db:.1f}dB → {target_db:.1f}dB")
        
        return normalized
    
    def enhance(self, audio_path: Path, output_path: Path) -> Path:
        """
        Main enhancement pipeline
        Applies all enabled enhancement techniques
        """
        print(f"\n🔧 Enhancing audio: {audio_path.name}")
        
        current_path = audio_path
        
        # Step 1: DeepFilterNet (if enabled)
        if self.use_deepfilternet:
            df_output = output_path.parent / f"{audio_path.stem}_df.wav"
            result = self.apply_deepfilternet(current_path, df_output)
            if result != current_path:
                current_path = result
        
        # Step 2: Load audio for processing
        audio, sr = self.load_audio(current_path)
        
        # Step 3: Spectral gating (if enabled)
        if self.use_spectral_gating:
            audio = self.apply_spectral_gating(audio, sr)
        
        # Step 4: Normalization
        audio = self.normalize_audio(audio)
        
        # Step 5: Save final result
        self.save_audio(audio, output_path, sr)
        
        print(f"✓ Enhancement complete: {output_path}")
        return output_path
    
    def enhance_quick(self, audio_path: Path, output_path: Path) -> Path:
        """
        Quick enhancement using only spectral gating (faster, no ML models)
        Good for testing or when DeepFilterNet is not available
        """
        print(f"\n🔧 Quick enhancement: {audio_path.name}")
        
        audio, sr = self.load_audio(audio_path)
        
        # Apply spectral gating
        if self.use_spectral_gating:
            audio = self.apply_spectral_gating(audio, sr)
        
        # Normalize
        audio = self.normalize_audio(audio)
        
        # Save
        self.save_audio(audio, output_path, sr)
        
        print(f"✓ Quick enhancement complete: {output_path}")
        return output_path


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
            '-ar', '44100',
            '-ac', '2',
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


if __name__ == "__main__":
    # Test the enhancer
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import TEMP_DIR, OUTPUT_DIR, ENHANCEMENT
    
    # Create test audio (1 second of noise + tone)
    test_audio = np.random.randn(16000) * 0.1
    test_audio[4000:12000] += np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, 8000)) * 0.5
    
    test_path = TEMP_DIR / "test_input.wav"
    output_path = OUTPUT_DIR / "test_enhanced.wav"
    
    sf.write(test_path, test_audio, 16000)
    print(f"Created test audio: {test_path}")
    
    # Test enhancement
    enhancer = AudioEnhancer(
        use_deepfilternet=ENHANCEMENT["use_deepfilternet"],
        use_spectral_gating=ENHANCEMENT["use_spectral_gating"],
        noise_reduction_strength=ENHANCEMENT["noise_reduction_strength"],
        normalize=ENHANCEMENT["normalize"],
        verbose=True
    )
    
    enhancer.enhance(test_path, output_path)
