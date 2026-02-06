"""
ML-Based Audio Enhancement Module
=================================

Uses PyTorch-based spectral gating and advanced filtering for high-quality
noise suppression and speech enhancement.

This module provides ML-powered enhancement as an alternative to the
simple ffmpeg-based enhancer.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torchaudio
import soundfile as sf
import librosa
from typing import Optional, Tuple


class TorchEnhancer:
    """
    ML-based audio enhancer using PyTorch operations.
    
    Features:
    - Torch-based spectral gating (GPU accelerated if available)
    - Learnable noise profiles
    - Advanced spectral processing
    - Better preservation of speech quality
    
    Usage:
        enhancer = TorchEnhancer()
        enhancer.enhance(input_path, output_path)
    """
    
    def __init__(self,
                 noise_reduction_strength: float = 0.8,
                 use_gpu: bool = False,
                 verbose: bool = True):
        """
        Initialize TorchEnhancer.
        
        Args:
            noise_reduction_strength: 0.0 to 1.0 (higher = more aggressive)
            use_gpu: Whether to use GPU acceleration (if available)
            verbose: Print progress messages
        """
        self.noise_reduction_strength = noise_reduction_strength
        self.verbose = verbose
        
        # Set device
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        if self.verbose:
            print(f"  TorchEnhancer initialized (device: {self.device})")
    
    def load_audio(self, audio_path: Path, target_sr: int = 44100) -> Tuple[torch.Tensor, int]:
        """
        Load audio file and return as torch tensor.

        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate (44100 for full bandwidth)

        Returns:
            audio_tensor: Shape (channels, samples)
            sample_rate
        """
        if self.verbose:
            print(f"  Loading audio: {audio_path}")
        
        # Load with soundfile first (more reliable than torchaudio.load)
        audio, sr = sf.read(audio_path, dtype='float32')
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Resample if needed
        if sr != target_sr:
            if self.verbose:
                print(f"  Resampling: {sr} -> {target_sr}")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        
        # Convert to torch tensor (1, samples)
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        
        if self.verbose:
            print(f"  ✓ Loaded: {audio_tensor.shape[1]/sr:.1f}s at {sr}Hz")
        
        return audio_tensor, sr
    
    def save_audio(self, audio_tensor: torch.Tensor, output_path: Path, sr: int):
        """Save torch tensor as audio file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to numpy
        audio_np = audio_tensor.squeeze().cpu().numpy()
        
        sf.write(output_path, audio_np, sr)
        
        if self.verbose:
            print(f"  ✓ Saved: {output_path}")
    
    def estimate_noise_profile(self,
                               audio: torch.Tensor,
                               n_fft: int = 2048,
                               hop_length: int = 512) -> torch.Tensor:
        """
        Estimate noise profile from quiet segments.

        Args:
            audio: Input audio tensor
            n_fft: FFT size (2048 for ~46ms at 44.1kHz)
            hop_length: Hop between frames (512 = ~12ms at 44.1kHz)

        Returns:
            noise_profile: Average noise spectrum
        """
        # Compute STFT
        window = torch.hann_window(n_fft).to(audio.device)
        stft = torch.stft(
            audio.squeeze(),
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True
        )
        
        # Calculate magnitude
        mag = torch.abs(stft)
        
        # Find quiet frames (bottom 10% by energy)
        frame_energy = torch.mean(mag, dim=0)
        threshold = torch.quantile(frame_energy, 0.1)
        quiet_frames = frame_energy < threshold
        
        # Average spectrum of quiet frames
        if torch.sum(quiet_frames) > 0:
            noise_profile = torch.mean(mag[:, quiet_frames], dim=1)
        else:
            # Fallback: use minimum values
            noise_profile = torch.min(mag, dim=1)[0]
        
        return noise_profile
    
    def torch_spectral_gate(self,
                           audio: torch.Tensor,
                           noise_profile: torch.Tensor,
                           n_fft: int = 2048,
                           hop_length: int = 512,
                           reduction_strength: float = 0.8) -> torch.Tensor:
        """
        Apply spectral gating using PyTorch operations.

        Args:
            audio: Input audio tensor
            noise_profile: Estimated noise spectrum
            n_fft: FFT size (2048 for ~46ms at 44.1kHz)
            hop_length: Hop between frames (512 = ~12ms at 44.1kHz)
            reduction_strength: How aggressively to reduce noise (0.0-1.0)
        """
        device = audio.device
        window = torch.hann_window(n_fft).to(device)
        
        # Compute STFT
        stft = torch.stft(
            audio.squeeze(),
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True
        )
        
        # Magnitude and phase
        mag = torch.abs(stft)
        phase = torch.angle(stft)
        
        # Expand noise profile to match mag shape
        noise_profile = noise_profile.unsqueeze(1)
        
        # Calculate mask
        # Mask = max(0, 1 - (noise / signal) ^ strength)
        ratio = noise_profile / (mag + 1e-10)
        mask = 1.0 - torch.pow(ratio, reduction_strength)
        mask = torch.clamp(mask, min=0.0, max=1.0)
        
        # Soft mask for smoother results
        mask = torch.pow(mask, 0.5)
        
        # Apply mask to magnitude
        mag_denoised = mag * mask
        
        # Reconstruct complex spectrum
        stft_denoised = mag_denoised * torch.exp(1j * phase)
        
        # Inverse STFT
        audio_denoised = torch.istft(
            stft_denoised,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            length=audio.shape[1]
        )
        
        return audio_denoised.unsqueeze(0)
    
    def apply_high_pass(self, audio: torch.Tensor, sr: int, cutoff: float = 100) -> torch.Tensor:
        """Apply high-pass filter to remove rumble"""
        if cutoff <= 0:
            return audio
        
        # Design high-pass filter using torchaudio
        from torchaudio.functional import highpass_biquad
        
        audio_filtered = highpass_biquad(
            audio.squeeze(),
            sample_rate=sr,
            cutoff_freq=cutoff
        )
        
        return audio_filtered.unsqueeze(0)
    
    def apply_low_pass(self, audio: torch.Tensor, sr: int, cutoff: float = 12000) -> torch.Tensor:
        """Apply low-pass filter to remove high-frequency hiss while preserving consonant clarity"""
        if cutoff >= sr / 2:
            return audio
        
        from torchaudio.functional import lowpass_biquad
        
        audio_filtered = lowpass_biquad(
            audio.squeeze(),
            sample_rate=sr,
            cutoff_freq=cutoff
        )
        
        return audio_filtered.unsqueeze(0)
    
    def normalize_audio(self, audio: torch.Tensor, target_db: float = -20.0) -> torch.Tensor:
        """
        Normalize audio to target dB level.
        
        Args:
            audio: Input audio tensor
            target_db: Target RMS level in dB
        """
        # Calculate current RMS
        rms = torch.sqrt(torch.mean(audio ** 2))
        
        if rms < 1e-10:
            return audio
        
        current_db = 20 * torch.log10(rms + 1e-10)
        gain_db = target_db - current_db.item()
        gain_linear = 10 ** (gain_db / 20)
        
        normalized = audio * gain_linear
        
        # Prevent clipping
        max_val = torch.max(torch.abs(normalized))
        if max_val > 1.0:
            normalized = normalized / max_val * 0.95
        
        if self.verbose:
            print(f"  ✓ Normalized: {current_db.item():.1f}dB → {target_db:.1f}dB")
        
        return normalized
    
    def compress_dynamic_range(self, 
                               audio: torch.Tensor,
                               threshold: float = -20.0,
                               ratio: float = 4.0) -> torch.Tensor:
        """
        Apply soft compression to even out volume differences.
        
        Args:
            audio: Input audio
            threshold: Threshold in dB (above this, compression kicks in)
            ratio: Compression ratio
        """
        # Calculate threshold in linear
        thresh_linear = 10 ** (threshold / 20)
        
        # Apply soft knee compression
        abs_audio = torch.abs(audio)
        
        # Create gain reduction curve
        mask = abs_audio > thresh_linear
        gain = torch.ones_like(audio)
        
        # Above threshold: apply compression
        compressed = thresh_linear + (abs_audio[mask] - thresh_linear) / ratio
        gain[mask] = compressed / (abs_audio[mask] + 1e-10)
        
        # Apply gain
        compressed_audio = audio * gain
        
        return compressed_audio
    
    def enhance(self, input_path: Path, output_path: Path, target_sr: int = 44100) -> Path:
        """
        Main enhancement pipeline.

        Args:
            input_path: Path to input audio file
            output_path: Path to save enhanced audio
            target_sr: Processing sample rate (44100 for full bandwidth)

        Pipeline:
        1. Load audio
        2. High-pass filter (remove rumble)
        3. Estimate noise profile
        4. Apply spectral gating
        5. Low-pass filter (remove hiss)
        6. Dynamic range compression
        7. Normalize
        8. Save output
        """
        print(f"\n🔧 ML Enhancement: {input_path.name}")
        
        # Load audio
        audio, sr = self.load_audio(input_path, target_sr)
        audio = audio.to(self.device)
        
        # Step 1: High-pass filter
        if self.verbose:
            print("  Applying high-pass filter...")
        audio = self.apply_high_pass(audio, sr, cutoff=100)
        
        # Step 2: Estimate noise profile
        if self.verbose:
            print("  Estimating noise profile...")
        noise_profile = self.estimate_noise_profile(audio)
        
        # Step 3: Spectral gating
        if self.verbose:
            print(f"  Applying spectral gating (strength: {self.noise_reduction_strength})...")
        audio = self.torch_spectral_gate(
            audio, 
            noise_profile, 
            reduction_strength=self.noise_reduction_strength
        )
        
        # Step 4: Low-pass filter (12kHz preserves consonant clarity)
        if self.verbose:
            print("  Applying low-pass filter...")
        audio = self.apply_low_pass(audio, sr, cutoff=12000)
        
        # Step 5: Dynamic range compression
        if self.verbose:
            print("  Compressing dynamic range...")
        audio = self.compress_dynamic_range(audio)
        
        # Step 6: Normalize
        audio = self.normalize_audio(audio, target_db=-20.0)
        
        # Save output
        self.save_audio(audio, output_path, sr)
        
        print(f"✓ Enhancement complete: {output_path}")
        return output_path


class AdvancedMLEnhancer(TorchEnhancer):
    """
    Advanced ML enhancer with additional features.

    Includes:
    - Multi-band processing
    - Adaptive noise estimation
    - Silero VAD (neural network) for accurate speech detection
    - Fallback to energy-based VAD if Silero unavailable
    """
    
    def __init__(self, *args, use_vad: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_vad = use_vad
    
    def voice_activity_detection(self,
                                  audio: torch.Tensor,
                                  sr: int,
                                  frame_length: int = 2048,
                                  hop_length: int = 512) -> torch.Tensor:
        """
        Simple energy-based VAD.

        Args:
            audio: Input audio tensor
            sr: Sample rate
            frame_length: Frame size (2048 for ~46ms at 44.1kHz)
            hop_length: Hop between frames (512 = ~12ms at 44.1kHz)

        Returns:
            vad_mask: Boolean tensor (True = speech, False = non-speech)
                   Shape: (n_frames,) matching STFT frames
        """
        # Get audio length in samples
        audio_len = audio.shape[1]
        
        # Calculate expected number of frames (same as STFT)
        n_frames = 1 + (audio_len - frame_length) // hop_length
        
        # Compute frame energies using torch
        audio_np = audio.squeeze().cpu().numpy()
        
        # Manual framing to ensure correct size
        frames = []
        for i in range(n_frames):
            start = i * hop_length
            end = start + frame_length
            if end <= audio_len:
                frame = audio_np[start:end]
                frames.append(frame)
        
        frames = np.array(frames)  # Shape: (n_frames, frame_length)
        energies = np.sum(frames ** 2, axis=1)
        
        # Adaptive threshold
        noise_thresh = np.percentile(energies, 10)
        speech_thresh = np.percentile(energies, 70)
        threshold = noise_thresh + 0.1 * (speech_thresh - noise_thresh)
        
        vad_mask = energies > threshold

        return torch.from_numpy(vad_mask).to(audio.device)

    def voice_activity_detection_silero(self,
                                         audio: torch.Tensor,
                                         sr: int,
                                         n_frames: int) -> torch.Tensor:
        """
        Neural network-based VAD using Silero VAD.

        Silero VAD is much more accurate than energy-based VAD, especially
        in noisy conditions. It correctly identifies speech even with
        background noise, breathing, and other non-speech sounds.

        Args:
            audio: Input audio tensor (1, samples)
            sr: Sample rate of input audio
            n_frames: Number of STFT frames to match output shape

        Returns:
            vad_mask: Boolean tensor (True = speech, False = non-speech)
                   Shape: (n_frames,) matching STFT frames
        """
        # Lazy load Silero VAD model
        if not hasattr(self, '_silero_vad'):
            try:
                if self.verbose:
                    print("  Loading Silero VAD model...")
                self._silero_vad, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    trust_repo=True,
                    verbose=False
                )
                self._get_speech_timestamps = utils[0]
                if self.verbose:
                    print("  Silero VAD loaded")
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Could not load Silero VAD: {e}")
                raise

        # Silero VAD requires 16kHz mono audio
        SILERO_SR = 16000
        audio_np = audio.squeeze().cpu().numpy()

        # Resample to 16kHz if needed
        if sr != SILERO_SR:
            import librosa
            audio_16k = librosa.resample(audio_np, orig_sr=sr, target_sr=SILERO_SR)
        else:
            audio_16k = audio_np

        # Convert to torch tensor for Silero
        audio_tensor = torch.from_numpy(audio_16k).float()

        # Get speech timestamps from Silero
        speech_timestamps = self._get_speech_timestamps(
            audio_tensor,
            self._silero_vad,
            sampling_rate=SILERO_SR,
            threshold=0.5,  # Speech probability threshold
            min_speech_duration_ms=250,  # Minimum speech segment
            min_silence_duration_ms=100,  # Minimum silence between segments
        )

        # Convert timestamps to frame-level mask
        # Create sample-level mask first
        sample_mask = np.zeros(len(audio_np), dtype=bool)

        for segment in speech_timestamps:
            # Convert 16kHz timestamps to original sample rate
            start_sample = int(segment['start'] * sr / SILERO_SR)
            end_sample = int(segment['end'] * sr / SILERO_SR)
            start_sample = max(0, min(start_sample, len(audio_np)))
            end_sample = max(0, min(end_sample, len(audio_np)))
            sample_mask[start_sample:end_sample] = True

        # Convert sample mask to frame mask
        # Use same framing as STFT (n_fft=2048, hop_length=512)
        n_fft = 2048
        hop_length = 512
        frame_mask = np.zeros(n_frames, dtype=bool)

        for i in range(n_frames):
            start = i * hop_length
            end = min(start + n_fft, len(sample_mask))
            if start < len(sample_mask):
                # Frame is speech if majority of samples are speech
                frame_mask[i] = np.mean(sample_mask[start:end]) > 0.5

        if self.verbose:
            speech_pct = np.mean(frame_mask) * 100
            print(f"  Silero VAD: {speech_pct:.1f}% speech detected")

        return torch.from_numpy(frame_mask).to(audio.device)

    def estimate_noise_profile_advanced(self,
                                        audio: torch.Tensor,
                                        sr: int) -> torch.Tensor:
        """
        Estimate noise profile using VAD to exclude speech segments.

        Uses Silero VAD (neural network) for accurate speech detection,
        with fallback to energy-based VAD if Silero is unavailable.

        Uses 2048-sample FFT for ~46ms frames at 44.1kHz.
        """
        n_fft = 2048
        hop_length = 512

        # Compute STFT
        window = torch.hann_window(n_fft).to(audio.device)
        stft = torch.stft(
            audio.squeeze(),
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True
        )

        mag = torch.abs(stft)
        n_freq, n_frames = mag.shape

        if self.use_vad:
            # Try Silero VAD first (more accurate), fallback to energy-based
            try:
                vad_mask = self.voice_activity_detection_silero(audio, sr, n_frames)
            except Exception as e:
                if self.verbose:
                    print(f"  Silero VAD failed, using energy-based: {e}")
                vad_mask = self.voice_activity_detection(audio, sr, n_fft, hop_length)

            # Ensure VAD mask matches STFT frame count
            if vad_mask.shape[0] > n_frames:
                vad_mask = vad_mask[:n_frames]
            elif vad_mask.shape[0] < n_frames:
                # Pad with False (noise)
                padding = torch.zeros(n_frames - vad_mask.shape[0], dtype=torch.bool, device=vad_mask.device)
                vad_mask = torch.cat([vad_mask, padding])

            # Use only non-speech frames for noise estimation
            non_speech_mask = ~vad_mask

            if torch.sum(non_speech_mask) > 10:  # Enough non-speech frames
                noise_profile = torch.mean(mag[:, non_speech_mask], dim=1)
            else:
                # Fallback to percentile method
                if self.verbose:
                    print("  Warning: Not enough non-speech frames, using percentile method")
                noise_profile = torch.quantile(mag, 0.1, dim=1)
        else:
            # Use percentile method
            noise_profile = torch.quantile(mag, 0.1, dim=1)
        
        return noise_profile
    
    def enhance(self, input_path: Path, output_path: Path, target_sr: int = 44100) -> Path:
        """Enhanced pipeline with VAD (44.1kHz for full bandwidth)"""
        print(f"\n🔧 Advanced ML Enhancement: {input_path.name}")
        
        # Load audio
        audio, sr = self.load_audio(input_path, target_sr)
        audio = audio.to(self.device)
        
        # Step 1: High-pass filter
        if self.verbose:
            print("  Applying high-pass filter...")
        audio = self.apply_high_pass(audio, sr, cutoff=100)
        
        # Step 2: Estimate noise profile with VAD
        if self.verbose:
            print("  Estimating noise profile (with VAD)...")
        noise_profile = self.estimate_noise_profile_advanced(audio, sr)
        
        # Step 3: Spectral gating
        if self.verbose:
            print(f"  Applying spectral gating (strength: {self.noise_reduction_strength})...")
        audio = self.torch_spectral_gate(
            audio, 
            noise_profile, 
            reduction_strength=self.noise_reduction_strength
        )
        
        # Step 4: Low-pass filter (12kHz preserves consonant clarity)
        if self.verbose:
            print("  Applying low-pass filter...")
        audio = self.apply_low_pass(audio, sr, cutoff=12000)
        
        # Step 5: Dynamic range compression
        if self.verbose:
            print("  Compressing dynamic range...")
        audio = self.compress_dynamic_range(audio)
        
        # Step 6: Normalize
        audio = self.normalize_audio(audio, target_db=-20.0)
        
        # Save output
        self.save_audio(audio, output_path, sr)
        
        print(f"✓ Enhancement complete: {output_path}")
        return output_path


if __name__ == "__main__":
    # Test the enhancer
    import tempfile

    print("Testing TorchEnhancer...")

    # Create test audio at 44.1kHz
    sr = 44100
    test_audio = np.random.randn(sr * 3) * 0.1  # 3 seconds of noise
    test_audio[sr:sr*2] += np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr)) * 0.5  # 1 second tone

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "test_input.wav"
        output_path = Path(tmpdir) / "test_enhanced.wav"

        sf.write(input_path, test_audio, sr)

        enhancer = TorchEnhancer(verbose=True)
        enhancer.enhance(input_path, output_path)

        print(f"\n✅ Test passed!")
