"""
SOTA Quality Metrics for Speech Enhancement
=============================================
Industry-standard metrics for objective quality assessment.

Metrics implemented:
- PESQ: Perceptual Evaluation of Speech Quality (ITU-T P.862)
- STOI: Short-Time Objective Intelligibility
- DNSMOS: Deep Noise Suppression Mean Opinion Score
- SI-SDR: Scale-Invariant Signal-to-Distortion Ratio

These metrics enable comparison with academic benchmarks and published research.

Usage:
    from src.sota_metrics import SOTAMetricsCalculator

    calc = SOTAMetricsCalculator()
    metrics = calc.calculate(enhanced_path)
    print(f"DNSMOS Overall: {metrics.dnsmos_ovrl:.2f}")

Requirements:
    pip install pesq pystoi onnxruntime
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import soundfile as sf
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class SOTAMetrics:
    """Container for SOTA quality metrics."""
    pesq: Optional[float] = None       # 1.0 - 4.5 (higher = better)
    stoi: Optional[float] = None       # 0.0 - 1.0 (higher = better)
    dnsmos_sig: Optional[float] = None # 1.0 - 5.0 (speech quality)
    dnsmos_bak: Optional[float] = None # 1.0 - 5.0 (background quality)
    dnsmos_ovrl: Optional[float] = None # 1.0 - 5.0 (overall quality)
    si_sdr: Optional[float] = None     # dB (higher = better)

    def to_dict(self) -> Dict[str, Optional[float]]:
        """Convert to dictionary."""
        return {
            'pesq': self.pesq,
            'stoi': self.stoi,
            'dnsmos_sig': self.dnsmos_sig,
            'dnsmos_bak': self.dnsmos_bak,
            'dnsmos_ovrl': self.dnsmos_ovrl,
            'si_sdr': self.si_sdr
        }

    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = ["SOTA Metrics:"]
        if self.dnsmos_ovrl is not None:
            lines.append(f"  DNSMOS Overall: {self.dnsmos_ovrl:.2f} / 5.0")
            lines.append(f"    - Speech (SIG): {self.dnsmos_sig:.2f}")
            lines.append(f"    - Background (BAK): {self.dnsmos_bak:.2f}")
        if self.pesq is not None:
            lines.append(f"  PESQ: {self.pesq:.2f} / 4.5")
        if self.stoi is not None:
            lines.append(f"  STOI: {self.stoi:.3f} / 1.0")
        if self.si_sdr is not None:
            lines.append(f"  SI-SDR: {self.si_sdr:.1f} dB")
        return "\n".join(lines)


class SOTAMetricsCalculator:
    """
    Calculate state-of-the-art speech quality metrics.

    This class provides methods to calculate industry-standard metrics
    used in speech enhancement research:

    - **DNSMOS** (no reference needed): Microsoft's neural MOS predictor
      - SIG: Speech signal quality (1-5)
      - BAK: Background noise quality (1-5)
      - OVRL: Overall quality (1-5)

    - **PESQ** (reference needed): ITU-T P.862 perceptual quality
      - Range: 1.0 - 4.5
      - Correlates well with human perception

    - **STOI** (reference needed): Short-Time Objective Intelligibility
      - Range: 0.0 - 1.0
      - Measures speech intelligibility

    - **SI-SDR** (reference needed): Scale-Invariant Signal-to-Distortion Ratio
      - Range: -inf to +inf dB
      - Measures signal quality improvement

    Usage:
        calc = SOTAMetricsCalculator()

        # Reference-free (DNSMOS only)
        metrics = calc.calculate(enhanced_path)

        # With reference (all metrics)
        metrics = calc.calculate(enhanced_path, reference_path)
    """

    SUPPORTED_SAMPLE_RATES = {
        'pesq': [8000, 16000],
        'stoi': 'any',
        'dnsmos': 16000,
    }

    def __init__(self, verbose: bool = True):
        """
        Initialize the metrics calculator.

        Args:
            verbose: Print progress messages and warnings
        """
        self.verbose = verbose
        self._dnsmos_model = None
        self._model_loaded = False

    def calculate_pesq(self, enhanced: np.ndarray, reference: np.ndarray,
                       sr: int) -> Optional[float]:
        """
        Calculate PESQ score (ITU-T P.862).

        PESQ (Perceptual Evaluation of Speech Quality) is the industry
        standard for measuring voice quality. It compares the enhanced
        audio against a clean reference.

        Args:
            enhanced: Enhanced audio signal (numpy array)
            reference: Clean reference signal (numpy array)
            sr: Sample rate

        Returns:
            PESQ score (1.0 - 4.5) or None if calculation fails
        """
        try:
            from pesq import pesq

            # PESQ only supports 8kHz or 16kHz
            if sr not in [8000, 16000]:
                import librosa
                enhanced = librosa.resample(enhanced, orig_sr=sr, target_sr=16000)
                reference = librosa.resample(reference, orig_sr=sr, target_sr=16000)
                sr = 16000

            mode = 'wb' if sr == 16000 else 'nb'
            score = pesq(sr, reference, enhanced, mode)
            return float(score)
        except ImportError:
            if self.verbose:
                print("  PESQ not available (pip install pesq)")
            return None
        except Exception as e:
            if self.verbose:
                print(f"  PESQ error: {e}")
            return None

    def calculate_stoi(self, enhanced: np.ndarray, reference: np.ndarray,
                       sr: int) -> Optional[float]:
        """
        Calculate STOI score (Short-Time Objective Intelligibility).

        STOI measures how intelligible speech is after processing.
        Higher values indicate better speech intelligibility.

        Args:
            enhanced: Enhanced audio signal (numpy array)
            reference: Clean reference signal (numpy array)
            sr: Sample rate

        Returns:
            STOI score (0.0 - 1.0) or None if calculation fails
        """
        try:
            from pystoi import stoi
            score = stoi(reference, enhanced, sr, extended=False)
            return float(score)
        except ImportError:
            if self.verbose:
                print("  STOI not available (pip install pystoi)")
            return None
        except Exception as e:
            if self.verbose:
                print(f"  STOI error: {e}")
            return None

    def calculate_dnsmos(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Calculate DNSMOS scores (no reference needed).

        Uses Microsoft's DNSMOS P.835 model for predicting Mean Opinion Scores.
        This is a neural network that predicts how humans would rate the audio.

        The model predicts three scores:
        - SIG: Speech signal quality (how clear is the speech?)
        - BAK: Background noise quality (how quiet is the background?)
        - OVRL: Overall quality (combined score)

        For long audio files, the audio is processed in chunks and scores
        are averaged.

        Args:
            audio: Audio signal (numpy array)
            sr: Sample rate

        Returns:
            Dictionary with 'sig', 'bak', 'ovrl' scores (1-5 each)
            Empty dict if calculation fails
        """
        try:
            import onnxruntime as ort

            # Resample to 16kHz if needed (model requirement)
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000

            # Load model (download if needed)
            if self._dnsmos_model is None:
                self._dnsmos_model = self._load_dnsmos_model()

            if self._dnsmos_model is None:
                return {}

            # Prepare audio for model
            # DNSMOS expects float32 audio with reasonable levels
            audio = audio.astype(np.float32)

            # Normalize audio to have reasonable RMS level
            # DNSMOS works best with audio around -26 dBFS
            rms = np.sqrt(np.mean(audio ** 2))
            if rms > 0:
                target_rms = 0.05  # ~-26 dBFS, typical speech level
                audio = audio * (target_rms / rms)

            # Clip to prevent overflow
            audio = np.clip(audio, -1.0, 1.0)

            # DNSMOS model expects fixed length: 144160 samples (9.01s @ 16kHz)
            CHUNK_SIZE = 144160
            CHUNK_DURATION = CHUNK_SIZE / 16000  # ~9 seconds

            # For long files, process in chunks and average
            if len(audio) > CHUNK_SIZE:
                # Sample chunks evenly throughout the audio
                num_chunks = min(10, len(audio) // CHUNK_SIZE)  # Max 10 chunks
                chunk_starts = np.linspace(0, len(audio) - CHUNK_SIZE, num_chunks, dtype=int)

                all_scores = []
                for start in chunk_starts:
                    chunk = audio[start:start + CHUNK_SIZE]
                    chunk = chunk[np.newaxis, :]  # Add batch dimension

                    ort_inputs = {self._dnsmos_model.get_inputs()[0].name: chunk}
                    scores = self._dnsmos_model.run(None, ort_inputs)[0][0]
                    all_scores.append(scores)

                # Average scores across chunks
                avg_scores = np.mean(all_scores, axis=0)
                return {
                    'sig': float(avg_scores[0]),
                    'bak': float(avg_scores[1]),
                    'ovrl': float(avg_scores[2])
                }
            else:
                # Pad short audio to required length
                if len(audio) < CHUNK_SIZE:
                    audio = np.pad(audio, (0, CHUNK_SIZE - len(audio)), mode='constant')

                # Model expects shape (batch, samples)
                audio = audio[np.newaxis, :]

                # Run inference
                ort_inputs = {self._dnsmos_model.get_inputs()[0].name: audio}
                scores = self._dnsmos_model.run(None, ort_inputs)[0][0]

                return {
                    'sig': float(scores[0]),
                    'bak': float(scores[1]),
                    'ovrl': float(scores[2])
                }
        except ImportError:
            if self.verbose:
                print("  DNSMOS requires: pip install onnxruntime")
            return {}
        except Exception as e:
            if self.verbose:
                print(f"  DNSMOS error: {e}")
            return {}

    def _load_dnsmos_model(self):
        """
        Load DNSMOS ONNX model.

        Downloads the model from Microsoft's DNS Challenge repo if not cached.

        Returns:
            ONNX InferenceSession or None if loading fails
        """
        try:
            import onnxruntime as ort
            from urllib.request import urlretrieve

            # Create cache directory
            model_dir = Path.home() / '.cache' / 'audio-restorer' / 'models'
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / 'dnsmos_p835.onnx'

            # Download if not cached
            if not model_path.exists():
                if self.verbose:
                    print("  Downloading DNSMOS model...")
                # Microsoft DNS Challenge DNSMOS model
                url = "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
                try:
                    urlretrieve(url, model_path)
                    if self.verbose:
                        print("  DNSMOS model downloaded")
                except Exception as e:
                    if self.verbose:
                        print(f"  Could not download DNSMOS model: {e}")
                    return None

            # Load the model
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            return ort.InferenceSession(str(model_path), sess_options)

        except Exception as e:
            if self.verbose:
                print(f"  Error loading DNSMOS model: {e}")
            return None

    def calculate_si_sdr(self, enhanced: np.ndarray, reference: np.ndarray) -> float:
        """
        Calculate Scale-Invariant Signal-to-Distortion Ratio.

        SI-SDR measures how well the enhanced signal matches the reference,
        independent of overall scaling. It's commonly used in source
        separation and speech enhancement research.

        Args:
            enhanced: Enhanced audio signal (numpy array)
            reference: Clean reference signal (numpy array)

        Returns:
            SI-SDR in dB (higher is better)
        """
        # Ensure same length
        min_len = min(len(enhanced), len(reference))
        enhanced = enhanced[:min_len]
        reference = reference[:min_len]

        # Remove mean (zero-center)
        reference = reference - np.mean(reference)
        enhanced = enhanced - np.mean(enhanced)

        # SI-SDR formula
        # s_target = <enhanced, reference> * reference / ||reference||^2
        dot = np.sum(enhanced * reference)
        s_target = dot * reference / (np.sum(reference ** 2) + 1e-8)
        e_noise = enhanced - s_target

        # SI-SDR = 10 * log10(||s_target||^2 / ||e_noise||^2)
        si_sdr = 10 * np.log10(np.sum(s_target ** 2) / (np.sum(e_noise ** 2) + 1e-8))
        return float(si_sdr)

    def calculate(self, enhanced_path: Path,
                  reference_path: Optional[Path] = None) -> SOTAMetrics:
        """
        Calculate all available SOTA metrics.

        This is the main entry point for metric calculation. It handles
        file loading, format conversion, and calls individual metric
        calculators.

        Args:
            enhanced_path: Path to enhanced audio file
            reference_path: Path to clean reference audio (optional)
                           If not provided, only DNSMOS will be calculated.

        Returns:
            SOTAMetrics dataclass with all calculated scores
        """
        enhanced_path = Path(enhanced_path)

        if self.verbose:
            print(f"\n  SOTA Metrics: {enhanced_path.name}")

        # Load enhanced audio
        enhanced, sr = sf.read(enhanced_path)

        # Convert to mono if stereo
        if len(enhanced.shape) > 1:
            enhanced = np.mean(enhanced, axis=1)

        if self.verbose:
            duration = len(enhanced) / sr
            print(f"  Duration: {duration:.1f}s at {sr}Hz")

        metrics = SOTAMetrics()

        # Reference-free metrics (DNSMOS)
        if self.verbose:
            print("  Calculating DNSMOS (reference-free)...")
        dnsmos = self.calculate_dnsmos(enhanced, sr)
        if dnsmos:
            metrics.dnsmos_sig = dnsmos.get('sig')
            metrics.dnsmos_bak = dnsmos.get('bak')
            metrics.dnsmos_ovrl = dnsmos.get('ovrl')
            if self.verbose:
                print(f"    SIG={metrics.dnsmos_sig:.2f}, "
                      f"BAK={metrics.dnsmos_bak:.2f}, "
                      f"OVRL={metrics.dnsmos_ovrl:.2f}")

        # Reference-based metrics (need clean reference)
        if reference_path and Path(reference_path).exists():
            if self.verbose:
                print("  Loading reference audio...")

            reference, ref_sr = sf.read(reference_path)

            # Convert to mono if stereo
            if len(reference.shape) > 1:
                reference = np.mean(reference, axis=1)

            # Resample reference to match enhanced if needed
            if ref_sr != sr:
                if self.verbose:
                    print(f"  Resampling reference from {ref_sr}Hz to {sr}Hz...")
                import librosa
                reference = librosa.resample(reference, orig_sr=ref_sr, target_sr=sr)

            # Match lengths (use shorter)
            min_len = min(len(enhanced), len(reference))
            enhanced_matched = enhanced[:min_len]
            reference_matched = reference[:min_len]

            # PESQ
            if self.verbose:
                print("  Calculating PESQ...")
            metrics.pesq = self.calculate_pesq(enhanced_matched, reference_matched, sr)
            if metrics.pesq and self.verbose:
                print(f"    PESQ: {metrics.pesq:.2f}")

            # STOI
            if self.verbose:
                print("  Calculating STOI...")
            metrics.stoi = self.calculate_stoi(enhanced_matched, reference_matched, sr)
            if metrics.stoi and self.verbose:
                print(f"    STOI: {metrics.stoi:.3f}")

            # SI-SDR
            if self.verbose:
                print("  Calculating SI-SDR...")
            metrics.si_sdr = self.calculate_si_sdr(enhanced_matched, reference_matched)
            if self.verbose:
                print(f"    SI-SDR: {metrics.si_sdr:.1f} dB")

        return metrics


if __name__ == "__main__":
    # Test the metrics calculator
    import tempfile

    print("Testing SOTAMetricsCalculator...")

    # Create test audio (simple sine wave)
    sr = 16000
    duration = 3  # seconds
    t = np.linspace(0, duration, sr * duration)

    # Clean reference (speech-like signal)
    reference = np.sin(2 * np.pi * 200 * t) * 0.3
    reference += np.sin(2 * np.pi * 400 * t) * 0.2
    reference += np.sin(2 * np.pi * 800 * t) * 0.1

    # Enhanced (slightly different)
    enhanced = reference + np.random.randn(len(reference)) * 0.02

    with tempfile.TemporaryDirectory() as tmpdir:
        ref_path = Path(tmpdir) / "reference.wav"
        enh_path = Path(tmpdir) / "enhanced.wav"

        sf.write(ref_path, reference.astype(np.float32), sr)
        sf.write(enh_path, enhanced.astype(np.float32), sr)

        calc = SOTAMetricsCalculator(verbose=True)

        # Test reference-free (DNSMOS only)
        print("\n--- Reference-free metrics ---")
        metrics = calc.calculate(enh_path)
        print(metrics)

        # Test with reference
        print("\n--- With reference ---")
        metrics = calc.calculate(enh_path, ref_path)
        print(metrics)

    print("\nSOTAMetricsCalculator test completed!")
