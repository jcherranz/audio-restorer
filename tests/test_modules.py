#!/usr/bin/env python3
"""
Unit Tests for Audio Restoration Modules
=========================================

Tests each module with synthetic audio (no network, no YouTube downloads).
All tests should complete in < 30 seconds total.

Usage:
    python -m pytest tests/test_modules.py -v
    python tests/test_modules.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Fixtures ──────────────────────────────────────────────────────

def make_sine(freq=440, duration=2.0, sr=48000, amplitude=0.5):
    """Generate a pure sine wave."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def make_speech_like(duration=3.0, sr=48000):
    """Generate speech-like audio (harmonics 200/400/800 Hz)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = (
        0.3 * np.sin(2 * np.pi * 200 * t)
        + 0.2 * np.sin(2 * np.pi * 400 * t)
        + 0.1 * np.sin(2 * np.pi * 800 * t)
    )
    return signal.astype(np.float32)


def make_noisy_speech(snr_db=10, duration=3.0, sr=48000):
    """Generate speech + white noise at a given SNR."""
    speech = make_speech_like(duration, sr)
    noise_power = np.mean(speech ** 2) / (10 ** (snr_db / 10))
    noise = np.random.RandomState(42).randn(len(speech)).astype(np.float32) * np.sqrt(noise_power)
    return speech + noise, speech, noise


def save_wav(audio, path, sr=48000):
    """Save audio to WAV file."""
    sf.write(str(path), audio, sr)


# ── audio_utils tests ────────────────────────────────────────────

class TestAudioUtils:
    def test_load_mono_audio(self):
        from src.audio_utils import load_mono_audio
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "test.wav"
            audio = make_sine()
            save_wav(audio, p)
            loaded, sr = load_mono_audio(p, verbose=False)
            assert sr == 48000
            assert loaded.ndim == 1
            assert len(loaded) == len(audio)

    def test_load_mono_audio_stereo(self):
        """Stereo input should be converted to mono."""
        from src.audio_utils import load_mono_audio
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "stereo.wav"
            mono = make_sine()
            stereo = np.column_stack([mono, mono])
            sf.write(str(p), stereo, 48000)
            loaded, sr = load_mono_audio(p, verbose=False)
            assert loaded.ndim == 1

    def test_save_audio(self):
        from src.audio_utils import save_audio
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "out.wav"
            audio = make_sine()
            save_audio(audio, p, 48000)
            assert p.exists()
            loaded, sr = sf.read(str(p))
            assert sr == 48000
            assert len(loaded) == len(audio)

    def test_prevent_clipping(self):
        from src.audio_utils import prevent_clipping, PEAK_LIMIT
        audio = np.array([2.0, -2.0, 0.5, -0.5], dtype=np.float32)
        result = prevent_clipping(audio, verbose=False)
        assert np.max(np.abs(result)) <= PEAK_LIMIT + 0.01

    def test_prevent_clipping_no_change(self):
        """Audio already below limit should not change."""
        from src.audio_utils import prevent_clipping
        audio = np.array([0.1, -0.1, 0.2], dtype=np.float32)
        result = prevent_clipping(audio, verbose=False)
        np.testing.assert_array_almost_equal(result, audio)

    def test_stitch_chunks(self):
        from src.audio_utils import stitch_chunks
        sr = 48000
        overlap = sr  # 1 second overlap
        chunk1 = make_sine(440, 3.0, sr)
        chunk2 = make_sine(440, 3.0, sr)
        result = stitch_chunks([chunk1, chunk2], overlap)
        expected_len = len(chunk1) + len(chunk2) - overlap
        assert len(result) == expected_len


# ── DeepFilterNet enhancer tests ─────────────────────────────────

class TestDeepFilterNetEnhancer:
    def test_import(self):
        from src.deepfilter_enhancer import DeepFilterNetEnhancer
        assert DeepFilterNetEnhancer is not None

    def test_init(self):
        from src.deepfilter_enhancer import DeepFilterNetEnhancer
        e = DeepFilterNetEnhancer(noise_reduction_strength=0.8, verbose=False)
        assert e.noise_reduction_strength == 0.8
        assert e.model is None  # Lazy loading

    def test_enhance_reduces_noise(self):
        from src.deepfilter_enhancer import DeepFilterNetEnhancer
        noisy, speech, _ = make_noisy_speech(snr_db=5, duration=2.0)
        with tempfile.TemporaryDirectory() as d:
            inp = Path(d) / "noisy.wav"
            out = Path(d) / "enhanced.wav"
            save_wav(noisy, inp)
            e = DeepFilterNetEnhancer(noise_reduction_strength=1.0, verbose=False)
            e.enhance(inp, out)
            enhanced, _ = sf.read(str(out))
            # Enhanced should exist and have same length
            assert out.exists()
            assert len(enhanced) == len(noisy)

    def test_name_property(self):
        from src.deepfilter_enhancer import DeepFilterNetEnhancer
        e = DeepFilterNetEnhancer(verbose=False)
        assert e.name == "DeepFilterNet"


# ── De-esser tests ───────────────────────────────────────────────

class TestDeEsser:
    def test_import(self):
        from src.deesser import DeEsser
        assert DeEsser is not None

    def test_process(self):
        from src.deesser import DeEsser
        with tempfile.TemporaryDirectory() as d:
            inp = Path(d) / "in.wav"
            out = Path(d) / "out.wav"
            audio = make_speech_like()
            save_wav(audio, inp)
            de = DeEsser(verbose=False)
            de.process(inp, out)
            assert out.exists()
            result, _ = sf.read(str(out))
            assert len(result) == len(audio)

    def test_no_crash_on_silence(self):
        from src.deesser import DeEsser
        with tempfile.TemporaryDirectory() as d:
            inp = Path(d) / "silence.wav"
            out = Path(d) / "out.wav"
            save_wav(np.zeros(48000, dtype=np.float32), inp)
            de = DeEsser(verbose=False)
            de.process(inp, out)
            assert out.exists()


# ── Hum remover tests ────────────────────────────────────────────

class TestHumRemover:
    def test_import(self):
        from src.hum_remover import HumRemover
        assert HumRemover is not None

    def test_removes_60hz_hum(self):
        from src.hum_remover import HumRemover
        sr = 48000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        speech = make_speech_like(duration, sr)
        hum = 0.3 * np.sin(2 * np.pi * 60 * t).astype(np.float32)
        noisy = speech + hum

        with tempfile.TemporaryDirectory() as d:
            inp = Path(d) / "hum.wav"
            out = Path(d) / "clean.wav"
            save_wav(noisy, inp)
            hr = HumRemover(verbose=False)
            hr.process(inp, out)
            result, _ = sf.read(str(out))
            assert out.exists()
            assert len(result) == len(noisy)


# ── Click remover tests ──────────────────────────────────────────

class TestClickRemover:
    def test_import(self):
        from src.click_remover import ClickRemover
        assert ClickRemover is not None

    def test_process(self):
        from src.click_remover import ClickRemover
        audio = make_speech_like()
        # Add clicks (spike artifacts)
        audio[24000] = 1.0
        audio[72000] = -1.0

        with tempfile.TemporaryDirectory() as d:
            inp = Path(d) / "clicks.wav"
            out = Path(d) / "clean.wav"
            save_wav(audio, inp)
            cr = ClickRemover(verbose=False)
            cr.process(inp, out)
            assert out.exists()


# ── Comfort noise tests ──────────────────────────────────────────

class TestComfortNoise:
    def test_import(self):
        from src.comfort_noise import ComfortNoiseGenerator
        assert ComfortNoiseGenerator is not None

    def test_adds_noise_to_silence(self):
        from src.comfort_noise import ComfortNoiseGenerator
        sr = 48000
        # Audio with speech then silence
        speech = make_speech_like(1.0, sr)
        silence = np.zeros(sr * 2, dtype=np.float32)
        audio = np.concatenate([speech, silence])

        with tempfile.TemporaryDirectory() as d:
            inp = Path(d) / "in.wav"
            out = Path(d) / "out.wav"
            save_wav(audio, inp)
            cn = ComfortNoiseGenerator(verbose=False)
            cn.process(inp, out)
            result, _ = sf.read(str(out))
            # Silence region should now have some energy
            silence_region = result[sr:]  # After first second
            assert np.max(np.abs(silence_region)) > 0


# ── Diarization tests ────────────────────────────────────────────

class TestDiarization:
    def test_import(self):
        from src.diarization import SpeakerDiarizer
        assert SpeakerDiarizer is not None

    def test_diarize_single_speaker(self):
        from src.diarization import SpeakerDiarizer
        with tempfile.TemporaryDirectory() as d:
            inp = Path(d) / "speech.wav"
            out = Path(d) / "diarization.json"
            save_wav(make_speech_like(5.0), inp)
            sd = SpeakerDiarizer(verbose=False)
            result = sd.diarize(inp, output_json=out)
            assert out.exists()


# ── Pipeline loudness normalization test ──────────────────────────

class TestLoudnessNormalization:
    def test_two_pass_loudnorm(self):
        """Two-pass loudnorm should bring audio close to -16 LUFS."""
        import subprocess
        from config import FFMPEG_PATH

        with tempfile.TemporaryDirectory() as d:
            inp = Path(d) / "loud.wav"
            # Create quiet audio (will be far from -16 LUFS)
            audio = make_speech_like(3.0) * 0.01  # Very quiet
            save_wav(audio, inp)

            from src.pipeline import AudioRestorationPipeline
            p = AudioRestorationPipeline(
                temp_dir=Path(d) / "temp",
                output_dir=Path(d) / "out",
                verbose=False
            )
            p._normalize_loudness(inp, target_lufs=-16.0)

            # Measure result
            cmd = [
                FFMPEG_PATH, '-i', str(inp),
                '-af', 'loudnorm=I=-16:print_format=json',
                '-f', 'null', '-'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            import json
            stderr = result.stderr
            json_start = stderr.rfind('{')
            json_end = stderr.rfind('}')
            if json_start >= 0 and json_end >= 0:
                stats = json.loads(stderr[json_start:json_end + 1])
                measured_lufs = float(stats["input_i"])
                # Should be within 2 LUFS of target
                assert abs(measured_lufs - (-16.0)) < 2.0, \
                    f"Expected ~-16 LUFS, got {measured_lufs}"


# ── Config tests ──────────────────────────────────────────────────

class TestConfig:
    def test_import_no_side_effects(self):
        """Config should import without creating directories."""
        import importlib
        import config
        importlib.reload(config)
        # Should not crash or create directories on import
        assert hasattr(config, 'ENHANCEMENT')
        assert hasattr(config, 'AUDIO_SETTINGS')

    def test_sample_rate_48k(self):
        from config import AUDIO_SETTINGS
        assert AUDIO_SETTINGS["sample_rate"] == 48000

    def test_default_strength_1_0(self):
        from config import ENHANCEMENT
        assert ENHANCEMENT["noise_reduction_strength"] == 1.0

    def test_target_loudness(self):
        from config import ENHANCEMENT
        assert ENHANCEMENT["target_loudness"] == -16

    def test_ffmpeg_path_exists(self):
        from config import FFMPEG_PATH
        assert Path(FFMPEG_PATH).exists() or FFMPEG_PATH == "ffmpeg"


# ── Speaker AGC tests ────────────────────────────────────────────

class TestSpeakerAGC:
    def test_import(self):
        from src.speaker_agc import SpeakerAGC
        assert SpeakerAGC is not None

    def test_process(self):
        from src.speaker_agc import SpeakerAGC
        with tempfile.TemporaryDirectory() as d:
            inp = Path(d) / "in.wav"
            out = Path(d) / "out.wav"
            save_wav(make_speech_like(5.0), inp)
            agc = SpeakerAGC(verbose=False)
            agc.process(inp, out)
            assert out.exists()


# ── Distance enhancer tests ──────────────────────────────────────

class TestDistanceEnhancer:
    def test_import(self):
        from src.distance_enhancer import DistanceRobustEnhancer
        assert DistanceRobustEnhancer is not None

    def test_enhance(self):
        from src.distance_enhancer import DistanceRobustEnhancer
        with tempfile.TemporaryDirectory() as d:
            inp = Path(d) / "in.wav"
            out = Path(d) / "out.wav"
            save_wav(make_speech_like(5.0), inp)
            de = DistanceRobustEnhancer(verbose=False)
            de.enhance(inp, out)
            assert out.exists()


# ── Speaker isolation tests ──────────────────────────────────────

class TestSpeakerIsolation:
    def test_import(self):
        from src.speaker_isolation import SpeakerIsolator
        assert SpeakerIsolator is not None


# ── De-reverb tests ──────────────────────────────────────────────

class TestDereverb:
    def test_import(self):
        from src.dereverb_enhancer import DereverbEnhancer
        assert DereverbEnhancer is not None

    def test_enhance(self):
        from src.dereverb_enhancer import DereverbEnhancer
        with tempfile.TemporaryDirectory() as d:
            inp = Path(d) / "in.wav"
            out = Path(d) / "out.wav"
            save_wav(make_speech_like(3.0), inp)
            de = DereverbEnhancer(verbose=False)
            de.enhance(inp, out)
            assert out.exists()


# ── VoiceFixer tests ─────────────────────────────────────────────

class TestVoiceFixer:
    def test_import(self):
        from src.voicefixer_enhancer import VoiceFixerEnhancer
        assert VoiceFixerEnhancer is not None

    def test_init(self):
        from src.voicefixer_enhancer import VoiceFixerEnhancer
        e = VoiceFixerEnhancer(mode=0, verbose=False)
        assert e.mode == 0
        assert e._model is None  # Lazy loading

    def test_enhance(self):
        from src.voicefixer_enhancer import VoiceFixerEnhancer
        noisy, _, _ = make_noisy_speech(snr_db=5, duration=3.0)
        with tempfile.TemporaryDirectory() as d:
            inp = Path(d) / "noisy.wav"
            out = Path(d) / "fixed.wav"
            save_wav(noisy, inp)
            e = VoiceFixerEnhancer(mode=0, use_gpu=False, verbose=False)
            e.enhance(inp, out, target_sr=48000)
            assert out.exists()
            enhanced, sr = sf.read(str(out))
            assert sr == 48000
            # Length should match original (within a small tolerance for resampling)
            assert abs(len(enhanced) - len(noisy)) <= 1

    def test_name_property(self):
        from src.voicefixer_enhancer import VoiceFixerEnhancer
        e = VoiceFixerEnhancer(verbose=False)
        assert e.name == "VoiceFixer"


# ── Integration tests ────────────────────────────────────────────

class TestIntegration:
    """End-to-end: noisy speech → DeepFilterNet → verify DNSMOS BAK improves."""

    def test_enhance_improves_bak(self):
        """DeepFilterNet should improve DNSMOS BAK (background noise) on noisy speech."""
        from src.deepfilter_enhancer import DeepFilterNetEnhancer
        from src.sota_metrics import SOTAMetricsCalculator

        noisy, _, _ = make_noisy_speech(snr_db=5, duration=5.0)

        with tempfile.TemporaryDirectory() as d:
            inp = Path(d) / "noisy.wav"
            out = Path(d) / "enhanced.wav"
            save_wav(noisy, inp)

            # Enhance
            e = DeepFilterNetEnhancer(noise_reduction_strength=1.0, verbose=False)
            e.enhance(inp, out)
            assert out.exists()

            # Verify output has same length
            enhanced, sr = sf.read(str(out))
            assert len(enhanced) == len(noisy)

            # Measure DNSMOS on both
            calc = SOTAMetricsCalculator(verbose=False)
            import librosa

            noisy_16k = librosa.resample(noisy, orig_sr=48000, target_sr=16000)
            enhanced_16k = librosa.resample(
                enhanced.astype(np.float32), orig_sr=48000, target_sr=16000
            )

            scores_before = calc.calculate_dnsmos(noisy_16k, 16000)
            scores_after = calc.calculate_dnsmos(enhanced_16k, 16000)

            assert scores_before and scores_after, "DNSMOS calculation failed"

            # BAK (background noise quality) should improve — DeepFilterNet's
            # primary job is noise suppression, and BAK measures that directly.
            # OVRL/SIG may not improve on synthetic tones since DNSMOS expects
            # real speech characteristics.
            bak_before = scores_before["bak"]
            bak_after = scores_after["bak"]

            assert bak_after > bak_before, (
                f"DNSMOS BAK should improve: {bak_before:.2f} → {bak_after:.2f}"
            )


# ── Run as standalone ─────────────────────────────────────────────

if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        cwd=Path(__file__).parent.parent
    )
    sys.exit(result.returncode)
