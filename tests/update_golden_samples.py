#!/usr/bin/env python3
"""
Update DNSMOS baselines for golden samples.

Usage:
  python tests/update_golden_samples.py
  python tests/update_golden_samples.py --id speech_like_clean
"""

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sota_metrics import SOTAMetricsCalculator


def make_speech_like(duration: float, sr: int) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = (
        0.3 * np.sin(2 * np.pi * 200 * t)
        + 0.2 * np.sin(2 * np.pi * 400 * t)
        + 0.1 * np.sin(2 * np.pi * 800 * t)
    )
    return signal.astype(np.float32)


def make_noisy_speech(snr_db: float, duration: float, sr: int, seed: int) -> np.ndarray:
    speech = make_speech_like(duration, sr)
    noise_power = np.mean(speech ** 2) / (10 ** (snr_db / 10))
    rng = np.random.RandomState(seed)
    noise = rng.randn(len(speech)).astype(np.float32) * np.sqrt(noise_power)
    return speech + noise


def generate_audio(sample: dict, sr: int) -> np.ndarray:
    duration = float(sample.get("duration_s", 3.0))
    audio_type = sample.get("type")

    if audio_type == "speech_like":
        return make_speech_like(duration, sr)

    if audio_type == "speech_noisy":
        snr_db = float(sample.get("snr_db", 10.0))
        seed = int(sample.get("seed", 42))
        return make_noisy_speech(snr_db, duration, sr, seed)

    raise ValueError(f"Unknown golden sample type: {audio_type}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Update golden DNSMOS baselines")
    parser.add_argument("--id", dest="sample_id", help="Update only a specific sample id")
    args = parser.parse_args()

    golden_path = Path(__file__).parent / "golden_samples.json"
    payload = json.loads(golden_path.read_text(encoding="utf-8"))
    samples = payload.get("samples", [])

    if args.sample_id:
        samples = [s for s in samples if s.get("id") == args.sample_id]
        if not samples:
            raise SystemExit(f"Sample id not found: {args.sample_id}")

    calc = SOTAMetricsCalculator(verbose=True)

    for sample in samples:
        audio = generate_audio(sample, sr=48000)
        audio_16k = librosa.resample(audio, orig_sr=48000, target_sr=16000)
        scores = calc.calculate_dnsmos(audio_16k, 16000)
        if not scores:
            raise SystemExit("DNSMOS calculation failed or model unavailable")

        sample["baseline"] = {
            "sig": float(scores["sig"]),
            "bak": float(scores["bak"]),
            "ovrl": float(scores["ovrl"]),
        }
        print(f"{sample['id']}: {sample['baseline']}")

    golden_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
