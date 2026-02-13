#!/usr/bin/env python3
"""
Lightweight quality smoke test for CI.

Generates short synthetic speech-like audio and verifies basic quality metrics
are within expected ranges. This is intentionally lightweight and deterministic.
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.measure_quality import measure_all


def make_speech_like(duration: float, sr: int) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = (
        0.3 * np.sin(2 * np.pi * 200 * t)
        + 0.2 * np.sin(2 * np.pi * 400 * t)
        + 0.1 * np.sin(2 * np.pi * 800 * t)
    )
    return signal.astype(np.float32)


def main() -> int:
    sr = 48000
    audio = make_speech_like(3.0, sr)
    temp_path = Path("temp") / "ci_quality_smoke.wav"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(temp_path), audio, sr)

    metrics = measure_all(temp_path)

    # Conservative thresholds to detect broken metrics without flaking.
    if metrics.quality_score < 50.0:
        print(f"Quality score too low: {metrics.quality_score:.2f}")
        return 1
    if metrics.snr_db < 10.0:
        print(f"SNR too low: {metrics.snr_db:.2f}")
        return 1
    if metrics.clarity_score < 0.40:
        print(f"Clarity too low: {metrics.clarity_score:.2f}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
