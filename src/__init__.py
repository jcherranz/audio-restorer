"""
Audio Restoration Pipeline

A Python package for enhancing audio quality in conference recordings.
Uses DeepFilterNet neural denoising and optional post-processing stages.

Main Components:
- pipeline.py: Main orchestration
- enhancers: DeepFilterNet, ML, simple
- post-processing: diarization, speaker isolation, AGC, etc.

Usage:
    from src.pipeline import AudioRestorationPipeline
    pipeline = AudioRestorationPipeline(...)
"""
