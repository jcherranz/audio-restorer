"""
Speaker Diarization Module
Uses energy-based segmentation and spectral features.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
from scipy import signal
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import timedelta
from .audio_utils import load_mono_audio


@dataclass
class SpeakerSegment:
    speaker: str
    start: float
    end: float
    
    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass 
class DiarizationResult:
    segments: List[SpeakerSegment]
    num_speakers: int
    audio_duration: float
    
    def to_dict(self) -> Dict:
        return {
            "num_speakers": self.num_speakers,
            "audio_duration": self.audio_duration,
            "segments": [
                {"speaker": seg.speaker, "start": round(seg.start, 3),
                 "end": round(seg.end, 3), "duration": round(seg.duration, 3)}
                for seg in self.segments
            ]
        }


class SpeakerDiarizer:
    """Simple speaker diarization using spectral features."""
    
    def __init__(self, min_speakers: int = 1, max_speakers: int = 5, verbose: bool = True):
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.verbose = verbose
        self._last_result: Optional[DiarizationResult] = None
    
    def diarize(self, audio_path: Path, output_json: Optional[Path] = None) -> DiarizationResult:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if self.verbose:
            print(f"\n🎭 Speaker Diarization")
            print(f"   Audio: {audio_path.name}")
        
        # Load audio
        audio, sr = load_mono_audio(audio_path, verbose=self.verbose)
        
        # Resample to 16kHz if needed
        if sr != 16000:
            audio = signal.resample(audio, int(len(audio) * 16000 / sr))
            sr = 16000
        
        audio_duration = len(audio) / sr
        
        if self.verbose:
            print(f"   Duration: {timedelta(seconds=int(audio_duration))}")
            print("   Analyzing speech segments...")
        
        # Segment audio by energy
        segments = self._segment_by_energy(audio, sr)
        
        if len(segments) < 2:
            result = DiarizationResult(
                [SpeakerSegment("SPEAKER_00", 0, audio_duration)],
                1, audio_duration
            )
        else:
            # Extract features and cluster
            features = self._extract_features(audio, sr, segments)
            clustered = self._cluster_segments(features, segments)
            unique_speakers = len(set(s.speaker for s in clustered))
            result = DiarizationResult(clustered, unique_speakers, audio_duration)
        
        self._last_result = result
        
        if self.verbose:
            print(f"   Found {result.num_speakers} speaker(s)")
        
        if output_json:
            output_json = Path(output_json)
            output_json.parent.mkdir(parents=True, exist_ok=True)
            with open(output_json, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            if self.verbose:
                print(f"   Saved to: {output_json}")
        
        return result
    
    def _segment_by_energy(self, audio: np.ndarray, sr: int) -> List[Tuple[float, float]]:
        """Split audio into speech segments using energy."""
        frame_size = int(0.025 * sr)
        hop_size = int(0.01 * sr)
        
        frames = []
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i + frame_size]
            energy = np.sqrt(np.mean(frame**2))
            frames.append(energy)
        
        energy = np.array(frames)
        threshold = np.percentile(energy, 30) * 2
        is_speech = energy > threshold
        
        segments = []
        start = None
        for i, speech in enumerate(is_speech):
            time = i * hop_size / sr
            if speech and start is None:
                start = time
            elif not speech and start is not None:
                if time - start > 0.5:
                    segments.append((start, time))
                start = None
        
        if start is not None:
            segments.append((start, len(audio) / sr))
        
        # Merge close segments
        merged = []
        for seg in segments:
            if merged and seg[0] - merged[-1][1] < 1.5:
                merged[-1] = (merged[-1][0], seg[1])
            else:
                merged.append(seg)
        
        return merged
    
    def _extract_features(self, audio: np.ndarray, sr: int, segments: List[Tuple[float, float]]) -> np.ndarray:
        """Extract spectral features from segments."""
        features = []
        for start, end in segments:
            start_samp = int(start * sr)
            end_samp = int(end * sr)
            segment = audio[start_samp:end_samp]
            
            if len(segment) < int(0.3 * sr):
                segment = np.pad(segment, (0, int(0.3 * sr) - len(segment)))
            
            # Spectral features
            fft = np.fft.rfft(segment)
            magnitude = np.abs(fft)
            freqs = np.fft.rfftfreq(len(segment), 1/sr)
            
            centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-8)
            bandwidth = np.sqrt(np.sum(((freqs - centroid)**2) * magnitude) / (np.sum(magnitude) + 1e-8))
            
            # Energy features
            energy_mean = np.mean(segment**2)
            energy_std = np.std(segment**2)
            
            # Zero crossing rate
            zcr = np.sum(np.diff(np.signbit(segment).astype(int)) != 0) / len(segment)
            
            feat = np.array([centroid, bandwidth, energy_mean, energy_std, zcr])
            features.append(feat / (np.linalg.norm(feat) + 1e-8))
        
        return np.array(features)
    
    def _cluster_segments(self, features: np.ndarray, segments: List[Tuple[float, float]]) -> List[SpeakerSegment]:
        """Cluster segments by speaker."""
        from sklearn.cluster import AgglomerativeClustering
        
        if len(features) < 3:
            return [SpeakerSegment("SPEAKER_00", s, e) for s, e in segments]
        
        max_clusters = min(self.max_speakers, len(features) - 1)
        best_labels = np.zeros(len(features), dtype=int)
        
        for n in range(self.min_speakers, max_clusters + 1):
            try:
                clustering = AgglomerativeClustering(n_clusters=n)
                labels = clustering.fit_predict(features)
                best_labels = labels
            except Exception:
                continue
        
        unique = sorted(set(best_labels))
        label_map = {old: f"SPEAKER_{i:02d}" for i, old in enumerate(unique)}
        
        return [SpeakerSegment(label_map[l], s, e) for (s, e), l in zip(segments, best_labels)]
    
    def get_main_speaker(self) -> Optional[Dict]:
        if not self._last_result or not self._last_result.segments:
            return None
        
        times = {}
        for seg in self._last_result.segments:
            times[seg.speaker] = times.get(seg.speaker, 0) + seg.duration
        
        if not times:
            return None
        
        main = max(times.items(), key=lambda x: x[1])
        total = sum(times.values())
        
        return {
            "speaker": main[0],
            "talk_time_seconds": main[1],
            "percentage": (main[1] / total * 100) if total > 0 else 0
        }
    
    def get_speaker_statistics(self) -> List[Dict]:
        if not self._last_result:
            return []
        
        stats = {}
        for seg in self._last_result.segments:
            if seg.speaker not in stats:
                stats[seg.speaker] = {"segments": 0, "time": 0}
            stats[seg.speaker]["segments"] += 1
            stats[seg.speaker]["time"] += seg.duration
        
        total = sum(s["time"] for s in stats.values())
        result = []
        for spk, s in sorted(stats.items(), key=lambda x: x[1]["time"], reverse=True):
            result.append({
                "speaker": spk,
                "segments": s["segments"],
                "total_time_seconds": round(s["time"], 2),
                "percentage": round(s["time"] / total * 100, 1) if total > 0 else 0
            })
        return result
    
    def print_summary(self):
        if not self._last_result:
            print("No results yet.")
            return
        
        print("\n" + "="*60)
        print("SPEAKER DIARIZATION SUMMARY")
        print("="*60)
        print(f"Speakers found: {self._last_result.num_speakers}")
        
        stats = self.get_speaker_statistics()
        if stats:
            print(f"\n{'Speaker':<12} {'Time':<10} {'%':<8} {'Segments'}")
            print("-" * 40)
            for s in stats:
                time_str = str(timedelta(seconds=int(s["total_time_seconds"])))[2:]
                print(f"{s['speaker']:<12} {time_str:<10} {s['percentage']:<8.1f} {s['segments']}")
        
        main = self.get_main_speaker()
        if main:
            print(f"\nMain Speaker: {main['speaker']} ({main['percentage']:.1f}%)")
        print("="*60)


def quick_diarize(audio_path: Path, output_dir: Optional[Path] = None, verbose: bool = True) -> DiarizationResult:
    diarizer = SpeakerDiarizer(verbose=verbose)
    
    output_json = None
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_json = output_dir / f"{Path(audio_path).stem}_diarization.json"
    
    result = diarizer.diarize(audio_path, output_json)
    if verbose:
        diarizer.print_summary()
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    
    result = quick_diarize(args.audio, args.output, verbose=not args.quiet)
    print(f"\nFound {result.num_speakers} speaker(s)")
