"""
Speaker Isolation Module
Isolates main speaker from conference recordings.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import soundfile as sf
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")


@dataclass
class IsolationResult:
    output_path: Path
    original_duration: float
    isolated_duration: float
    removed_duration: float
    retention_percentage: float
    num_segments_kept: int
    main_speaker: str


class SpeakerIsolator:
    """Isolate main speaker from conference recordings."""
    
    def __init__(self, crossfade_duration: float = 0.05,
                 min_segment_duration: float = 0.3,
                 verbose: bool = True):
        self.crossfade_duration = crossfade_duration
        self.min_segment_duration = min_segment_duration
        self.verbose = verbose
    
    def identify_main_speaker(self, diarization: Dict) -> Tuple[str, List[Dict]]:
        """Identify main speaker from diarization results."""
        segments = diarization.get('segments', [])
        
        if not segments:
            return "SPEAKER_00", []
        
        speaker_times: Dict[str, float] = {}
        speaker_segments: Dict[str, List[Dict]] = {}
        
        for seg in segments:
            speaker = seg['speaker']
            duration = seg.get('duration', seg['end'] - seg['start'])
            
            if duration < self.min_segment_duration:
                continue
            
            speaker_times[speaker] = speaker_times.get(speaker, 0) + duration
            
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append(seg)
        
        if not speaker_times:
            return "SPEAKER_00", segments
        
        main_speaker = max(speaker_times.items(), key=lambda x: x[1])[0]
        return main_speaker, speaker_segments.get(main_speaker, [])
    
    def _apply_crossfade(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply crossfade at start and end."""
        fade_samples = int(self.crossfade_duration * sr)
        
        if len(audio) <= 2 * fade_samples:
            return audio
        
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        
        audio = audio.copy()
        audio[:fade_samples] *= fade_in
        audio[-fade_samples:] *= fade_out
        
        return audio
    
    def isolate_main_speaker(self, input_audio: Path,
                             diarization_json: Path,
                             output_audio: Path) -> IsolationResult:
        """Isolate main speaker using diarization results."""
        input_audio = Path(input_audio)
        diarization_json = Path(diarization_json)
        output_audio = Path(output_audio)
        
        if not input_audio.exists():
            raise FileNotFoundError(f"Audio file not found: {input_audio}")
        if not diarization_json.exists():
            raise FileNotFoundError(f"Diarization file not found: {diarization_json}")
        
        if self.verbose:
            print(f"\n🔇 Speaker Isolation")
            print(f"   Input: {input_audio.name}")
        
        # Load audio
        audio, sr = sf.read(str(input_audio), dtype='float32')
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        original_duration = len(audio) / sr
        
        # Load diarization
        with open(diarization_json, 'r') as f:
            diarization = json.load(f)
        
        main_speaker, main_segments = self.identify_main_speaker(diarization)
        
        if not main_segments:
            if self.verbose:
                print("   No main speaker found, keeping all audio")
            sf.write(str(output_audio), audio, sr)
            return IsolationResult(
                output_path=output_audio,
                original_duration=original_duration,
                isolated_duration=original_duration,
                removed_duration=0,
                retention_percentage=100.0,
                num_segments_kept=1,
                main_speaker="NONE"
            )
        
        # Sort segments by time
        main_segments.sort(key=lambda x: x['start'])
        
        # Extract and combine segments
        isolated_segments = []
        for seg in main_segments:
            start_sample = int(seg['start'] * sr)
            end_sample = int(seg['end'] * sr)
            
            # Ensure within bounds
            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)
            
            if end_sample > start_sample:
                segment_audio = audio[start_sample:end_sample]
                # Apply crossfade
                segment_audio = self._apply_crossfade(segment_audio, sr)
                isolated_segments.append(segment_audio)
        
        if not isolated_segments:
            if self.verbose:
                print("   No segments extracted")
            sf.write(str(output_audio), audio, sr)
            return IsolationResult(
                output_path=output_audio,
                original_duration=original_duration,
                isolated_duration=original_duration,
                removed_duration=0,
                retention_percentage=100.0,
                num_segments_kept=0,
                main_speaker=main_speaker
            )
        
        # Concatenate all segments
        isolated_audio = np.concatenate(isolated_segments)
        isolated_duration = len(isolated_audio) / sr
        removed_duration = original_duration - isolated_duration
        retention_percentage = (isolated_duration / original_duration) * 100
        
        # Save output
        output_audio.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_audio), isolated_audio, sr)
        
        if self.verbose:
            print(f"   Main speaker: {main_speaker}")
            print(f"   Segments kept: {len(isolated_segments)}")
            print(f"   Original: {timedelta(seconds=int(original_duration))}")
            print(f"   Isolated: {timedelta(seconds=int(isolated_duration))}")
            print(f"   Removed: {timedelta(seconds=int(removed_duration))}")
            print(f"   Retention: {retention_percentage:.1f}%")
            print(f"   Saved: {output_audio.name}")
        
        return IsolationResult(
            output_path=output_audio,
            original_duration=original_duration,
            isolated_duration=isolated_duration,
            removed_duration=removed_duration,
            retention_percentage=retention_percentage,
            num_segments_kept=len(isolated_segments),
            main_speaker=main_speaker
        )
    
    def isolate_with_diarization(self, input_audio: Path,
                                  output_audio: Path,
                                  diarizer=None) -> IsolationResult:
        """Run diarization then isolate main speaker."""
        from .diarization import SpeakerDiarizer
        
        input_audio = Path(input_audio)
        output_audio = Path(output_audio)
        
        # Create temp diarization file
        temp_dir = output_audio.parent / "temp"
        temp_dir.mkdir(exist_ok=True)
        diarization_json = temp_dir / f"{input_audio.stem}_diarization.json"
        
        # Run diarization
        if diarizer is None:
            diarizer = SpeakerDiarizer(verbose=self.verbose)
        
        diarizer.diarize(input_audio, output_json=diarization_json)
        
        # Isolate main speaker
        return self.isolate_main_speaker(input_audio, diarization_json, output_audio)


def quick_isolate(input_audio: Path, output_audio: Optional[Path] = None,
                  diarization_json: Optional[Path] = None,
                  verbose: bool = True) -> IsolationResult:
    """Quick one-liner to isolate main speaker."""
    input_audio = Path(input_audio)
    
    if output_audio is None:
        output_audio = input_audio.parent / f"{input_audio.stem}_isolated.wav"
    else:
        output_audio = Path(output_audio)
    
    isolator = SpeakerIsolator(verbose=verbose)
    
    if diarization_json and Path(diarization_json).exists():
        return isolator.isolate_main_speaker(input_audio, diarization_json, output_audio)
    else:
        return isolator.isolate_with_diarization(input_audio, output_audio)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Isolate main speaker from audio")
    parser.add_argument("input", type=Path, help="Input audio file")
    parser.add_argument("-o", "--output", type=Path, help="Output audio file")
    parser.add_argument("-d", "--diarization", type=Path, help="Diarization JSON file")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    
    args = parser.parse_args()
    
    result = quick_isolate(
        args.input,
        args.output,
        args.diarization,
        verbose=not args.quiet
    )
    
    print(f"\n✅ Isolation complete!")
    print(f"   Kept {result.retention_percentage:.1f}% of audio ({result.num_segments_kept} segments)")
    print(f"   Main speaker: {result.main_speaker}")
