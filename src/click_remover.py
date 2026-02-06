"""
Click/Pop Remover Module
========================

Detects and removes transient artifacts (clicks, pops, digital glitches) from
audio recordings.

Common sources of clicks/pops:
- Microphone handling noise
- Electrical interference
- Digital clipping/glitches
- Mouth clicks and lip smacks
- Equipment switching

Implementation uses peak detection to find amplitude spikes that exceed
the local average by a threshold, then interpolates over the detected
click regions.

Usage:
    from src.click_remover import ClickRemover

    remover = ClickRemover()
    remover.process(input_path, output_path)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import binary_dilation, binary_erosion
from typing import List, Tuple
from .audio_utils import load_mono_audio, save_audio, prevent_clipping


class ClickRemover:
    """
    Removes clicks and pops using peak detection and interpolation.

    Detects amplitude spikes that exceed the local RMS by a threshold,
    then replaces them with interpolated values from surrounding audio.
    """

    def __init__(self,
                 threshold_db: float = 15.0,
                 min_click_ms: float = 0.5,
                 max_click_ms: float = 10.0,
                 window_ms: float = 50.0,
                 verbose: bool = True):
        """
        Initialize click remover.

        Args:
            threshold_db: Detection threshold in dB above local RMS.
                Lower = more sensitive (catches more clicks, may affect speech).
                Higher = less sensitive (misses subtle clicks).
            min_click_ms: Minimum click duration to detect (ms).
                Clicks shorter than this are likely noise, not artifacts.
            max_click_ms: Maximum click duration to fix (ms).
                Longer "clicks" may be legitimate audio.
            window_ms: Window size for local RMS calculation (ms).
            verbose: Print progress messages.
        """
        self.threshold_db = threshold_db
        self.min_click_ms = min_click_ms
        self.max_click_ms = max_click_ms
        self.window_ms = window_ms
        self.verbose = verbose

    @property
    def name(self) -> str:
        return "ClickRemover"

    def _calculate_local_rms(self, audio: np.ndarray, sr: int,
                              window_ms: float) -> np.ndarray:
        """
        Calculate local RMS in sliding window.

        Args:
            audio: Audio samples.
            sr: Sample rate.
            window_ms: Window size in milliseconds.

        Returns:
            Array of local RMS values (same length as audio).
        """
        window_samples = max(1, int(window_ms / 1000 * sr))

        # Pad audio for edge handling
        padded = np.pad(audio, (window_samples // 2, window_samples // 2), mode='reflect')

        # Compute squared values
        squared = padded ** 2

        # Sliding window sum using cumsum trick
        cumsum = np.cumsum(squared)
        cumsum = np.insert(cumsum, 0, 0)
        window_sum = cumsum[window_samples:] - cumsum[:-window_samples]

        # RMS = sqrt(mean(squared))
        rms = np.sqrt(window_sum / window_samples)

        # Trim to original length
        return rms[:len(audio)]

    def _detect_clicks(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Detect click regions using peak detection.

        Args:
            audio: Audio samples.
            sr: Sample rate.

        Returns:
            Boolean mask where True indicates a click.
        """
        # Calculate local RMS
        local_rms = self._calculate_local_rms(audio, sr, self.window_ms)

        # Calculate threshold in linear scale
        # Click is detected when |sample| > local_rms * threshold_factor
        threshold_factor = 10 ** (self.threshold_db / 20)

        # Detect samples exceeding threshold
        abs_audio = np.abs(audio)
        click_mask = abs_audio > (local_rms * threshold_factor)

        # Convert time parameters to samples
        min_samples = max(1, int(self.min_click_ms / 1000 * sr))
        max_samples = int(self.max_click_ms / 1000 * sr)

        # Dilate to connect nearby click samples
        click_mask = binary_dilation(click_mask, iterations=min_samples)

        # Find connected regions
        labeled_mask = np.zeros_like(click_mask, dtype=int)
        region_id = 0
        in_region = False
        region_start = 0

        for i in range(len(click_mask)):
            if click_mask[i] and not in_region:
                # Start of new region
                in_region = True
                region_start = i
                region_id += 1
            elif not click_mask[i] and in_region:
                # End of region
                in_region = False
                region_length = i - region_start

                # Only keep regions within size bounds
                if min_samples <= region_length <= max_samples:
                    labeled_mask[region_start:i] = region_id
                else:
                    # Region too short or too long, not a click
                    click_mask[region_start:i] = False

        return click_mask

    def _interpolate_click(self, audio: np.ndarray, start: int, end: int,
                           context_samples: int = 10) -> np.ndarray:
        """
        Replace click region with interpolated audio.

        Uses cubic spline interpolation from surrounding context.

        Args:
            audio: Full audio array.
            start: Start index of click region.
            end: End index of click region.
            context_samples: Number of samples to use from each side.

        Returns:
            Modified audio array with click interpolated.
        """
        # Get context from before and after
        pre_start = max(0, start - context_samples)
        post_end = min(len(audio), end + context_samples)

        # If not enough context, just use linear interpolation
        if start - pre_start < 2 or post_end - end < 2:
            # Simple linear interpolation
            audio[start:end] = np.linspace(
                audio[max(0, start-1)],
                audio[min(len(audio)-1, end)],
                end - start
            )
            return audio

        # Get context samples
        pre_context = audio[pre_start:start]
        post_context = audio[end:post_end]

        # Create x coordinates for interpolation
        pre_x = np.arange(pre_start, start)
        post_x = np.arange(end, post_end)
        interp_x = np.arange(start, end)

        # Combine context
        context_x = np.concatenate([pre_x, post_x])
        context_y = np.concatenate([pre_context, post_context])

        # Use cubic spline interpolation
        try:
            from scipy.interpolate import CubicSpline
            cs = CubicSpline(context_x, context_y)
            audio[start:end] = cs(interp_x)
        except Exception:
            # Fall back to linear interpolation
            audio[start:end] = np.interp(interp_x, context_x, context_y)

        return audio

    def _get_click_regions(self, click_mask: np.ndarray) -> List[Tuple[int, int]]:
        """
        Convert click mask to list of (start, end) regions.

        Args:
            click_mask: Boolean mask.

        Returns:
            List of (start, end) tuples.
        """
        regions = []
        in_region = False
        start = 0

        for i in range(len(click_mask)):
            if click_mask[i] and not in_region:
                in_region = True
                start = i
            elif not click_mask[i] and in_region:
                in_region = False
                regions.append((start, i))

        # Handle region at end
        if in_region:
            regions.append((start, len(click_mask)))

        return regions

    def process(self, input_path: Path, output_path: Path,
                target_sr: int = None) -> Path:
        """
        Remove clicks from audio file.

        Args:
            input_path: Path to input audio file.
            output_path: Path to save processed audio.
            target_sr: Ignored (kept for interface compatibility).

        Returns:
            Path to output file.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if self.verbose:
            print(f"\n🔊 Click/Pop Removal: {input_path.name}")

        # Load audio
        audio, sr = load_mono_audio(input_path, verbose=self.verbose)

        if self.verbose:
            print(f"  Threshold: {self.threshold_db:.1f} dB above local RMS")

        # Detect clicks
        if self.verbose:
            print("  Detecting clicks...")

        click_mask = self._detect_clicks(audio, sr)
        click_regions = self._get_click_regions(click_mask)

        if self.verbose:
            total_click_samples = np.sum(click_mask)
            click_duration_ms = total_click_samples / sr * 1000
            print(f"  Found {len(click_regions)} click(s) ({click_duration_ms:.1f} ms total)")

        # Interpolate over clicks
        if click_regions:
            if self.verbose:
                print("  Removing clicks...")

            # Work on a copy
            output = audio.copy()

            for start, end in click_regions:
                output = self._interpolate_click(output, start, end)

            if self.verbose:
                print(f"  Removed {len(click_regions)} click(s)")
        else:
            output = audio
            if self.verbose:
                print("  No clicks detected")

        output = prevent_clipping(output, verbose=self.verbose)
        save_audio(output, output_path, sr)

        if self.verbose:
            print(f"  ✓ Saved: {output_path}")

        return output_path


def quick_remove_clicks(input_path: Path,
                        output_path: Path = None,
                        threshold_db: float = 15.0,
                        verbose: bool = True) -> Path:
    """
    Quick one-liner for click removal.

    Args:
        input_path: Input audio file.
        output_path: Output file (default: input_declicked.wav).
        threshold_db: Detection threshold.
        verbose: Print progress.

    Returns:
        Path to output file.
    """
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_declicked.wav"

    remover = ClickRemover(threshold_db=threshold_db, verbose=verbose)
    return remover.process(input_path, output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Remove clicks and pops from audio")
    parser.add_argument("input", type=Path, help="Input audio file")
    parser.add_argument("-o", "--output", type=Path, help="Output audio file")
    parser.add_argument("--threshold", type=float, default=15.0,
                        help="Detection threshold in dB (default: 15)")
    parser.add_argument("--min-click", type=float, default=0.5,
                        help="Minimum click duration in ms (default: 0.5)")
    parser.add_argument("--max-click", type=float, default=10.0,
                        help="Maximum click duration in ms (default: 10)")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")

    args = parser.parse_args()

    output = args.output or args.input.parent / f"{args.input.stem}_declicked.wav"

    remover = ClickRemover(
        threshold_db=args.threshold,
        min_click_ms=args.min_click,
        max_click_ms=args.max_click,
        verbose=not args.quiet
    )

    remover.process(args.input, output)
    print(f"\n✅ Click removal complete: {output}")
