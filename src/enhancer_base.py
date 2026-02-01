"""
Enhancer Base Class
===================

Abstract base class that defines the interface all audio enhancers must implement.
This ensures consistency across different enhancement approaches and enables
easy addition of new enhancers.

Usage:
    from src.enhancer_base import BaseEnhancer

    class MyEnhancer(BaseEnhancer):
        @property
        def name(self) -> str:
            return "MyEnhancer"

        def enhance(self, input_path, output_path) -> Path:
            # Implementation here
            return output_path
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class BaseEnhancer(ABC):
    """
    Abstract base class for all audio enhancers.

    All enhancers in this project should inherit from this class to ensure
    a consistent interface. This enables:
    - Easy swapping of enhancers in the pipeline
    - Consistent error handling
    - Future agent sessions can add new enhancers following this contract

    Attributes:
        noise_reduction_strength: How aggressively to reduce noise (0.0-1.0)
        use_gpu: Whether to use GPU acceleration if available
        verbose: Whether to print progress messages
    """

    def __init__(self,
                 noise_reduction_strength: float = 0.8,
                 use_gpu: bool = False,
                 verbose: bool = True):
        """
        Initialize the enhancer.

        Args:
            noise_reduction_strength: Noise reduction intensity (0.0-1.0).
                Higher values = more aggressive noise removal.
            use_gpu: Use GPU acceleration if available.
            verbose: Print progress messages during enhancement.
        """
        self.noise_reduction_strength = noise_reduction_strength
        self.use_gpu = use_gpu
        self.verbose = verbose

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Human-readable name of this enhancer.

        Returns:
            Name string (e.g., "TorchEnhancer", "DeepFilterNet")
        """
        pass

    @abstractmethod
    def enhance(self, input_path: Path, output_path: Path) -> Path:
        """
        Enhance an audio file.

        This is the main method that performs audio enhancement.
        Implementations should:
        1. Load audio from input_path
        2. Apply enhancement processing
        3. Save enhanced audio to output_path
        4. Return the output_path

        Args:
            input_path: Path to input audio file (WAV, MP3, etc.)
            output_path: Path where enhanced audio should be saved

        Returns:
            Path to the enhanced audio file (same as output_path)

        Raises:
            FileNotFoundError: If input_path doesn't exist
            RuntimeError: If enhancement fails
        """
        pass

    def validate_input(self, input_path: Path) -> None:
        """
        Validate that input file exists and is readable.

        Args:
            input_path: Path to validate

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not a supported format
        """
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        supported_formats = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
        if input_path.suffix.lower() not in supported_formats:
            raise ValueError(
                f"Unsupported audio format: {input_path.suffix}. "
                f"Supported: {supported_formats}"
            )

    def log(self, message: str) -> None:
        """
        Print a message if verbose mode is enabled.

        Args:
            message: Message to print
        """
        if self.verbose:
            print(f"  [{self.name}] {message}")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"noise_reduction_strength={self.noise_reduction_strength}, "
            f"use_gpu={self.use_gpu}, "
            f"verbose={self.verbose})"
        )
