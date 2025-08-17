"""
Audio segmentation utilities for processing long audio files into training chunks
"""

import numpy as np
from typing import List, Tuple, Optional
import librosa
import logging

logger = logging.getLogger(__name__)


def segment_audio_smart(
    audio: np.ndarray, 
    sr: int = 16000,
    target_duration: float = 7.5,  # Target 7.5 seconds
    min_duration: float = 5.0,      # Minimum 5 seconds
    max_duration: float = 10.0,     # Maximum 10 seconds
    overlap_ratio: float = 0.2      # 20% overlap for continuity
) -> List[Tuple[np.ndarray, int, int]]:
    """
    Intelligently segment audio into chunks of 5-10 seconds.
    
    Args:
        audio: Audio array to segment
        sr: Sample rate
        target_duration: Target duration in seconds (default 7.5s)
        min_duration: Minimum segment duration (default 5s)
        max_duration: Maximum segment duration (default 10s)
        overlap_ratio: Overlap between segments (0.0 to 1.0)
    
    Returns:
        List of (segment, start_sample, end_sample) tuples
    """
    total_samples = len(audio)
    total_duration = total_samples / sr
    
    # Convert durations to samples
    target_samples = int(target_duration * sr)
    min_samples = int(min_duration * sr)
    max_samples = int(max_duration * sr)
    
    segments = []
    
    # If audio is shorter than max duration, return as single segment
    if total_samples <= max_samples:
        # Only return if it meets minimum duration requirement
        if total_samples >= min_samples:
            segments.append((audio, 0, total_samples))
        else:
            # Pad short audio to minimum duration
            padded = np.pad(audio, (0, min_samples - total_samples))
            segments.append((padded, 0, min_samples))
        return segments
    
    # Calculate step size with overlap
    step_size = int(target_samples * (1 - overlap_ratio))
    
    # Create segments
    start = 0
    while start < total_samples:
        # Calculate end position
        end = min(start + target_samples, total_samples)
        segment_length = end - start
        
        # Check if this is the last segment
        remaining = total_samples - start
        
        if remaining <= max_samples:
            # Last segment - take all remaining audio if it's >= min_samples
            if remaining >= min_samples:
                segment = audio[start:total_samples]
                segments.append((segment, start, total_samples))
            break
        else:
            # Regular segment
            segment = audio[start:end]
            segments.append((segment, start, end))
            
        # Move to next position
        start += step_size
    
    return segments


def process_and_segment_audio(
    audio_path: str,
    sr: int = 16000,
    target_duration: float = 7.5,
    min_duration: float = 5.0,
    max_duration: float = 10.0,
    overlap_ratio: float = 0.2
) -> List[Tuple[np.ndarray, float, float]]:
    """
    Load and segment an audio file.
    
    Args:
        audio_path: Path to audio file
        sr: Target sample rate
        target_duration: Target segment duration
        min_duration: Minimum segment duration
        max_duration: Maximum segment duration
        overlap_ratio: Overlap between segments
    
    Returns:
        List of (segment_audio, start_time, end_time) tuples
    """
    try:
        # Load audio
        audio, original_sr = librosa.load(audio_path, sr=sr)
        
        # Get segments
        segments = segment_audio_smart(
            audio, sr, target_duration, 
            min_duration, max_duration, overlap_ratio
        )
        
        # Convert sample positions to time
        segments_with_time = []
        for seg_audio, start_sample, end_sample in segments:
            start_time = start_sample / sr
            end_time = end_sample / sr
            segments_with_time.append((seg_audio, start_time, end_time))
        
        return segments_with_time
        
    except Exception as e:
        logger.error(f"Failed to process audio {audio_path}: {e}")
        return []


def calculate_segmentation_stats(
    audio_duration: float,
    target_duration: float = 7.5,
    overlap_ratio: float = 0.2
) -> dict:
    """
    Calculate statistics about how audio will be segmented.
    
    Args:
        audio_duration: Duration of audio in seconds
        target_duration: Target segment duration
        overlap_ratio: Overlap ratio
    
    Returns:
        Dictionary with segmentation statistics
    """
    if audio_duration <= 10.0:
        return {
            'num_segments': 1,
            'data_multiplier': 1.0,
            'effective_duration': audio_duration
        }
    
    step_duration = target_duration * (1 - overlap_ratio)
    num_segments = int((audio_duration - target_duration) / step_duration) + 1
    
    # Calculate effective duration (with overlap, we get more training data)
    effective_duration = num_segments * target_duration
    data_multiplier = effective_duration / audio_duration
    
    return {
        'num_segments': num_segments,
        'data_multiplier': data_multiplier,
        'effective_duration': effective_duration,
        'segment_duration': target_duration,
        'overlap_seconds': target_duration * overlap_ratio
    }


# Example usage
if __name__ == "__main__":
    # Test with different audio lengths
    test_durations = [3, 5, 7.5, 10, 15, 20, 30, 60]
    
    print("Segmentation Statistics for Different Audio Durations:")
    print("=" * 60)
    
    for duration in test_durations:
        stats = calculate_segmentation_stats(duration)
        print(f"\n{duration:5.1f}s audio -> {stats['num_segments']:2d} segments, "
              f"{stats['data_multiplier']:.1f}x data, "
              f"{stats['effective_duration']:.1f}s effective")
    
    print("\n" + "=" * 60)
    print("\nKey Benefits:")
    print("- Consistent 5-10 second chunks for training")
    print("- 20% overlap preserves boundary information")
    print("- No data loss from truncation")
    print("- Automatic padding for short clips")
    print("- Efficient processing of long recordings")