#!/usr/bin/env python3
"""
Smart audio chunking utilities for long-form audio files.
Designed to create 5-10 second chunks with intelligent splitting.
"""

import numpy as np
import librosa
import soundfile as sf
from typing import List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import webrtcvad
import struct
from tqdm import tqdm


@dataclass
class AudioChunk:
    """Represents a chunk of audio with metadata"""
    audio: np.ndarray
    start_time: float
    end_time: float
    sample_rate: int
    chunk_id: str
    
    @property
    def duration(self):
        return self.end_time - self.start_time


def segment_audio_vad(
    audio: np.ndarray,
    sr: int = 16000,
    target_duration: float = 7.5,  # Target 7.5s (middle of 5-10s range)
    min_duration: float = 5.0,
    max_duration: float = 10.0,
    vad_aggressiveness: int = 2,
    frame_duration_ms: int = 30,
    min_silence_duration: float = 0.5
) -> List[Tuple[int, int]]:
    """
    Segment audio using Voice Activity Detection to find natural boundaries.
    
    Args:
        audio: Audio signal array
        sr: Sample rate
        target_duration: Target duration for chunks in seconds
        min_duration: Minimum chunk duration
        max_duration: Maximum chunk duration
        vad_aggressiveness: WebRTC VAD aggressiveness (0-3, higher = more aggressive)
        frame_duration_ms: Frame size for VAD in milliseconds
        min_silence_duration: Minimum silence duration to consider as boundary
        
    Returns:
        List of (start_sample, end_sample) tuples
    """
    # Convert to 16-bit PCM for VAD
    if audio.dtype != np.int16:
        audio_int16 = (audio * 32767).astype(np.int16)
    else:
        audio_int16 = audio
    
    # Initialize VAD
    vad = webrtcvad.Vad(vad_aggressiveness)
    
    # Calculate frame size
    frame_size = int(sr * frame_duration_ms / 1000)
    
    # Find speech/silence boundaries
    speech_frames = []
    for i in range(0, len(audio_int16) - frame_size, frame_size):
        frame = audio_int16[i:i + frame_size]
        # Convert to bytes for VAD
        frame_bytes = struct.pack(f'{len(frame)}h', *frame)
        is_speech = vad.is_speech(frame_bytes, sr)
        speech_frames.append((i, is_speech))
    
    # Find silence boundaries (potential split points)
    silence_boundaries = []
    silence_start = None
    min_silence_samples = int(min_silence_duration * sr)
    
    for i, (sample_idx, is_speech) in enumerate(speech_frames):
        if not is_speech:
            if silence_start is None:
                silence_start = sample_idx
        else:
            if silence_start is not None:
                silence_duration = sample_idx - silence_start
                if silence_duration >= min_silence_samples:
                    # Found a significant silence - mark its midpoint as boundary
                    boundary = silence_start + silence_duration // 2
                    silence_boundaries.append(boundary)
                silence_start = None
    
    # Create chunks based on boundaries
    chunks = []
    chunk_start = 0
    target_samples = int(target_duration * sr)
    min_samples = int(min_duration * sr)
    max_samples = int(max_duration * sr)
    
    for i in range(len(audio_int16)):
        current_duration = i - chunk_start
        
        # Check if we should create a chunk
        should_split = False
        
        if current_duration >= max_samples:
            # Must split - exceeded max duration
            should_split = True
        elif current_duration >= target_samples:
            # Look for nearest silence boundary
            future_boundaries = [b for b in silence_boundaries 
                                if b > chunk_start and b <= i + frame_size * 10]
            if future_boundaries:
                # Found a good boundary nearby
                i = future_boundaries[0]
                should_split = True
            elif current_duration >= target_samples * 1.2:
                # No boundary found but significantly over target
                should_split = True
        
        if should_split and current_duration >= min_samples:
            chunks.append((chunk_start, i))
            chunk_start = i
    
    # Add final chunk if it meets minimum duration
    if len(audio_int16) - chunk_start >= min_samples:
        chunks.append((chunk_start, len(audio_int16)))
    elif chunks:
        # Extend last chunk to include remaining audio
        chunks[-1] = (chunks[-1][0], len(audio_int16))
    
    return chunks


def segment_audio_simple(
    audio: np.ndarray,
    sr: int = 16000,
    target_duration: float = 7.5,
    overlap_ratio: float = 0.1
) -> List[Tuple[int, int]]:
    """
    Simple fixed-duration segmentation with overlap.
    Fallback when VAD is not available or fails.
    
    Args:
        audio: Audio signal array
        sr: Sample rate  
        target_duration: Target duration for chunks in seconds
        overlap_ratio: Overlap between chunks (0.0 to 1.0)
        
    Returns:
        List of (start_sample, end_sample) tuples
    """
    target_samples = int(target_duration * sr)
    step_size = int(target_samples * (1 - overlap_ratio))
    
    chunks = []
    for start in range(0, len(audio) - target_samples + 1, step_size):
        chunks.append((start, start + target_samples))
    
    # Handle remaining audio
    if len(audio) > chunks[-1][1] if chunks else 0:
        # Add final chunk
        start = max(0, len(audio) - target_samples)
        chunks.append((start, len(audio)))
    
    return chunks


def chunk_audio_file(
    input_path: str,
    output_dir: str,
    target_duration: float = 7.5,
    min_duration: float = 5.0,
    max_duration: float = 10.0,
    use_vad: bool = True,
    save_chunks: bool = False
) -> List[AudioChunk]:
    """
    Chunk a long audio file into smaller segments.
    
    Args:
        input_path: Path to input audio file
        output_dir: Directory to save chunks (if save_chunks=True)
        target_duration: Target duration for chunks
        min_duration: Minimum chunk duration
        max_duration: Maximum chunk duration
        use_vad: Whether to use VAD for intelligent splitting
        save_chunks: Whether to save chunks to disk
        
    Returns:
        List of AudioChunk objects
    """
    # Load audio
    audio, sr = librosa.load(input_path, sr=16000)
    
    # Get chunks
    try:
        if use_vad:
            chunk_boundaries = segment_audio_vad(
                audio, sr, 
                target_duration=target_duration,
                min_duration=min_duration,
                max_duration=max_duration
            )
        else:
            chunk_boundaries = segment_audio_simple(
                audio, sr,
                target_duration=target_duration
            )
    except Exception as e:
        print(f"VAD failed ({e}), falling back to simple segmentation")
        chunk_boundaries = segment_audio_simple(
            audio, sr,
            target_duration=target_duration
        )
    
    # Create AudioChunk objects
    chunks = []
    base_name = Path(input_path).stem
    
    for i, (start, end) in enumerate(chunk_boundaries):
        chunk_audio = audio[start:end]
        chunk = AudioChunk(
            audio=chunk_audio,
            start_time=start / sr,
            end_time=end / sr,
            sample_rate=sr,
            chunk_id=f"{base_name}_chunk{i:04d}"
        )
        chunks.append(chunk)
        
        # Save chunk if requested
        if save_chunks:
            output_path = Path(output_dir) / f"{chunk.chunk_id}.wav"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_path), chunk_audio, sr)
    
    return chunks


def process_coraal_for_training(
    coraal_dir: str,
    output_dir: str,
    target_duration: float = 7.5,
    use_vad: bool = True,
    max_files: Optional[int] = None
) -> dict:
    """
    Process CORAAL dataset files into training-ready chunks.
    
    Args:
        coraal_dir: Path to CORAAL cache directory
        output_dir: Output directory for processed chunks
        target_duration: Target chunk duration
        use_vad: Whether to use VAD
        max_files: Maximum number of files to process (for testing)
        
    Returns:
        Dictionary with processing statistics
    """
    coraal_path = Path(coraal_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all WAV files
    wav_files = list(coraal_path.glob("*/audio/*.wav"))
    if max_files:
        wav_files = wav_files[:max_files]
    
    print(f"Found {len(wav_files)} CORAAL audio files")
    
    stats = {
        'files_processed': 0,
        'total_chunks': 0,
        'total_duration': 0,
        'chunk_durations': [],
        'errors': []
    }
    
    # Process each file
    for wav_file in tqdm(wav_files, desc="Processing CORAAL files"):
        try:
            # Extract component and speaker info from path
            # Path format: .../coraal/COMPONENT/audio/COMPONENT_speaker_*.wav
            component = wav_file.parent.parent.name
            
            # Create output subdirectory for this component
            component_output = output_path / component
            
            # Chunk the audio file
            chunks = chunk_audio_file(
                str(wav_file),
                str(component_output),
                target_duration=target_duration,
                use_vad=use_vad,
                save_chunks=True
            )
            
            stats['files_processed'] += 1
            stats['total_chunks'] += len(chunks)
            
            for chunk in chunks:
                duration = chunk.duration
                stats['total_duration'] += duration
                stats['chunk_durations'].append(duration)
                
        except Exception as e:
            print(f"Error processing {wav_file}: {e}")
            stats['errors'].append(str(wav_file))
    
    # Calculate statistics
    if stats['chunk_durations']:
        durations = np.array(stats['chunk_durations'])
        stats['avg_chunk_duration'] = float(np.mean(durations))
        stats['median_chunk_duration'] = float(np.median(durations))
        stats['min_chunk_duration'] = float(np.min(durations))
        stats['max_chunk_duration'] = float(np.max(durations))
        
        # Count chunks in target range (5-10 seconds)
        in_range = np.sum((durations >= 5.0) & (durations <= 10.0))
        stats['chunks_in_target_range'] = int(in_range)
        stats['percent_in_range'] = float(in_range / len(durations) * 100)
    
    return stats


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Chunk audio files for training")
    parser.add_argument("--input", type=str, help="Input audio file or directory")
    parser.add_argument("--output", type=str, help="Output directory for chunks")
    parser.add_argument("--coraal_dir", type=str, 
                       default="/home/timholds/.cache/accent_datasets/coraal",
                       help="CORAAL cache directory")
    parser.add_argument("--target_duration", type=float, default=7.5,
                       help="Target chunk duration in seconds")
    parser.add_argument("--no_vad", action="store_true",
                       help="Disable VAD and use simple segmentation")
    parser.add_argument("--max_files", type=int, default=None,
                       help="Maximum files to process (for testing)")
    
    args = parser.parse_args()
    
    if args.input:
        # Process single file
        chunks = chunk_audio_file(
            args.input,
            args.output or "chunks",
            target_duration=args.target_duration,
            use_vad=not args.no_vad,
            save_chunks=True
        )
        print(f"Created {len(chunks)} chunks")
        for chunk in chunks:
            print(f"  {chunk.chunk_id}: {chunk.duration:.2f}s")
    else:
        # Process CORAAL dataset
        stats = process_coraal_for_training(
            args.coraal_dir,
            args.output or "coraal_chunks",
            target_duration=args.target_duration,
            use_vad=not args.no_vad,
            max_files=args.max_files
        )
        
        print("\n" + "="*60)
        print("CORAAL CHUNKING RESULTS")
        print("="*60)
        print(f"Files processed: {stats['files_processed']}")
        print(f"Total chunks created: {stats['total_chunks']}")
        print(f"Total audio duration: {stats['total_duration']:.2f}s")
        
        if stats.get('avg_chunk_duration'):
            print(f"\nChunk duration statistics:")
            print(f"  Average: {stats['avg_chunk_duration']:.2f}s")
            print(f"  Median: {stats['median_chunk_duration']:.2f}s")
            print(f"  Min: {stats['min_chunk_duration']:.2f}s")
            print(f"  Max: {stats['max_chunk_duration']:.2f}s")
            print(f"  In target range (5-10s): {stats['chunks_in_target_range']} "
                  f"({stats['percent_in_range']:.1f}%)")
        
        if stats['errors']:
            print(f"\nErrors: {len(stats['errors'])} files failed")
        
        # Save stats
        with open("chunking_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        print("\nStats saved to chunking_stats.json")