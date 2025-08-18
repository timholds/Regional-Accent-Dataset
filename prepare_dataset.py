#!/usr/bin/env python3
"""
Prepare Unified Dataset for Regional Accent Classification

This script loads, combines, and prepares datasets for training.
It creates train/val/test splits and saves them for later use.
"""

import argparse
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import librosa
import soundfile as sf
from tqdm import tqdm
import logging

from unified_dataset import UnifiedAccentDataset, UnifiedSample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare unified dataset for accent classification"
    )
    
    # Input arguments
    parser.add_argument(
        "--data_root", 
        type=str, 
        default=".",
        help="Root directory containing datasets"
    )
    parser.add_argument(
        "--datasets", 
        nargs="+", 
        default=["TIMIT", "FilteredCommonVoice", "CORAAL", "SAA", "SBCSAE"],
        choices=["TIMIT", "CommonVoice", "FilteredCommonVoice", "CORAAL", "SAA", "SBCSAE"],
        help="Datasets to include (default: ALL available free datasets)"
    )
    
    # Split arguments
    parser.add_argument(
        "--val_ratio", 
        type=float, 
        default=0.15,
        help="Validation set ratio (default: 0.15)"
    )
    parser.add_argument(
        "--test_ratio", 
        type=float, 
        default=0.15,
        help="Test set ratio (default: 0.15)"
    )
    parser.add_argument(
        "--stratify_by", 
        type=str, 
        default="region_label",
        help="Column to stratify splits by (default: region_label)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="prepared_dataset",
        help="Output directory for prepared dataset"
    )
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        default=None,
        help="Name for this dataset configuration (auto-generated if not provided)"
    )
    
    # Chunking arguments
    parser.add_argument(
        "--chunk_duration", 
        type=float, 
        default=7.5,
        help="Target chunk duration in seconds for long audio (default: 7.5)"
    )
    parser.add_argument(
        "--chunk_overlap", 
        type=float, 
        default=2.5,
        help="Overlap between chunks in seconds (default: 2.5)"
    )
    parser.add_argument(
        "--min_chunk_duration", 
        type=float, 
        default=5.0,
        help="Minimum chunk duration in seconds (default: 5.0)"
    )
    parser.add_argument(
        "--max_chunk_duration", 
        type=float, 
        default=10.0,
        help="Maximum duration before chunking (default: 10.0)"
    )
    parser.add_argument(
        "--disable_chunking", 
        action="store_true",
        help="Disable chunking of long audio files (ALL files get chunk metadata regardless)"
    )
    
    # Processing arguments
    parser.add_argument(
        "--max_samples_per_dataset", 
        type=int, 
        default=None,
        help="Maximum samples to use from each dataset (for testing)"
    )
    parser.add_argument(
        "--min_samples_per_speaker", 
        type=int, 
        default=1,
        help="Minimum samples per speaker to include"
    )
    parser.add_argument(
        "--max_samples_per_speaker", 
        type=int, 
        default=100,
        help="Maximum samples per speaker to prevent speaker memorization"
    )
    
    # Additional options
    parser.add_argument(
        "--dry_run", 
        action="store_true",
        help="Show statistics without saving files"
    )
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Overwrite existing dataset"
    )
    
    return parser.parse_args()


def chunk_all_audio_samples(samples, chunk_duration=7.5, chunk_overlap=2.5, 
                            min_duration=5.0, max_duration=10.0):
    """
    Chunk ALL audio samples at dataset preparation time, treating short files as single chunks.
    This ensures consistent chunking metadata for all samples and avoids runtime chunking overhead.
    """
    chunked_samples = []
    target_sr = 16000
    chunk_length = int(chunk_duration * target_sr)
    overlap_length = int(chunk_overlap * target_sr)
    
    logger.info(f"Processing {len(samples)} samples for chunking...")
    
    # Process ALL samples - either chunk long ones or treat short ones as single chunks
    files_to_chunk = []
    for sample in tqdm(samples, desc="Scanning audio durations"):
        # Skip non-audio files
        if not os.path.exists(sample.audio_path) or sample.audio_path.endswith('.csv'):
            # Non-audio files get chunk metadata indicating full file
            sample.chunk_start_sample = 0
            sample.chunk_end_sample = 0  # Will be updated if we know the length
            chunked_samples.append(sample)
            continue
        
        try:
            # Use sample.duration if available to avoid file I/O
            if hasattr(sample, 'duration') and sample.duration is not None:
                duration = sample.duration
            else:
                # Get audio duration without loading the entire file
                with sf.SoundFile(sample.audio_path) as f:
                    orig_sr = f.samplerate
                    orig_length = len(f)
                    duration = orig_length / orig_sr
            
            # ALL files get processed - either chunked or treated as single chunks
            if duration <= max_duration:
                # Short files become single chunks with metadata
                audio_length_samples = int(duration * target_sr)
                sample.chunk_start_sample = 0
                sample.chunk_end_sample = audio_length_samples
                sample.duration = duration  # Ensure duration is set
                chunked_samples.append(sample)
            else:
                files_to_chunk.append((sample, duration))
                
        except Exception as e:
            logger.warning(f"Error checking duration for {sample.sample_id}: {e}")
            # Give it default chunk metadata
            sample.chunk_start_sample = 0
            sample.chunk_end_sample = int(target_sr * 3.0)  # Assume 3 seconds default
            chunked_samples.append(sample)
    
    logger.info(f"Found {len(files_to_chunk)} files needing chunking (>{max_duration}s)")
    
    if len(files_to_chunk) == 0:
        logger.info("No files need chunking!")
        return chunked_samples
    
    logger.info(f"Chunking {len(files_to_chunk)} long audio files into ~{chunk_duration}s chunks...")
    logger.info(f"  Chunk overlap: {chunk_overlap}s")
    logger.info(f"  Min chunk duration: {min_duration}s")
    
    chunks_created = 0
    
    # Second pass: actually chunk the files that need it
    for sample, duration in tqdm(files_to_chunk, desc="Chunking long audio files"):
        try:
            # Load audio for chunking
            audio, sr = librosa.load(sample.audio_path, sr=target_sr)
            audio_length = len(audio)
            
            # Calculate chunk positions
            step = chunk_length - overlap_length
            chunk_count = 0
            
            for start_idx in range(0, audio_length - chunk_length + 1, step):
                end_idx = start_idx + chunk_length
                chunk_duration_actual = (end_idx - start_idx) / target_sr
                
                # Create a new sample for this chunk
                chunk_sample = UnifiedSample(
                    sample_id=f"{sample.sample_id}_chunk{chunk_count:03d}",
                    dataset_name=sample.dataset_name,
                    speaker_id=sample.speaker_id,
                    audio_path=sample.audio_path,
                    transcript=sample.transcript,
                    region_label=sample.region_label,
                    original_accent_label=sample.original_accent_label,
                    state=sample.state,
                    gender=sample.gender,
                    age=sample.age,
                    native_language=sample.native_language,
                    duration=chunk_duration_actual,
                    sample_rate=target_sr,
                    is_validated=sample.is_validated,
                    quality_score=sample.quality_score
                )
                
                # Store chunk boundaries as metadata
                chunk_sample.chunk_start_sample = start_idx
                chunk_sample.chunk_end_sample = end_idx
                
                chunked_samples.append(chunk_sample)
                chunk_count += 1
                chunks_created += 1
            
            # Handle the last chunk if there's significant remaining audio
            remaining = audio_length - (chunk_count * step)
            if remaining >= int(min_duration * target_sr):
                start_idx = max(0, audio_length - chunk_length)
                end_idx = audio_length
                chunk_duration_actual = (end_idx - start_idx) / target_sr
                
                chunk_sample = UnifiedSample(
                    sample_id=f"{sample.sample_id}_chunk{chunk_count:03d}",
                    dataset_name=sample.dataset_name,
                    speaker_id=sample.speaker_id,
                    audio_path=sample.audio_path,
                    transcript=sample.transcript,
                    region_label=sample.region_label,
                    original_accent_label=sample.original_accent_label,
                    state=sample.state,
                    gender=sample.gender,
                    age=sample.age,
                    native_language=sample.native_language,
                    duration=chunk_duration_actual,
                    sample_rate=target_sr,
                    is_validated=sample.is_validated,
                    quality_score=sample.quality_score
                )
                chunk_sample.chunk_start_sample = start_idx
                chunk_sample.chunk_end_sample = end_idx
                chunked_samples.append(chunk_sample)
                chunks_created += 1
                
        except Exception as e:
            logger.warning(f"Error chunking {sample.sample_id}: {e}")
            # Keep the original sample if chunking fails
            chunked_samples.append(sample)
    
    logger.info(f"Chunking complete: {len(samples)} samples -> {len(chunked_samples)} samples")
    logger.info(f"  Files chunked: {len(files_to_chunk)}")
    logger.info(f"  Total chunks created: {chunks_created}")
    return chunked_samples


def filter_samples(samples, args):
    """Apply filtering criteria to samples"""
    if len(samples) == 0:
        return []
        
    df = pd.DataFrame([s.to_dict() for s in samples])
    
    # Filter by minimum samples per speaker
    speaker_counts = df['speaker_id'].value_counts()
    valid_speakers = speaker_counts[speaker_counts >= args.min_samples_per_speaker].index
    df_filtered = df[df['speaker_id'].isin(valid_speakers)]
    
    if len(df_filtered) < len(df):
        print(f"Filtered out {len(df) - len(df_filtered)} samples from speakers with < {args.min_samples_per_speaker} samples")
    
    # Limit maximum samples per speaker to prevent memorization
    if args.max_samples_per_speaker and args.max_samples_per_speaker > 0:
        balanced_samples = []
        speaker_counts_after = df_filtered['speaker_id'].value_counts()
        capped_speakers = 0
        
        for speaker_id in df_filtered['speaker_id'].unique():
            speaker_samples = df_filtered[df_filtered['speaker_id'] == speaker_id]
            original_count = len(speaker_samples)
            
            if original_count > args.max_samples_per_speaker:
                # Randomly sample max_samples_per_speaker from this speaker
                speaker_samples = speaker_samples.sample(n=args.max_samples_per_speaker, random_state=args.seed)
                capped_speakers += 1
                
            balanced_samples.append(speaker_samples)
        
        df_filtered = pd.concat(balanced_samples, ignore_index=True)
        
        if capped_speakers > 0:
            total_capped_samples = sum(max(0, count - args.max_samples_per_speaker) 
                                     for count in speaker_counts_after.values)
            print(f"Capped {capped_speakers} speakers at {args.max_samples_per_speaker} samples each")
            print(f"Removed {total_capped_samples} samples to prevent speaker memorization")
    
    # Convert back to UnifiedSample objects, preserving chunk boundaries
    filtered_samples = []
    for _, row in df_filtered.iterrows():
        # Create sample with only valid UnifiedSample fields
        sample_dict = {k: v for k, v in row.to_dict().items() 
                      if k in UnifiedSample.__dataclass_fields__}
        sample = UnifiedSample(**sample_dict)
        
        # Preserve chunk boundaries if they exist
        if 'chunk_start_sample' in row and pd.notna(row['chunk_start_sample']):
            sample.chunk_start_sample = int(row['chunk_start_sample'])
            sample.chunk_end_sample = int(row['chunk_end_sample'])
        
        filtered_samples.append(sample)
    
    return filtered_samples


def main():
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Generate dataset name if not provided
    if args.dataset_name is None:
        args.dataset_name = "accent_dataset"
    
    # Setup output directory
    output_path = Path(args.output_dir) / args.dataset_name
    
    # Check if dataset already exists
    if output_path.exists() and not args.force:
        print(f"Dataset already exists at {output_path}")
        print("Overwriting existing dataset...")
        # Remove the return statement to allow overwriting
        # return
    
    print("="*60)
    print("PREPARING UNIFIED ACCENT DATASET")
    print("="*60)
    print(f"Datasets to include: {', '.join(args.datasets)}")
    print(f"Output directory: {output_path}")
    print(f"Random seed: {args.seed}")
    print()
    
    # Initialize unified dataset
    unified = UnifiedAccentDataset(args.data_root)
    
    # Load specified datasets
    print("Loading datasets...")
    all_samples = unified.load_all_datasets(args.datasets)
    print(f"✓ Loaded {len(all_samples)} total samples")
    
    # Consolidate Upper Midwest into Midwest
    print("Consolidating regions...")
    for sample in all_samples:
        if sample.region_label == "Upper Midwest":
            sample.region_label = "Midwest"
    
    # Show consolidated region distribution
    region_counts = {}
    for sample in all_samples:
        region = sample.region_label
        region_counts[region] = region_counts.get(region, 0) + 1
    print(f"✓ Consolidated to {len(region_counts)} regions:")
    for region, count in sorted(region_counts.items()):
        print(f"  - {region}: {count} samples")
    
    # Apply chunking to ALL audio files (now enabled by default)
    if not args.disable_chunking:
        chunked_samples = chunk_all_audio_samples(
            all_samples,
            chunk_duration=args.chunk_duration,
            chunk_overlap=args.chunk_overlap,
            min_duration=args.min_chunk_duration,
            max_duration=args.max_chunk_duration
        )
        print(f"✓ {len(chunked_samples)} samples after chunking (all samples have chunk metadata)")
    else:
        # Even when chunking is disabled, add basic chunk metadata
        for sample in all_samples:
            if not hasattr(sample, 'chunk_start_sample'):
                sample.chunk_start_sample = 0
                sample.chunk_end_sample = int(16000 * (sample.duration or 3.0))
        chunked_samples = all_samples
        print("✓ Chunking disabled (but basic chunk metadata added)")
    
    # Apply filtering
    filtered_samples = filter_samples(chunked_samples, args)
    print(f"✓ {len(filtered_samples)} samples after filtering")
    
    # Limit samples per dataset if specified (for testing)
    if args.max_samples_per_dataset:
        limited_samples = []
        for dataset in args.datasets:
            dataset_samples = [s for s in filtered_samples if s.dataset_name == dataset]
            if len(dataset_samples) > args.max_samples_per_dataset:
                dataset_samples = np.random.choice(
                    dataset_samples, 
                    args.max_samples_per_dataset, 
                    replace=False
                ).tolist()
            limited_samples.extend(dataset_samples)
        filtered_samples = limited_samples
        print(f"✓ Limited to {len(filtered_samples)} samples (max {args.max_samples_per_dataset} per dataset)")
    
    # Check for missing datasets and show warning
    samples_by_dataset = {}
    for dataset in args.datasets:
        dataset_samples = [s for s in filtered_samples if s.dataset_name == dataset]
        samples_by_dataset[dataset] = len(dataset_samples)
        if len(dataset_samples) == 0:
            print(f"\n⚠️  WARNING: Dataset '{dataset}' has 0 samples! ⚠️")
            print(f"   This dataset will not contribute to the training data.")
            print(f"   Please check if the dataset is properly loaded or if filtering is too strict.\n")
    
    # Show statistics
    print("\nDataset Statistics:")
    print("-" * 40)
    print("\nSamples per dataset:")
    for dataset, count in samples_by_dataset.items():
        status = " ⚠️ NO SAMPLES!" if count == 0 else ""
        print(f"  {dataset}: {count}{status}")
    
    stats = unified.get_statistics()
    for key, value in stats.items():
        if isinstance(value, dict) and key != "samples_per_dataset":  # Skip this since we show it above
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        elif not isinstance(value, dict):
            print(f"{key}: {value}")
    
    if args.dry_run:
        print("\n✓ Dry run complete. No files were saved.")
        return
    
    # Create train/val/test splits (preserving chunk metadata)
    print(f"\nCreating splits (train: {1-args.val_ratio-args.test_ratio:.0%}, "
          f"val: {args.val_ratio:.0%}, test: {args.test_ratio:.0%})...")
    
    # Create a DataFrame that includes chunk metadata for splitting
    def samples_to_df_with_chunks(samples):
        data = []
        for s in samples:
            d = s.to_dict()
            # Always include chunk boundaries
            if hasattr(s, 'chunk_start_sample'):
                d['chunk_start_sample'] = s.chunk_start_sample
                d['chunk_end_sample'] = s.chunk_end_sample
            else:
                # Fallback - shouldn't happen now but just in case
                d['chunk_start_sample'] = 0
                d['chunk_end_sample'] = int(16000 * (s.duration or 3.0))
            data.append(d)
        return pd.DataFrame(data)
    
    # Use custom splitting that preserves chunk metadata
    df_with_chunks = samples_to_df_with_chunks(filtered_samples)
    
    # Simple speaker-based split that preserves chunk metadata
    speakers = df_with_chunks['speaker_id'].unique()
    np.random.shuffle(speakers)
    
    n_speakers = len(speakers)
    train_end = int(n_speakers * (1 - args.val_ratio - args.test_ratio))
    val_end = int(n_speakers * (1 - args.test_ratio))
    
    train_speakers = speakers[:train_end]
    val_speakers = speakers[train_end:val_end]
    test_speakers = speakers[val_end:]
    
    train_df = df_with_chunks[df_with_chunks['speaker_id'].isin(train_speakers)]
    val_df = df_with_chunks[df_with_chunks['speaker_id'].isin(val_speakers)]
    test_df = df_with_chunks[df_with_chunks['speaker_id'].isin(test_speakers)]
    
    # Convert back to samples (this preserves chunk metadata)
    def df_to_samples_with_chunks(df):
        samples = []
        for _, row in df.iterrows():
            sample_dict = {k: v for k, v in row.to_dict().items() 
                          if k in UnifiedSample.__dataclass_fields__}
            sample = UnifiedSample(**sample_dict)
            # Restore chunk boundaries
            if 'chunk_start_sample' in row and pd.notna(row['chunk_start_sample']):
                sample.chunk_start_sample = int(row['chunk_start_sample'])
                sample.chunk_end_sample = int(row['chunk_end_sample'])
            samples.append(sample)
        return samples
    
    train_samples = df_to_samples_with_chunks(train_df)
    val_samples = df_to_samples_with_chunks(val_df)
    test_samples = df_to_samples_with_chunks(test_df)
    
    print(f"✓ Train: {len(train_samples)} samples")
    print(f"✓ Val: {len(val_samples)} samples")
    print(f"✓ Test: {len(test_samples)} samples")
    
    # Verify no speaker overlap
    train_speakers = set(s.speaker_id for s in train_samples)
    val_speakers = set(s.speaker_id for s in val_samples)
    test_speakers = set(s.speaker_id for s in test_samples)
    
    assert len(train_speakers & val_speakers) == 0, "Speaker overlap between train and val!"
    assert len(train_speakers & test_speakers) == 0, "Speaker overlap between train and test!"
    assert len(val_speakers & test_speakers) == 0, "Speaker overlap between val and test!"
    print("✓ No speaker overlap between splits")
    
    # Save dataset
    print(f"\nSaving dataset to {output_path}...")
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert samples to DataFrames (chunk metadata already preserved)
    # We can reuse the DataFrames we already created for splitting
    # No need to convert again since we already have train_df, val_df, test_df with chunk metadata
    
    train_df.to_csv(output_path / "train.csv", index=False)
    val_df.to_csv(output_path / "val.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)
    
    # Save metadata
    metadata = {
        "dataset_name": args.dataset_name,
        "created_at": datetime.now().isoformat(),
        "chunking_enabled": not args.disable_chunking,
        "chunking_params": {
            "chunk_duration": args.chunk_duration,
            "chunk_overlap": args.chunk_overlap,
            "min_chunk_duration": args.min_chunk_duration,
            "max_chunk_duration": args.max_chunk_duration
        } if not args.disable_chunking else None,
        "args": vars(args),
        "statistics": {
            "total_samples": len(filtered_samples),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "test_samples": len(test_samples),
            "num_speakers": {
                "total": len(set(s.speaker_id for s in filtered_samples)),
                "train": len(train_speakers),
                "val": len(val_speakers),
                "test": len(test_speakers)
            },
            "datasets_included": args.datasets,
            "region_distribution": pd.DataFrame([s.to_dict() for s in filtered_samples])['region_label'].value_counts().to_dict()
        },
        "label_mapping": {
            region: idx for idx, region in enumerate(
                sorted(set(s.region_label for s in filtered_samples))
            )
        }
    }
    
    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save dataset configuration for easy loading
    config = {
        "dataset_path": str(output_path),
        "train_csv": "train.csv",
        "val_csv": "val.csv",
        "test_csv": "test.csv",
        "metadata": "metadata.json",
        "num_labels": len(metadata["label_mapping"]),
        "label_mapping": metadata["label_mapping"],
        "pre_chunked": True,  # Always true now since we always add chunk metadata
        "chunk_info_columns": ["chunk_start_sample", "chunk_end_sample"]  # Always present
    }
    
    with open(output_path / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "="*60)
    print("✓ DATASET PREPARATION COMPLETE!")
    print("="*60)
    print(f"\nDataset saved to: {output_path}")
    print(f"\nTo train a model with this dataset, run:")
    print(f"  python train_accent_classifier.py --dataset_path {output_path}")
    print("\nFiles created:")
    print(f"  - train.csv ({len(train_samples)} samples)")
    print(f"  - val.csv ({len(val_samples)} samples)")
    print(f"  - test.csv ({len(test_samples)} samples)")
    print(f"  - metadata.json (dataset info)")
    print(f"  - config.json (loading config)")


if __name__ == "__main__":
    main()