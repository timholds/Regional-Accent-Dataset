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
        default=["TIMIT", "CommonVoice", "CORAAL", "SAA", "SBCSAE"],
        choices=["TIMIT", "CommonVoice", "CORAAL", "SAA", "SBCSAE"],
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
    if output_path.exists():
        print(f"Dataset already exists at {output_path}, overwriting...")
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
    
    # Apply chunking to ALL audio files
    chunked_samples = chunk_all_audio_samples(
        all_samples,
        chunk_duration=args.chunk_duration,
        chunk_overlap=args.chunk_overlap,
        min_duration=args.min_chunk_duration,
        max_duration=args.max_chunk_duration
    )
    print(f"✓ {len(chunked_samples)} samples after chunking (all samples have chunk metadata)")
    
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
        # Now that chunk_start_sample and chunk_end_sample are part of UnifiedSample,
        # to_dict() will include them automatically
        return pd.DataFrame([s.to_dict() for s in samples])
    
    # Use custom splitting that preserves chunk metadata
    df_with_chunks = samples_to_df_with_chunks(filtered_samples)
    
    # Stratified speaker-based split by dataset AND region
    print("Using stratified splitting by dataset AND region...")
    
    # Group speakers by dataset and region
    df_with_chunks['dataset_region'] = df_with_chunks['dataset_name'] + '_' + df_with_chunks['region_label']
    speaker_groups = df_with_chunks.groupby('speaker_id')['dataset_region'].first().reset_index()
    
    train_speakers = []
    val_speakers = []
    test_speakers = []
    
    # Track which speakers need sample-level splitting
    speakers_needing_sample_split = []
    
    # For each unique dataset-region combination, split speakers
    for dataset_region in speaker_groups['dataset_region'].unique():
        group_speakers = speaker_groups[speaker_groups['dataset_region'] == dataset_region]['speaker_id'].values
        np.random.shuffle(group_speakers)
        
        n_group = len(group_speakers)
        if n_group == 1:
            # If only one speaker, we'll need to split their samples
            speakers_needing_sample_split.extend(group_speakers)
        elif n_group == 2:
            # If two speakers, put one in train and one in val
            train_speakers.extend(group_speakers[:1])
            val_speakers.extend(group_speakers[1:])
        else:
            # Normal split for groups with 3+ speakers
            train_end = int(n_group * (1 - args.val_ratio - args.test_ratio))
            val_end = int(n_group * (1 - args.test_ratio))
            
            # Ensure at least one sample in val and test if possible
            train_end = max(1, train_end)
            if n_group >= 3:
                val_end = max(train_end + 1, val_end)
            
            train_speakers.extend(group_speakers[:train_end])
            val_speakers.extend(group_speakers[train_end:val_end])
            test_speakers.extend(group_speakers[val_end:])
    
    # Handle speakers that need sample-level splitting (single speakers in dataset-region groups)
    if speakers_needing_sample_split:
        print(f"  Speakers needing sample-level split: {len(speakers_needing_sample_split)}")
        
        # Get all samples for these speakers
        sample_split_df = df_with_chunks[df_with_chunks['speaker_id'].isin(speakers_needing_sample_split)]
        
        # For each speaker, split their samples
        train_dfs = []
        val_dfs = []
        test_dfs = []
        
        for speaker in speakers_needing_sample_split:
            speaker_samples = sample_split_df[sample_split_df['speaker_id'] == speaker]
            n_samples = len(speaker_samples)
            
            if n_samples >= 3:
                # Split samples for this speaker
                indices = np.arange(n_samples)
                np.random.shuffle(indices)
                
                train_end = int(n_samples * (1 - args.val_ratio - args.test_ratio))
                val_end = int(n_samples * (1 - args.test_ratio))
                
                # Ensure at least one sample in each split if possible
                train_end = max(1, train_end)
                val_end = max(train_end + 1, val_end) if n_samples > 2 else n_samples
                
                train_dfs.append(speaker_samples.iloc[indices[:train_end]])
                val_dfs.append(speaker_samples.iloc[indices[train_end:val_end]])
                if val_end < n_samples:
                    test_dfs.append(speaker_samples.iloc[indices[val_end:]])
            elif n_samples == 2:
                # Put one in train, one in val
                train_dfs.append(speaker_samples.iloc[:1])
                val_dfs.append(speaker_samples.iloc[1:])
            else:
                # Only one sample, put in train
                train_dfs.append(speaker_samples)
        
        # Combine the sample-level splits with speaker-level splits
        train_df_speaker = df_with_chunks[df_with_chunks['speaker_id'].isin(train_speakers)]
        val_df_speaker = df_with_chunks[df_with_chunks['speaker_id'].isin(val_speakers)]
        test_df_speaker = df_with_chunks[df_with_chunks['speaker_id'].isin(test_speakers)]
        
        # Concatenate all dataframes
        train_dfs_all = [train_df_speaker] + train_dfs
        val_dfs_all = [val_df_speaker] + val_dfs
        test_dfs_all = [test_df_speaker] + test_dfs
        
        train_df = pd.concat(train_dfs_all, ignore_index=True) if train_dfs_all else pd.DataFrame()
        val_df = pd.concat(val_dfs_all, ignore_index=True) if val_dfs_all else pd.DataFrame()
        test_df = pd.concat(test_dfs_all, ignore_index=True) if test_dfs_all else pd.DataFrame()
    else:
        # No sample-level splitting needed
        train_df = df_with_chunks[df_with_chunks['speaker_id'].isin(train_speakers)]
        val_df = df_with_chunks[df_with_chunks['speaker_id'].isin(val_speakers)]
        test_df = df_with_chunks[df_with_chunks['speaker_id'].isin(test_speakers)]
    
    # Print statistics about the split
    print(f"  Total speakers: {len(speaker_groups)}")
    print(f"  Train-only speakers: {len(train_speakers)}")
    print(f"  Val-only speakers: {len(val_speakers)}")
    print(f"  Test-only speakers: {len(test_speakers)}")
    print(f"  Sample-split speakers: {len(speakers_needing_sample_split)}")
    
    # Convert back to samples (chunk metadata now automatically preserved)
    def df_to_samples_with_chunks(df):
        samples = []
        for _, row in df.iterrows():
            # Filter to only UnifiedSample fields
            sample_dict = {k: v for k, v in row.to_dict().items() 
                          if k in UnifiedSample.__dataclass_fields__}
            # chunk_start_sample and chunk_end_sample are now in __dataclass_fields__
            sample = UnifiedSample(**sample_dict)
            samples.append(sample)
        return samples
    
    train_samples = df_to_samples_with_chunks(train_df)
    val_samples = df_to_samples_with_chunks(val_df)
    test_samples = df_to_samples_with_chunks(test_df)
    
    print(f"✓ Train: {len(train_samples)} samples")
    print(f"✓ Val: {len(val_samples)} samples")
    print(f"✓ Test: {len(test_samples)} samples")
    
    # Verify each dataset appears in all splits
    print("\nDataset distribution across splits:")
    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        dataset_counts = split_df['dataset_name'].value_counts()
        print(f"  {split_name}:")
        for dataset, count in dataset_counts.items():
            print(f"    {dataset}: {count} samples")
    
    # Check if any dataset is missing from val or test
    train_datasets = set(train_df['dataset_name'].unique())
    val_datasets = set(val_df['dataset_name'].unique())
    test_datasets = set(test_df['dataset_name'].unique())
    
    missing_from_val = train_datasets - val_datasets
    missing_from_test = train_datasets - test_datasets
    
    if missing_from_val:
        print(f"\n⚠️  WARNING: Datasets missing from validation set: {missing_from_val}")
    if missing_from_test:
        print(f"⚠️  WARNING: Datasets missing from test set: {missing_from_test}")
    
    # Verify speaker overlap (note: sample-split speakers will appear in multiple splits)
    train_speakers_set = set(s.speaker_id for s in train_samples)
    val_speakers_set = set(s.speaker_id for s in val_samples)
    test_speakers_set = set(s.speaker_id for s in test_samples)
    
    # Check overlap
    train_val_overlap = train_speakers_set & val_speakers_set
    train_test_overlap = train_speakers_set & test_speakers_set
    val_test_overlap = val_speakers_set & test_speakers_set
    
    # These overlaps are OK if they're from sample-split speakers
    sample_split_set = set(speakers_needing_sample_split) if speakers_needing_sample_split else set()
    
    unexpected_train_val = train_val_overlap - sample_split_set
    unexpected_train_test = train_test_overlap - sample_split_set
    unexpected_val_test = val_test_overlap - sample_split_set
    
    if unexpected_train_val:
        print(f"⚠️  WARNING: Unexpected speaker overlap between train and val: {unexpected_train_val}")
    if unexpected_train_test:
        print(f"⚠️  WARNING: Unexpected speaker overlap between train and test: {unexpected_train_test}")
    if unexpected_val_test:
        print(f"⚠️  WARNING: Unexpected speaker overlap between val and test: {unexpected_val_test}")
    
    if train_val_overlap or train_test_overlap or val_test_overlap:
        print(f"✓ Speaker overlap from sample-level splits: {len(train_val_overlap | train_test_overlap | val_test_overlap)} speakers")
    else:
        print("✓ No speaker overlap between splits")
    
    # Calculate and display duration statistics before saving
    def calculate_duration_stats(samples):
        """Calculate duration statistics for a set of samples"""
        total_duration = 0
        durations_by_region = {}
        durations_by_dataset = {}
        
        for s in samples:
            # Calculate actual chunk duration
            if hasattr(s, 'chunk_start_sample') and hasattr(s, 'chunk_end_sample'):
                duration = (s.chunk_end_sample - s.chunk_start_sample) / 16000.0
            elif s.duration:
                duration = s.duration
            else:
                duration = 0
            
            total_duration += duration
            
            # By region
            if s.region_label not in durations_by_region:
                durations_by_region[s.region_label] = 0
            durations_by_region[s.region_label] += duration
            
            # By dataset
            if s.dataset_name not in durations_by_dataset:
                durations_by_dataset[s.dataset_name] = 0
            durations_by_dataset[s.dataset_name] += duration
        
        return total_duration, durations_by_region, durations_by_dataset
    
    # Format duration for display
    def format_duration(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
    
    # Calculate durations
    total_duration, durations_by_region, durations_by_dataset = calculate_duration_stats(filtered_samples)
    
    # Display duration statistics
    print(f"\n📊 Duration Statistics:")
    print(f"   Total audio: {format_duration(total_duration)} ({total_duration/3600:.1f} hours)")
    
    print(f"\n   Duration by region (actual audio time):")
    df_all = pd.DataFrame([s.to_dict() for s in filtered_samples])
    region_sample_counts = df_all['region_label'].value_counts().to_dict()
    
    for region in sorted(durations_by_region.keys()):
        duration = durations_by_region[region]
        samples = region_sample_counts.get(region, 0)
        avg_dur = duration / samples if samples > 0 else 0
        pct = (duration / total_duration * 100) if total_duration > 0 else 0
        print(f"     {region:20s}: {format_duration(duration)} ({pct:5.1f}%) - {samples:6d} chunks, avg {avg_dur:.1f}s/chunk")
    
    print(f"\n   Duration by dataset:")
    dataset_sample_counts = df_all['dataset_name'].value_counts().to_dict()
    for dataset in sorted(durations_by_dataset.keys()):
        duration = durations_by_dataset[dataset]
        samples = dataset_sample_counts.get(dataset, 0)
        pct = (duration / total_duration * 100) if total_duration > 0 else 0
        print(f"     {dataset:20s}: {format_duration(duration)} ({pct:5.1f}%) - {samples:6d} chunks")
    
    # Save dataset
    print(f"\nSaving dataset to {output_path}...")
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert samples to DataFrames (chunk metadata already preserved)
    # We can reuse the DataFrames we already created for splitting
    # No need to convert again since we already have train_df, val_df, test_df with chunk metadata
    
    train_df.to_csv(output_path / "train.csv", index=False)
    val_df.to_csv(output_path / "val.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)
    
    # Recalculate durations for metadata (already calculated above for display)
    # total_duration, durations_by_region, durations_by_dataset already calculated
    train_duration, train_durations_by_region, _ = calculate_duration_stats(train_samples)
    val_duration, val_durations_by_region, _ = calculate_duration_stats(val_samples)
    test_duration, test_durations_by_region, _ = calculate_duration_stats(test_samples)
    
    # region_sample_counts and dataset_sample_counts already calculated above
    
    # Save metadata
    metadata = {
        "dataset_name": args.dataset_name,
        "created_at": datetime.now().isoformat(),
        "chunking_enabled": True,
        "chunking_params": {
            "chunk_duration": args.chunk_duration,
            "chunk_overlap": args.chunk_overlap,
            "min_chunk_duration": args.min_chunk_duration,
            "max_chunk_duration": args.max_chunk_duration
        },
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
            "total_duration": {
                "seconds": total_duration,
                "formatted": format_duration(total_duration),
                "train_seconds": train_duration,
                "train_formatted": format_duration(train_duration),
                "val_seconds": val_duration,
                "val_formatted": format_duration(val_duration),
                "test_seconds": test_duration,
                "test_formatted": format_duration(test_duration)
            },
            "duration_by_region": {
                region: {
                    "seconds": duration,
                    "formatted": format_duration(duration),
                    "percentage": (duration / total_duration * 100) if total_duration > 0 else 0,
                    "hours": duration / 3600
                }
                for region, duration in sorted(durations_by_region.items())
            },
            "duration_by_dataset": {
                dataset: {
                    "seconds": duration,
                    "formatted": format_duration(duration),
                    "percentage": (duration / total_duration * 100) if total_duration > 0 else 0,
                    "hours": duration / 3600
                }
                for dataset, duration in sorted(durations_by_dataset.items())
            },
            "samples_by_region": region_sample_counts,
            "samples_by_dataset": dataset_sample_counts,
            "average_duration_by_region": {
                region: durations_by_region[region] / region_sample_counts[region]
                for region in region_sample_counts.keys()
                if region in durations_by_region
            },
            "datasets_included": args.datasets,
            "region_distribution": region_sample_counts  # This is the chunked sample distribution
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