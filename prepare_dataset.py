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

from unified_dataset import UnifiedAccentDataset


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
        default=["TIMIT"],
        choices=["TIMIT", "CommonVoice", "CORAAL"],
        help="Datasets to include (default: TIMIT)"
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
        default=5,
        help="Minimum samples per speaker to include"
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


def filter_samples(samples, args):
    """Apply filtering criteria to samples"""
    df = pd.DataFrame([s.to_dict() for s in samples])
    
    # Filter by minimum samples per speaker
    speaker_counts = df['speaker_id'].value_counts()
    valid_speakers = speaker_counts[speaker_counts >= args.min_samples_per_speaker].index
    df_filtered = df[df['speaker_id'].isin(valid_speakers)]
    
    if len(df_filtered) < len(df):
        print(f"Filtered out {len(df) - len(df_filtered)} samples from speakers with < {args.min_samples_per_speaker} samples")
    
    # Convert back to UnifiedSample objects
    from unified_dataset import UnifiedSample
    filtered_samples = []
    for _, row in df_filtered.iterrows():
        sample = UnifiedSample(**row.to_dict())
        filtered_samples.append(sample)
    
    return filtered_samples


def main():
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Generate dataset name if not provided
    if args.dataset_name is None:
        datasets_str = "_".join(sorted(args.datasets))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.dataset_name = f"accent_dataset_{datasets_str}_{timestamp}"
    
    # Setup output directory
    output_path = Path(args.output_dir) / args.dataset_name
    
    # Check if dataset already exists
    if output_path.exists() and not args.force:
        print(f"Dataset already exists at {output_path}")
        print("Use --force to overwrite or choose a different --dataset_name")
        return
    
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
    
    # Apply filtering
    filtered_samples = filter_samples(all_samples, args)
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
    
    # Show statistics
    print("\nDataset Statistics:")
    print("-" * 40)
    stats = unified.get_statistics()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
    
    if args.dry_run:
        print("\n✓ Dry run complete. No files were saved.")
        return
    
    # Create train/val/test splits
    print(f"\nCreating splits (train: {1-args.val_ratio-args.test_ratio:.0%}, "
          f"val: {args.val_ratio:.0%}, test: {args.test_ratio:.0%})...")
    
    # Update unified dataset with filtered samples
    unified.all_samples = filtered_samples
    
    train_samples, val_samples, test_samples = unified.create_train_val_test_split(
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        stratify_by=args.stratify_by,
        seed=args.seed
    )
    
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
    
    # Save splits as CSV
    train_df = pd.DataFrame([s.to_dict() for s in train_samples])
    val_df = pd.DataFrame([s.to_dict() for s in val_samples])
    test_df = pd.DataFrame([s.to_dict() for s in test_samples])
    
    train_df.to_csv(output_path / "train.csv", index=False)
    val_df.to_csv(output_path / "val.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)
    
    # Save metadata
    metadata = {
        "dataset_name": args.dataset_name,
        "created_at": datetime.now().isoformat(),
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
        "label_mapping": metadata["label_mapping"]
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