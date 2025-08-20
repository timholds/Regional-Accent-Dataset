# US Regional Accent Classifier

Fine-tuning Wav2Vec2 to classify US regional accents using multiple datasets.

## Quick Start - Full Pipeline

```bash
# Prepare unified dataset from all three sources (TIMIT, CommonVoice, CORAAL)
python prepare_dataset.py --datasets TIMIT CommonVoice CORAAL --output_dir prepared_dataset

# Train with optimized settings for 29+ hours of data
python train_accent_classifier.py \
    --dataset_path ./prepared_dataset/accent_dataset \
    --learning_rate 3e-4 \
    --gradient_accumulation_steps 4 \
    --batch_size 8 \
    --epochs 50 \
    --dropout_rate 0.25 \
    --weight_decay 0.01 \
    --class_weight_power 0.7 \
    --unfreeze_last_n_layers 4 \
    --hidden_dim 768 \
    --warmup_ratio 0.15 \
    --desc "Optimized: LR 3e-4, unfreeze 4 layers, hidden=768"
```

## Overview

This project trains a model to identify US regional accents from audio. We combine three major datasets (TIMIT, Mozilla Common Voice, and CORAAL) into a unified training pipeline.

## Quick Start

```bash
# 1. Activate environment
source accent-env/bin/activate

# 2. Prepare dataset
python prepare_dataset.py --datasets TIMIT CommonVoice CORAAL

# 3. Train model
python train_accent_classifier.py --dataset_path prepared_datasets/accent_dataset_*
```

## Detailed Usage

### Dataset Preparation

```bash
# Basic usage with one dataset
python prepare_dataset.py --datasets TIMIT

# Multiple datasets with custom settings
python prepare_dataset.py \
    --datasets TIMIT CommonVoice CORAAL \
    --output_dir prepared_datasets \
    --dataset_name my_accent_dataset \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --min_samples_per_speaker 5 \
    --seed 42

# Testing with limited data
python prepare_dataset.py \
    --datasets TIMIT \
    --max_samples_per_dataset 100 \
    --dry_run  # Preview without saving
```

### Model Training

```bash
# Basic training
python train_accent_classifier.py \
    --dataset_path prepared_datasets/my_accent_dataset \
    --model_name facebook/wav2vec2-base \
    --num_epochs 10 \
    --batch_size 16 \
    --learning_rate 3e-4 \
    --output_dir models/accent_classifier

# With LoRA (efficient fine-tuning)
python train_accent_classifier.py \
    --dataset_path prepared_datasets/my_accent_dataset \
    --use_lora \
    --num_epochs 5 \
    --batch_size 8

# With Weights & Biases logging
python train_accent_classifier.py \
    --dataset_path prepared_datasets/my_accent_dataset \
    --use_wandb \
    --wandb_project accent-classifier
```

## Label Mapping

Each dataset labels accents differently:
- TIMIT uses these old-school dialect codes (dr1, dr2, etc.)
- Common Voice is just whatever people typed in ("southern drawl", "boston", "midwest")  
- CORAAL uses city names (ATL for Atlanta, DET for Detroit)

We map everything to 8 regions:
```
1. New England (Boston, Maine, etc.)
2. New York Metropolitan (NYC area)
3. Mid-Atlantic (Philly, Baltimore, DC)
4. South Atlantic (Virginia down to Florida)
5. Deep South (Alabama, Mississippi, Georgia, etc.)
6. Upper Midwest (Michigan, Wisconsin, Minnesota)
7. Lower Midwest (Ohio, Indiana, Illinois)
8. West (everything past Colorado)
```

The mapping works like:
- TIMIT dr1 → New England (easy)
- Common Voice "boston accent" → New England (pattern matching)
- CORAAL ATL → Atlanta → Georgia → Deep South (location lookup)

We keep the original labels too in case we need to debug why something got mapped weird.

## Project Structure

```
Regional-Accent-Dataset/
├── Core Scripts
│   ├── prepare_dataset.py          # Prepares unified dataset from multiple sources
│   ├── train_accent_classifier.py  # Main training script with optimizations
│   ├── unified_dataset.py          # Dataset orchestrator and unification logic
│   ├── optimized_dataset.py        # Optimized dataset wrapper for training
│   └── region_mappings.py          # State → Region mapping definitions
│
├── loaders/                         # Dataset-specific loaders
│   ├── timit_loader.py             # TIMIT dataset loader
│   └── saa_kaggle_loader.py        # Speech Accent Archive loader (Kaggle version)
│
├── scripts/
│   └── download/                   # Dataset download utilities
│       ├── download_coraal.py      # Download CORAAL components
│       ├── download_real_sbcsae.py # Download SBCSAE data
│       └── download_full_coraal.sh # Batch download script
│
├── Documentation
│   ├── README.md                   # This file
│   ├── DATASETS.md                 # Detailed dataset documentation
│   ├── PROJECT_NOTES.md            # Development notes and optimizations
│   └── CLAUDE.md                   # AI assistant instructions
│
├── prepared_datasets/               # Output from prepare_dataset.py
│   └── accent_dataset_*/           # Timestamped dataset versions
│       ├── train.csv                # Training samples
│       ├── val.csv                  # Validation samples
│       ├── test.csv                 # Test samples
│       ├── metadata.json            # Dataset statistics
│       └── config.json              # Loading configuration
│
└── ~/.cache/accent_datasets/        # Downloaded/cached datasets (in home directory)
    ├── timit/                      # TIMIT audio files
    ├── coraal/                     # CORAAL components  
    ├── saa/                        # Speech Accent Archive
    ├── common_voice/               # Common Voice data
    └── sbcsae/                     # Santa Barbara Corpus
```

Note: Run `cleanup_timit_data.py` after downloading TIMIT to fix the nested directory structure and remove duplicate `.WAV.wav` files.

## Requirements

```bash
source accent-env/bin/activate
pip install -r requirements.txt
```

## Datasets

- **TIMIT**: Classic speech dataset with dialect regions (~5.4 hours, 630 speakers)
- **Mozilla Common Voice**: Crowdsourced with self-reported accents (2,984 hours US English)
- **CORAAL**: Corpus of Regional African American Language (150 hours conversational)

See `docs/DATASETS_DOCUMENTATION.md` for detailed dataset information.

## Output Structure

After dataset preparation:
```
prepared_datasets/my_accent_dataset/
├── train.csv          # Training samples
├── val.csv            # Validation samples  
├── test.csv           # Test samples
├── metadata.json      # Dataset statistics
└── config.json        # Loading configuration
```

## Troubleshooting

- **Import errors**: Ensure you're in the project directory and venv is activated
- **Dataset not found**: Check paths in `data/` directory match expected structure
- **Memory errors**: Reduce `batch_size` or use `max_samples_per_dataset`
- **No samples after filtering**: Reduce `min_samples_per_speaker` requirement