# US Regional Accent Classifier

Fine-tuning Wav2Vec2 to classify US regional accents using multiple datasets.

## Quick Start - Full Pipeline

```bash
# Prepare unified dataset from all three sources (TIMIT, CommonVoice, CORAAL)
python prepare_dataset.py --datasets TIMIT CommonVoice CORAAL --output_dir prepared_dataset
# Train the model
python train_accent_classifier.py --dataset_path prepared_dataset/accent_dataset
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
regional-accent/
├── prepare_dataset.py          # Prepares unified dataset
├── train_accent_classifier.py  # Training script
├── unified_dataset.py          # Dataset loading/unification
├── region_mappings.py          # State → Region mappings
├── audio_data_analysis.py      # Audio compatibility checks
└── data/                       # Downloaded datasets (after cleanup)
    ├── timit_docs/             # TIMIT documentation files
    ├── TRAIN/                  # Training audio data
    │   ├── DR1/                # Dialect regions DR1-DR8
    │   └── ...                 # Each contains speaker folders
    ├── TEST/                   # Test audio data
    │   ├── DR1/                # Same structure as TRAIN
    │   └── ...
    ├── CommonVoice/
    └── CORAAL/
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

## Model Details

### Key Configuration

Based on the current training script (`train_accent_classifier.py`), here are the key model details:

1. **Model Architecture**: Using `Wav2Vec2ForSequenceClassification` from Hugging Face Transformers
2. **Feature Encoder Freezing**: The CNN feature encoder is frozen via `model.freeze_feature_encoder()` (line 217)
3. **Transformer Layer Freezing**: First 10 of 12 transformer layers are frozen (lines 221-226), keeping only layers 10-11 trainable
4. **Class Weighting**: Implements class weights using `sqrt(max_count/count)` capped at 3.0 for handling imbalanced data (lines 478-499)
5. **Gradient Clipping**: Uses gradient clipping with `max_grad_norm=0.5` (line 289)
6. **Learning Rate**: Default learning rate is 3e-5 with linear warmup scheduler

## Troubleshooting

- **Import errors**: Ensure you're in the project directory and venv is activated
- **Dataset not found**: Check paths in `data/` directory match expected structure
- **Memory errors**: Reduce `batch_size` or use `max_samples_per_dataset`
- **No samples after filtering**: Reduce `min_samples_per_speaker` requirement