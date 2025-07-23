# US Regional Accent Classifier

Fine-tuning Wav2Vec2 to classify US regional accents using multiple datasets.

## Overview

This project trains a model to identify US regional accents from audio. We combine three major datasets (TIMIT, Mozilla Common Voice, and CORAAL) into a unified training pipeline.

## Quick Start

```bash
# Prepare dataset
python prepare_dataset.py --datasets TIMIT CommonVoice CORAAL

# Train model
python train_accent_classifier.py --dataset_path prepared_datasets/accent_dataset_*
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
└── data/                       # Downloaded datasets
    ├── TIMIT/
    ├── CommonVoice/
    └── CORAAL/
```

## Requirements

```bash
source accent-env/bin/activate
pip install -r requirements.txt
```

## Datasets

- **TIMIT**: Classic speech dataset with dialect regions
- **Mozilla Common Voice**: Crowdsourced with self-reported accents
- **CORAAL**: Corpus of Regional African American Language

See DATASET_PREPARATION_GUIDE.md for detailed setup instructions.