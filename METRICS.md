# Training Metrics Reference Guide

This document explains the key metrics logged during training and their expected ranges for healthy model training.

## Core Performance Metrics

### Accuracy Metrics
- **val/accuracy**: Standard classification accuracy (0-1)
  - Early training: 0.125-0.25 (above random baseline of 0.125 for 8 classes)
  - Mid training: 0.35-0.55
  - Good model: 0.60-0.75
  - Excellent: > 0.75

- **val/top2_accuracy**: Correct class in top 2 predictions
  - Should be 15-25% higher than top-1 accuracy
  - Good: > 0.70
  - If much higher than top-1 (>30% gap): model is confused between specific region pairs

- **val/top3_accuracy**: Correct class in top 3 predictions  
  - Should be 25-35% higher than top-1 accuracy
  - Good: > 0.80
  - Ceiling around 0.90-0.95

### F1 Scores
- **val/macro_f1**: Balanced performance across all classes
  - More reliable than accuracy for imbalanced datasets
  - Should track closely with accuracy (within 5-10%)
  - Large gap (>15%) indicates severe class imbalance issues

## Training Stability Metrics

### Loss Standard Deviation (loss_std)
Measures variance in loss across batches - indicates data quality and labeling consistency.

**Normal ranges:**
- **Epochs 1-5**: 0.3-0.8 (model exploring)
- **Epochs 5-15**: 0.1-0.3 (should decrease)
- **Epochs 15+**: 0.05-0.2 (should stabilize)

**Warning signs:**
- **> 1.0 consistently**: Severe labeling issues or corrupted samples
- **Sudden spikes** (0.2 → 1.5): Batch contains mislabeled/out-of-distribution samples
- **Never below 0.5**: Mixed data quality (e.g., studio vs phone recordings)
- **Increases over time**: Model memorizing, failing on edge cases

### Gradient Norms
Indicates optimization stability and learning dynamics.

**grad_norm (average gradient magnitude):**
- **Normal**: 0.2-0.4 (with clipping at 0.5)
- **Early training**: Can spike to clip value occasionally
- **Concerning**: Consistently at max (0.5) - learning rate too high
- **Critical**: Near zero (<0.01) - learning rate too low or model saturated

**grad_norm_std (variation in gradient norms):**
- **Healthy**: 0.05-0.15 (consistent gradient flow)
- **Concerning**: > 0.3 (unstable optimization)
- **Critical**: > 0.5 (gradient explosions/vanishing)

## Model Confidence & Uncertainty

### Average Entropy (avg_entropy)
Measures model's prediction confidence (lower = more confident).

- **Early training**: 1.5-2.0 (high uncertainty)
- **Mid training**: 0.8-1.5 (gaining confidence)
- **Well-trained**: 0.3-0.8 (confident predictions)
- **Overfit**: < 0.2 (overconfident, poor generalization)

### High Entropy Samples
The model logs the top 10 most uncertain predictions. These reveal:
- Ambiguous accent boundaries (e.g., Mid-Atlantic/South border)
- Poor quality audio samples
- Mislabeled data points

## Data Quality Indicators

### Speaker Consistency
Percentage of speakers where all their samples get the same prediction.

**Expected values (with proper train/test splits):**
- **Random baseline**: ~12.5% (1/8 for 8 classes)
- **Good generalization**: 40-60%
- **Reasonable**: 60-75%
- **Suspicious**: > 85%

**Why high consistency (>85%) is problematic:**
- Indicates model learned speaker→region mapping instead of accent→region
- Even with no train/test speaker overlap, suggests memorizing speaker-specific patterns
- Should see variation as speakers code-switch or change prosody

### Per-Dataset Performance
Tracks accuracy for each data source (TIMIT, CommonVoice, CORAAL, etc.).

**Expected patterns:**
- **TIMIT**: Often highest (clean, controlled recordings)
- **CommonVoice**: More variable (crowdsourced, varied quality)
- **CORAAL**: May be lower (conversational, natural speech)

**Warning signs:**
- One dataset >20% below others: Quality or labeling issues
- All datasets similar but low: Model capacity issue
- High train, low val on specific dataset: Overfitting to that dataset's characteristics

### Duration Performance Buckets
Performance by audio clip length.

**Expected patterns:**
- **0-5s**: Lowest accuracy (insufficient accent cues)
- **5-10s**: Moderate improvement
- **10-15s**: Best performance
- **15+s**: Similar or slightly better than 10-15s

**Warning signs:**
- Short clips same as long: Model using non-accent features
- Performance decreases with length: Model can't handle longer context
- Huge jumps between buckets: Inconsistent preprocessing

## Confusion Analysis

### Top Confusion Pairs
Region pairs with >10% confusion rate reveal:

**Expected confusions (linguistically similar):**
- Mid-Atlantic ↔ South Atlantic (geographic neighbors)
- Upper Midwest ↔ Lower Midwest (similar dialect features)
- Deep South ↔ South Atlantic (overlapping regions)

**Unexpected confusions (indicates problems):**
- New England ↔ Deep South (very different)
- West ↔ New York (distinct accents)

## Early Stopping Indicators

### Learning Curve Metrics
- **learning/improvement_rate**: Should be positive but decreasing
  - Healthy: Starts at 0.02-0.05, decreases smoothly
  - Plateaued: < 0.001 for multiple epochs
  - Overfitting: Becomes negative

### When to Stop Training
Stop when ALL of these occur:
1. val/loss increases for 3+ epochs
2. val/accuracy plateaus (< 0.5% change) for 5+ epochs
3. train/loss_std < 0.1 but val/loss_std > 0.5 (overfitting)

## Diagnostic Patterns

### Healthy Training at Epoch 10
```
train/loss_std: 0.15
train/grad_norm: 0.35
train/grad_norm_std: 0.08
val/speaker_consistency: 0.65
val/accuracy: 0.55
val/avg_entropy: 1.0
```

### Mislabeled Data
```
train/loss_std: 0.95         # Very high variance
train/grad_norm: 0.5          # Hitting clip limit
train/grad_norm_std: 0.4      # Unstable
val/speaker_consistency: 0.45  # Low (model confused)
val/accuracy: 0.35
```

### Memorizing Speakers
```
train/loss_std: 0.05          # Too perfect on train
train/grad_norm: 0.1           # Small updates
val/speaker_consistency: 0.92  # Way too high!
val/accuracy: 0.38            # But accuracy is poor
val/avg_entropy: 0.15         # Overconfident
```

### Insufficient Model Capacity
```
train/loss_std: 0.4           # Can't fit training data well
train/accuracy: 0.45          # Low even on train
val/accuracy: 0.42            # Similar to train (not overfitting)
val/avg_entropy: 1.8          # High uncertainty
All metrics plateau early
```

## Action Items Based on Metrics

### If loss_std is high (>0.5 after epoch 10):
1. Review high_entropy_samples for mislabeling
2. Check per_dataset_accuracy for problematic sources
3. Filter out low-quality audio
4. Consider removing datasets with poor performance

### If gradients are unstable (grad_norm_std > 0.3):
1. Reduce learning rate by 50%
2. Increase batch size or gradient_accumulation_steps
3. Ensure gradient clipping is enabled
4. Check for data loading issues

### If speaker_consistency > 85%:
1. Add more speakers per region
2. Implement audio augmentation (pitch shift, speed perturbation)
3. Increase dropout rate
4. Ensure speaker diversity in training batches
5. Consider mixing speakers within batches

### If specific regions consistently confused:
1. Add more samples from boundary states
2. Review region mapping logic
3. Consider merging highly confused regions
4. Collect targeted data for problem regions

### If model plateaus early:
1. Increase learning rate (if gradients stable)
2. Unfreeze more layers (increase model capacity)
3. Increase hidden_dim in classifier
4. Consider using larger base model (wav2vec2-large)
5. Add more diverse training data

## Recommended Monitoring Dashboard

Priority metrics to watch during training:
1. **val/loss** - Primary early stopping metric
2. **val/macro_f1** - Balanced performance indicator
3. **train/loss_std** - Data quality monitor
4. **val/speaker_consistency** - Overfitting detector
5. **val/per_dataset_accuracy** - Data source health
6. **val/top_confusions** - Problem region pairs

These metrics together tell the complete story of your model's training health and indicate exactly what interventions are needed.