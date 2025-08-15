## Model Details

### Key Configuration

Based on the current training script (`train_accent_classifier.py`), here are the key model details:

1. **Model Architecture**: Using `Wav2Vec2ForSequenceClassification` from Hugging Face Transformers
2. **Feature Encoder Freezing**: The CNN feature encoder is frozen via `model.freeze_feature_encoder()` (line 217)
3. **Transformer Layer Freezing**: First 10 of 12 transformer layers are frozen (lines 221-226), keeping only layers 10-11 trainable
4. **Class Weighting**: Implements class weights using `sqrt(max_count/count)` capped at 3.0 for handling imbalanced data (lines 478-499)
5. **Gradient Clipping**: Uses gradient clipping with `max_grad_norm=0.5` (line 289)
6. **Learning Rate**: Default learning rate is 3e-5 with linear warmup scheduler

### Parameter-to-Data Ratio Analysis

Understanding the relationship between model size and dataset size is crucial for successful fine-tuning. Here's how our setup compares to successful speech models:

| Model/Setup | Trainable Params | Training Data | Hours | Params per Hour | Status |
|------------|------------------|---------------|-------|-----------------|---------|
| **Wav2Vec2 Original** | 95M | 960 hours LibriSpeech | 960h | 99K/hour | ✅ SOTA results |
| **Wav2Vec2 Low-Resource** | 95M | 10 hours | 10h | 9.5M/hour | ✅ Works well |
| **Whisper Fine-tune** | 39M (small) | 5 hours custom | 5h | 7.8M/hour | ✅ Good results |
| **XLSR-53 Languages** | 95M | 50 hours per lang | 50h | 1.9M/hour | ✅ Strong performance |
| **Our Current Setup** | 19.5M | 4,410 samples | 3.7h | **5.3M/hour** | ❌ Overfitting |
| **Recommended for Us** | 1-4M | 4,410 samples | 3.7h | 0.3-1.1M/hour | 🎯 Target |

**Key Insights:**
- Successful low-resource fine-tuning typically uses 1-10M params per hour of audio
- With only 3.7 hours of training data, we should train at most 4-5M parameters
- Our current 19.5M trainable params is ~5x too many for our dataset size
- The original Wav2Vec2 had 260x more data but only 5x more parameters

## Models to train
Frozen wav2vec with a classifier head on top to check if we can map the features directly to a linearly seperable space
Unfrozen wav2vec with a classifier head on top
LoRA on top of frozen wav2vec


