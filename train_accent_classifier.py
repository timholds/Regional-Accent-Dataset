#!/usr/bin/env python3
"""
Training script for US Regional Accent Classifier

This script trains a Wav2Vec2-based accent classification model
using a prepared dataset.
"""

import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Processor,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import wandb
import json
import seaborn as sns
import matplotlib.pyplot as plt

from unified_dataset import UnifiedSample, UnifiedAccentDatasetTorch


def collate_fn(batch):
    """Custom collate function to handle batching"""
    # Stack tensors
    input_values = torch.stack([item['input_values'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    # Keep metadata as lists (if needed)
    speaker_ids = [item['speaker_id'] for item in batch]
    
    return {
        'input_values': input_values,
        'attention_mask': attention_mask,
        'labels': labels,
        'speaker_ids': speaker_ids
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train Regional Accent Classifier")
    
    # Data arguments
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to prepared dataset directory")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, 
                       default="facebook/wav2vec2-base",
                       help="Pretrained model to use")
    parser.add_argument("--num_labels", type=int, default=8,
                       help="Number of accent classes")
    parser.add_argument("--use_lora", action="store_true",
                       help="Use LoRA for efficient fine-tuning")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=32,
                       help="Evaluation batch size")
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500,
                       help="Number of warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Max gradient norm for clipping")
    
    # Other arguments
    parser.add_argument("--output_dir", type=str, default="accent_classifier_output",
                       help="Output directory for model and logs")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=100,
                       help="Evaluate every N steps")
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="Log every N steps")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="accent-classifier",
                       help="W&B project name")
    
    return parser.parse_args()


def load_prepared_dataset(dataset_path: str, processor):
    """Load a prepared dataset"""
    dataset_path = Path(dataset_path)
    
    # Load config
    with open(dataset_path / "config.json", 'r') as f:
        config = json.load(f)
    
    # Load metadata
    with open(dataset_path / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    print(f"Loading dataset: {metadata['dataset_name']}")
    print(f"Created: {metadata['created_at']}")
    print(f"Datasets included: {metadata['statistics']['datasets_included']}")
    
    # Load CSV files
    train_df = pd.read_csv(dataset_path / "train.csv")
    val_df = pd.read_csv(dataset_path / "val.csv")
    test_df = pd.read_csv(dataset_path / "test.csv")
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    
    # Convert to UnifiedSample objects
    def df_to_samples(df):
        samples = []
        for _, row in df.iterrows():
            sample = UnifiedSample(
                sample_id=row['sample_id'],
                dataset_name=row['dataset_name'],
                speaker_id=row['speaker_id'],
                audio_path=row['audio_path'],
                transcript=row.get('transcript', ''),
                region_label=row['region_label'],
                original_accent_label=row.get('original_accent_label', ''),
                state=row.get('state', None),
                gender=row.get('gender', 'U'),
                age=row.get('age', None),
                native_language=row.get('native_language', None),
                duration=row.get('duration', None),
                sample_rate=row.get('sample_rate', 16000),
                is_validated=row.get('is_validated', True),
                quality_score=row.get('quality_score', None)
            )
            samples.append(sample)
        return samples
    
    train_samples = df_to_samples(train_df)
    val_samples = df_to_samples(val_df)
    test_samples = df_to_samples(test_df)
    
    # Create PyTorch datasets
    train_dataset = UnifiedAccentDatasetTorch(train_samples, processor)
    val_dataset = UnifiedAccentDatasetTorch(val_samples, processor)
    test_dataset = UnifiedAccentDatasetTorch(test_samples, processor)
    
    return train_dataset, val_dataset, test_dataset, config, metadata


def setup_model(args, num_labels):
    """Setup model with optional LoRA"""
    print(f"Loading model: {args.model_name}")
    
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        output_hidden_states=True
    )
    
    if args.use_lora:
        print("Applying LoRA for efficient fine-tuning...")
        # Note: You'll need to install peft: pip install peft
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=16,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj"],  # Wav2Vec2 attention layers
            )
            
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        except ImportError:
            print("Warning: peft not installed. Install with: pip install peft")
            print("Continuing without LoRA...")
    
    return model


def train_epoch(model, train_loader, optimizer, scheduler, device, args):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc="Training")
    
    for step, batch in enumerate(progress_bar):
        # Move batch to device
        input_values = batch['input_values'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(
            input_values=input_values,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss / args.gradient_accumulation_steps
        total_loss += loss.item()
        
        # Backward pass
        loss.backward()
        
        # Update weights
        if (step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': loss.item() * args.gradient_accumulation_steps,
            'lr': scheduler.get_last_lr()[0]
        })
        
        # Logging
        if args.use_wandb and step % args.logging_steps == 0:
            wandb.log({
                'train/loss': loss.item() * args.gradient_accumulation_steps,
                'train/learning_rate': scheduler.get_last_lr()[0],
                'train/step': step
            })
    
    return total_loss / len(train_loader)


def evaluate(model, eval_loader, device, dataset_name="val"):
    """Evaluate model"""
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc=f"Evaluating {dataset_name}"):
            input_values = batch['input_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_values=input_values,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            
            predictions = outputs.logits.argmax(dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    avg_loss = total_loss / len(eval_loader)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'predictions': all_predictions,
        'labels': all_labels
    }


def save_confusion_matrix(y_true, y_pred, labels, output_path):
    """Save confusion matrix plot"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=args)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load processor
    print(f"\nLoading processor from {args.model_name}")
    processor = Wav2Vec2Processor.from_pretrained(args.model_name)
    
    # Load prepared dataset
    print(f"\nLoading prepared dataset from {args.dataset_path}")
    train_dataset, val_dataset, test_dataset, config, metadata = load_prepared_dataset(
        args.dataset_path, processor
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.eval_batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.eval_batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    print(f"\nDataLoader info:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Get number of labels and label names
    num_labels = config['num_labels']
    label_mapping = config['label_mapping']
    label_names = [k for k, v in sorted(label_mapping.items(), key=lambda x: x[1])]
    print(f"\nNumber of accent classes: {num_labels}")
    print(f"Classes: {', '.join(label_names)}")
    
    # Setup model
    model = setup_model(args, num_labels)
    model.to(device)
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    print(f"\nStarting training for {args.num_epochs} epochs...")
    best_val_accuracy = 0
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, args)
        print(f"Average training loss: {train_loss:.4f}")
        
        # Evaluate
        val_results = evaluate(model, val_loader, device, "validation")
        print(f"Validation loss: {val_results['loss']:.4f}")
        print(f"Validation accuracy: {val_results['accuracy']:.4f}")
        
        # Log to wandb
        if args.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train/epoch_loss': train_loss,
                'val/loss': val_results['loss'],
                'val/accuracy': val_results['accuracy']
            })
        
        # Save best model
        if val_results['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_results['accuracy']
            save_path = os.path.join(args.output_dir, 'best_model')
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            print(f"Saved best model with accuracy: {best_val_accuracy:.4f}")
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_results = evaluate(model, test_loader, device, "test")
    print(f"Test loss: {test_results['loss']:.4f}")
    print(f"Test accuracy: {test_results['accuracy']:.4f}")
    
    # Generate and save classification report
    print("\nClassification Report:")
    report = classification_report(
        test_results['labels'], 
        test_results['predictions'], 
        target_names=label_names,
        digits=3
    )
    print(report)
    
    # Save confusion matrix
    save_confusion_matrix(
        test_results['labels'],
        test_results['predictions'],
        label_names,
        os.path.join(args.output_dir, 'confusion_matrix.png')
    )
    
    # Save final results
    results = {
        'args': vars(args),
        'dataset_info': {
            'name': metadata['dataset_name'],
            'datasets_included': metadata['statistics']['datasets_included'],
            'total_samples': metadata['statistics']['total_samples'],
            'label_mapping': label_mapping
        },
        'best_val_accuracy': best_val_accuracy,
        'test_accuracy': test_results['accuracy'],
        'test_loss': test_results['loss'],
        'classification_report': report
    }
    
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    if args.use_wandb:
        wandb.log({
            'test/loss': test_results['loss'],
            'test/accuracy': test_results['accuracy']
        })
        wandb.finish()
    
    print(f"\nTraining complete! Results saved to {args.output_dir}")
    print(f"  - Best model: {os.path.join(args.output_dir, 'best_model')}")
    print(f"  - Results: {os.path.join(args.output_dir, 'results.json')}")
    print(f"  - Confusion matrix: {os.path.join(args.output_dir, 'confusion_matrix.png')}")


if __name__ == "__main__":
    main()