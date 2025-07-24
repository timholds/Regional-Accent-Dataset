#!/usr/bin/env python3
"""
Batch inference script for processing multiple audio files

Usage:
    python batch_inference.py --input_dir path/to/audio/dir --model_path path/to/model
"""

import argparse
import os
import glob
from inference import load_model, preprocess_audio, predict, REGIONS
import pandas as pd
from tqdm import tqdm
import torch


def process_directory(input_dir, model, processor, device="cpu"):
    """Process all WAV files in a directory."""
    # Find all WAV files
    wav_files = glob.glob(os.path.join(input_dir, "**/*.wav"), recursive=True)
    print(f"Found {len(wav_files)} WAV files")
    
    results = []
    
    for audio_path in tqdm(wav_files, desc="Processing files"):
        try:
            # Preprocess audio
            inputs = preprocess_audio(audio_path, processor)
            
            # Make prediction
            pred_idx, confidence, all_probs = predict(model, inputs, device)
            predicted_region = REGIONS[pred_idx]
            
            # Store result
            result = {
                'file_path': audio_path,
                'predicted_region': predicted_region,
                'confidence': confidence
            }
            
            # Add probabilities for each region
            for i, region in enumerate(REGIONS):
                result[f'prob_{region.replace(" ", "_")}'] = all_probs[i]
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            results.append({
                'file_path': audio_path,
                'predicted_region': 'ERROR',
                'confidence': 0.0,
                'error': str(e)
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Batch Regional Accent Classification")
    parser.add_argument(
        "--input_dir", 
        type=str, 
        required=True,
        help="Directory containing audio files"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="accent_classifier_output/best_model",
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--output_csv", 
        type=str, 
        default="batch_predictions.csv",
        help="Output CSV file for predictions"
    )
    parser.add_argument(
        "--use_lora", 
        action="store_true",
        help="Whether the model uses LoRA adapters"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on"
    )
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model, processor = load_model(args.model_path, args.use_lora)
    
    # Process directory
    results = process_directory(args.input_dir, model, processor, args.device)
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"\nResults saved to: {args.output_csv}")
    
    # Print summary
    if 'predicted_region' in df.columns:
        print("\nPrediction Summary:")
        print("-" * 40)
        region_counts = df['predicted_region'].value_counts()
        for region, count in region_counts.items():
            if region != 'ERROR':
                print(f"{region:>20}: {count} files")
        
        if 'ERROR' in region_counts:
            print(f"\nErrors: {region_counts['ERROR']} files")
    
    # Calculate average confidence per region
    print("\nAverage Confidence by Region:")
    print("-" * 40)
    for region in REGIONS:
        region_df = df[df['predicted_region'] == region]
        if len(region_df) > 0:
            avg_conf = region_df['confidence'].mean()
            print(f"{region:>20}: {avg_conf:.2%}")


if __name__ == "__main__":
    main()