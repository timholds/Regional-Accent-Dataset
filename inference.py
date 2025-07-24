#!/usr/bin/env python3
"""
Regional Accent Classifier Inference Script

This script loads a trained accent classification model and performs inference
on audio files to predict the speaker's regional accent.

Usage:
    python inference.py --audio_path path/to/audio.wav --model_path path/to/model
"""

import argparse
import os
import numpy as np
import torch
import librosa
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
from peft import PeftModel
import json


# Define the 8 TIMIT-aligned regions
REGIONS = [
    "New England",
    "New York Metropolitan", 
    "Mid-Atlantic",
    "South Atlantic",
    "Deep South",
    "Upper Midwest",
    "Lower Midwest",
    "West"
]


def load_model(model_path, use_lora=False):
    """Load the trained model and processor."""
    print(f"Loading model from {model_path}...")
    
    # Load processor
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    
    # Load model
    if use_lora:
        # Load base model first
        base_model = Wav2Vec2ForSequenceClassification.from_pretrained(
            "facebook/wav2vec2-base", 
            num_labels=8
        )
        # Load LoRA weights
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()  # Merge LoRA weights with base model
    else:
        model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
    
    model.eval()
    return model, processor


def preprocess_audio(audio_path, processor, target_sr=16000):
    """Load and preprocess audio file."""
    print(f"Loading audio from {audio_path}...")
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=target_sr)
    
    # Process audio
    inputs = processor(
        audio, 
        sampling_rate=target_sr, 
        return_tensors="pt",
        padding=True
    )
    
    return inputs


def predict(model, inputs, device="cpu"):
    """Make prediction on preprocessed audio."""
    model = model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Get probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Get prediction
        pred_idx = torch.argmax(logits, dim=-1).item()
        confidence = probs[0, pred_idx].item()
        
    return pred_idx, confidence, probs[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Regional Accent Classifier Inference")
    parser.add_argument(
        "--audio_path", 
        type=str, 
        required=True,
        help="Path to audio file (WAV format)"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="accent_classifier_output/best_model",
        help="Path to trained model directory"
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
    parser.add_argument(
        "--show_all_probs", 
        action="store_true",
        help="Show probabilities for all regions"
    )
    
    args = parser.parse_args()
    
    # Validate audio file
    if not os.path.exists(args.audio_path):
        raise FileNotFoundError(f"Audio file not found: {args.audio_path}")
    
    if not args.audio_path.lower().endswith('.wav'):
        print("Warning: Audio file should be in WAV format for best results")
    
    # Load model
    model, processor = load_model(args.model_path, args.use_lora)
    
    # Preprocess audio
    inputs = preprocess_audio(args.audio_path, processor)
    
    # Make prediction
    pred_idx, confidence, all_probs = predict(model, inputs, args.device)
    predicted_region = REGIONS[pred_idx]
    
    # Display results
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    print(f"Audio file: {args.audio_path}")
    print(f"Predicted region: {predicted_region}")
    print(f"Confidence: {confidence:.2%}")
    
    if args.show_all_probs:
        print("\nAll region probabilities:")
        print("-"*30)
        # Sort by probability
        sorted_indices = np.argsort(all_probs)[::-1]
        for idx in sorted_indices:
            print(f"{REGIONS[idx]:>20}: {all_probs[idx]:.2%}")
    
    print("="*50)
    
    # Save results to JSON
    results = {
        "audio_file": args.audio_path,
        "predicted_region": predicted_region,
        "confidence": float(confidence),
        "all_probabilities": {
            REGIONS[i]: float(all_probs[i]) for i in range(len(REGIONS))
        }
    }
    
    output_path = args.audio_path.replace('.wav', '_prediction.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()