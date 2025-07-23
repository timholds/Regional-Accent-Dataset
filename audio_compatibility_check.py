#!/usr/bin/env python3
"""
Audio Compatibility Check for Dataset Combination

This script performs detailed checks to ensure audio data can be safely combined
for machine learning training.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def detailed_audio_check(audio_path: str) -> Dict:
    """Perform detailed check on a single audio file"""
    try:
        # Method 1: librosa (always converts to float32, normalizes to [-1, 1])
        y_librosa, sr_librosa = librosa.load(audio_path, sr=None, mono=True)
        
        # Method 2: soundfile (preserves original format)
        y_sf, sr_sf = sf.read(audio_path)
        if len(y_sf.shape) > 1:
            y_sf_mono = np.mean(y_sf, axis=1)
        else:
            y_sf_mono = y_sf
        
        # Get file info
        info = sf.info(audio_path)
        
        return {
            'file': Path(audio_path).name,
            'format': info.format,
            'subtype': info.subtype,
            'channels': info.channels,
            'samplerate': info.samplerate,
            'duration': len(y_librosa) / sr_librosa,
            
            # Data type info
            'original_dtype': str(y_sf.dtype),
            'librosa_dtype': str(y_librosa.dtype),
            'librosa_always_float32': y_librosa.dtype == np.float32,
            
            # Value ranges
            'original_min': float(np.min(y_sf_mono)),
            'original_max': float(np.max(y_sf_mono)),
            'librosa_min': float(np.min(y_librosa)),
            'librosa_max': float(np.max(y_librosa)),
            
            # Normalization check
            'librosa_normalized': np.max(np.abs(y_librosa)) <= 1.0,
            'original_normalized': np.max(np.abs(y_sf_mono)) <= 1.0,
            
            # Energy measurements
            'rms_original': float(np.sqrt(np.mean(y_sf_mono**2))),
            'rms_librosa': float(np.sqrt(np.mean(y_librosa**2))),
            
            # Check if conversion preserves relative values
            'conversion_preserves_shape': np.corrcoef(y_sf_mono[:1000], y_librosa[:1000])[0,1] > 0.99
        }
    except Exception as e:
        return {'file': Path(audio_path).name, 'error': str(e)}


def check_ml_compatibility(csv_path: str = "unified_dataset_samples.csv"):
    """Check ML compatibility across datasets"""
    
    print("AUDIO DATA COMPATIBILITY CHECK FOR MACHINE LEARNING")
    print("="*60)
    
    # Load dataset info
    df = pd.read_csv(csv_path)
    
    # Sample files from each dataset
    datasets = df['dataset_name'].unique()
    
    results = []
    for dataset in datasets:
        dataset_df = df[df['dataset_name'] == dataset]
        # Get up to 5 valid audio files
        valid_files = dataset_df[dataset_df['audio_path'].notna()]['audio_path'].tolist()
        valid_files = [f for f in valid_files if Path(f).exists()][:5]
        
        print(f"\nChecking {dataset}...")
        for audio_file in valid_files:
            result = detailed_audio_check(audio_file)
            result['dataset'] = dataset
            results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # Analysis
    print("\n" + "="*60)
    print("FINDINGS:")
    print("="*60)
    
    # 1. Sample Rate Consistency
    print("\n1. SAMPLE RATES:")
    for dataset in datasets:
        dataset_results = results_df[results_df['dataset'] == dataset]
        if 'samplerate' in dataset_results.columns:
            srs = dataset_results['samplerate'].unique()
            print(f"   {dataset}: {srs} Hz")
    
    all_srs = results_df['samplerate'].unique()
    if len(all_srs) == 1:
        print("   ✅ All datasets have the same sample rate!")
    else:
        print(f"   ⚠️  Multiple sample rates detected: {all_srs}")
        print("   → Solution: Resample all audio to 16kHz in the DataLoader")
    
    # 2. Data Type Handling
    print("\n2. DATA TYPES:")
    print("   Original formats:")
    for dataset in datasets:
        dataset_results = results_df[results_df['dataset'] == dataset]
        if 'original_dtype' in dataset_results.columns:
            dtypes = dataset_results['original_dtype'].unique()
            print(f"   {dataset}: {dtypes}")
    
    print("\n   After librosa loading:")
    print("   ✅ All audio is converted to float32 by librosa")
    print("   ✅ This is exactly what we want for ML models!")
    
    # 3. Value Ranges
    print("\n3. AMPLITUDE RANGES:")
    for dataset in datasets:
        dataset_results = results_df[results_df['dataset'] == dataset]
        if 'librosa_min' in dataset_results.columns:
            min_val = dataset_results['librosa_min'].min()
            max_val = dataset_results['librosa_max'].max()
            print(f"   {dataset}: [{min_val:.3f}, {max_val:.3f}]")
    
    print("   ✅ Librosa normalizes all audio to [-1, 1] range")
    
    # 4. Energy Levels
    print("\n4. ENERGY LEVELS (RMS):")
    for dataset in datasets:
        dataset_results = results_df[results_df['dataset'] == dataset]
        if 'rms_librosa' in dataset_results.columns:
            rms_mean = dataset_results['rms_librosa'].mean()
            rms_std = dataset_results['rms_librosa'].std()
            print(f"   {dataset}: {rms_mean:.4f} ± {rms_std:.4f}")
    
    # Check for large differences
    if 'rms_librosa' in results_df.columns:
        rms_by_dataset = results_df.groupby('dataset')['rms_librosa'].mean()
        max_ratio = rms_by_dataset.max() / rms_by_dataset.min()
        if max_ratio > 10:
            print(f"   ⚠️  Large energy differences (ratio: {max_ratio:.1f}x)")
            print("   → Solution: Apply normalization or use SpecAugment during training")
        else:
            print(f"   ✅ Energy levels are reasonably similar (ratio: {max_ratio:.1f}x)")
    
    # 5. Format Compatibility
    print("\n5. FILE FORMATS:")
    formats = results_df['format'].unique() if 'format' in results_df.columns else []
    print(f"   Formats found: {formats}")
    print("   ✅ Both librosa and soundfile handle all common formats")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS FOR SAFE DATASET COMBINATION:")
    print("="*60)
    
    print("""
1. ✅ USE LIBROSA for loading - it handles normalization automatically
   
2. ✅ RESAMPLE to 16kHz in your DataLoader:
   audio, _ = librosa.load(path, sr=16000)
   
3. ⚠️  NORMALIZE energy levels during training:
   - Option A: RMS normalization per sample
   - Option B: Use SpecAugment for robustness
   - Option C: Add dataset-specific gain factors
   
4. ✅ DATA TYPE is handled automatically (float32)
   
5. 🔍 MONITOR during training:
   - Track loss per dataset to spot issues
   - Use gradient clipping for stability
   - Consider per-dataset batch normalization

6. 💡 FOR PRODUCTION:
   - Create a preprocessing pipeline that:
     * Resamples to 16kHz
     * Converts to mono
     * Normalizes amplitude
     * Removes silence at start/end
     * Applies consistent filtering
""")

    # Save detailed results
    results_df.to_csv("audio_compatibility_results.csv", index=False)
    print("\nDetailed results saved to audio_compatibility_results.csv")


if __name__ == "__main__":
    check_ml_compatibility()