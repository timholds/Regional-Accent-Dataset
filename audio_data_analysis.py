#!/usr/bin/env python3
"""
Audio Data Analysis Tool for Regional Accent Dataset

This tool performs comprehensive analysis of audio data across different datasets
to ensure compatibility and identify potential issues before training.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
from collections import defaultdict
import json

warnings.filterwarnings('ignore', category=UserWarning)


class AudioDataAnalyzer:
    """Comprehensive audio data analysis for accent datasets"""
    
    def __init__(self, samples: List[Dict], sample_size: int = 100):
        """
        Initialize analyzer with unified samples
        
        Args:
            samples: List of unified samples from datasets
            sample_size: Number of samples to analyze per dataset
        """
        self.samples = samples
        self.sample_size = sample_size
        self.results = defaultdict(lambda: defaultdict(list))
        
    def analyze_audio_file(self, audio_path: str) -> Dict:
        """Analyze a single audio file"""
        try:
            # Get file info
            file_info = sf.info(audio_path)
            
            # Load audio with librosa (returns normalized float32)
            audio_librosa, sr_librosa = librosa.load(audio_path, sr=None, mono=True)
            
            # Also load with soundfile to get original format info
            audio_sf, sr_sf = sf.read(audio_path)
            
            # If stereo, convert to mono for analysis
            if len(audio_sf.shape) > 1:
                audio_sf = np.mean(audio_sf, axis=1)
            
            analysis = {
                # File format info
                'format': file_info.format,
                'subtype': file_info.subtype,
                'channels': file_info.channels,
                'original_samplerate': file_info.samplerate,
                
                # Audio characteristics
                'duration_seconds': len(audio_librosa) / sr_librosa,
                'num_samples': len(audio_librosa),
                
                # Sample rate info
                'librosa_sr': sr_librosa,
                'soundfile_sr': sr_sf,
                'sr_match': sr_librosa == sr_sf,
                
                # Amplitude statistics (from librosa normalized data)
                'amplitude_mean': float(np.mean(audio_librosa)),
                'amplitude_std': float(np.std(audio_librosa)),
                'amplitude_min': float(np.min(audio_librosa)),
                'amplitude_max': float(np.max(audio_librosa)),
                'amplitude_range': float(np.max(audio_librosa) - np.min(audio_librosa)),
                
                # Check for clipping
                'has_clipping': (np.abs(audio_librosa) >= 0.99).any(),
                'clipping_percentage': float((np.abs(audio_librosa) >= 0.99).sum() / len(audio_librosa) * 100),
                
                # Signal characteristics
                'rms_energy': float(np.sqrt(np.mean(audio_librosa**2))),
                'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(audio_librosa))),
                
                # Check for silence
                'silence_threshold': 0.01,
                'silence_percentage': float((np.abs(audio_librosa) < 0.01).sum() / len(audio_librosa) * 100),
                
                # Data type info
                'librosa_dtype': str(audio_librosa.dtype),
                'soundfile_dtype': str(audio_sf.dtype),
                
                # File size
                'file_size_mb': os.path.getsize(audio_path) / (1024 * 1024)
            }
            
            # Add spectral features
            try:
                spectral_centroids = librosa.feature.spectral_centroid(y=audio_librosa, sr=sr_librosa)[0]
                analysis['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
                analysis['spectral_centroid_std'] = float(np.std(spectral_centroids))
            except:
                analysis['spectral_centroid_mean'] = None
                analysis['spectral_centroid_std'] = None
            
            return analysis
            
        except Exception as e:
            return {
                'error': str(e),
                'audio_path': audio_path
            }
    
    def analyze_datasets(self) -> Dict:
        """Analyze audio characteristics across all datasets"""
        print("Analyzing audio data across datasets...")
        
        # Group samples by dataset
        dataset_samples = defaultdict(list)
        for sample in self.samples:
            dataset_samples[sample['dataset_name']].append(sample)
        
        # Analyze each dataset
        for dataset_name, samples in dataset_samples.items():
            print(f"\nAnalyzing {dataset_name}...")
            
            # Sample a subset for analysis
            n_samples = min(len(samples), self.sample_size)
            sampled = np.random.choice(samples, n_samples, replace=False)
            
            for sample in tqdm(sampled, desc=f"Processing {dataset_name}"):
                audio_path = sample['audio_path']
                if os.path.exists(audio_path):
                    analysis = self.analyze_audio_file(audio_path)
                    
                    # Store results by dataset
                    for key, value in analysis.items():
                        if value is not None and key != 'error':
                            self.results[dataset_name][key].append(value)
        
        return self.results
    
    def generate_report(self) -> Dict:
        """Generate comprehensive analysis report"""
        report = {}
        
        for dataset_name, metrics in self.results.items():
            dataset_report = {}
            
            # Sample rate analysis
            if 'original_samplerate' in metrics:
                srs = metrics['original_samplerate']
                dataset_report['sample_rates'] = {
                    'unique_values': list(set(srs)),
                    'most_common': max(set(srs), key=srs.count),
                    'all_same': len(set(srs)) == 1
                }
            
            # Format analysis
            if 'format' in metrics:
                formats = metrics['format']
                dataset_report['formats'] = {
                    'unique_formats': list(set(formats)),
                    'format_counts': dict(pd.Series(formats).value_counts())
                }
            
            # Duration analysis
            if 'duration_seconds' in metrics:
                durations = metrics['duration_seconds']
                dataset_report['duration'] = {
                    'mean': np.mean(durations),
                    'std': np.std(durations),
                    'min': np.min(durations),
                    'max': np.max(durations),
                    'median': np.median(durations)
                }
            
            # Amplitude analysis
            if 'amplitude_max' in metrics:
                dataset_report['amplitude'] = {
                    'max_values': {
                        'mean': np.mean(metrics['amplitude_max']),
                        'std': np.std(metrics['amplitude_max']),
                        'min': np.min(metrics['amplitude_max']),
                        'max': np.max(metrics['amplitude_max'])
                    },
                    'range_values': {
                        'mean': np.mean(metrics['amplitude_range']),
                        'std': np.std(metrics['amplitude_range'])
                    },
                    'rms_energy': {
                        'mean': np.mean(metrics['rms_energy']),
                        'std': np.std(metrics['rms_energy'])
                    }
                }
            
            # Quality issues
            if 'has_clipping' in metrics:
                dataset_report['quality_issues'] = {
                    'files_with_clipping': sum(metrics['has_clipping']),
                    'clipping_percentage': np.mean(metrics['clipping_percentage']),
                    'high_silence_files': sum(s > 50 for s in metrics['silence_percentage'])
                }
            
            # Channels
            if 'channels' in metrics:
                channels = metrics['channels']
                dataset_report['channels'] = {
                    'unique_values': list(set(channels)),
                    'all_mono': all(c == 1 for c in channels),
                    'channel_counts': dict(pd.Series(channels).value_counts())
                }
            
            report[dataset_name] = dataset_report
        
        return report
    
    def plot_analysis(self, output_dir: str = "audio_analysis_plots"):
        """Generate visualization plots"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Duration distribution
        plt.figure(figsize=(12, 6))
        for i, (dataset_name, metrics) in enumerate(self.results.items()):
            if 'duration_seconds' in metrics:
                plt.subplot(1, len(self.results), i+1)
                plt.hist(metrics['duration_seconds'], bins=30, alpha=0.7)
                plt.title(f"{dataset_name} Duration Distribution")
                plt.xlabel("Duration (seconds)")
                plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/duration_distribution.png")
        plt.close()
        
        # Amplitude distribution
        plt.figure(figsize=(12, 6))
        for i, (dataset_name, metrics) in enumerate(self.results.items()):
            if 'amplitude_max' in metrics:
                plt.subplot(1, len(self.results), i+1)
                plt.hist(metrics['amplitude_max'], bins=30, alpha=0.7)
                plt.title(f"{dataset_name} Max Amplitude")
                plt.xlabel("Max Amplitude")
                plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/amplitude_distribution.png")
        plt.close()
        
        # RMS Energy comparison
        plt.figure(figsize=(10, 6))
        rms_data = []
        labels = []
        for dataset_name, metrics in self.results.items():
            if 'rms_energy' in metrics:
                rms_data.append(metrics['rms_energy'])
                labels.append(dataset_name)
        
        if rms_data:
            plt.boxplot(rms_data, labels=labels)
            plt.title("RMS Energy Distribution by Dataset")
            plt.ylabel("RMS Energy")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/rms_energy_comparison.png")
            plt.close()
    
    def check_compatibility(self) -> Dict:
        """Check compatibility issues between datasets"""
        issues = {
            'critical': [],
            'warnings': [],
            'info': []
        }
        
        # Check sample rates
        all_srs = []
        for dataset_name, metrics in self.results.items():
            if 'original_samplerate' in metrics:
                all_srs.extend([(dataset_name, sr) for sr in metrics['original_samplerate']])
        
        unique_srs = set(sr for _, sr in all_srs)
        if len(unique_srs) > 1:
            issues['warnings'].append(
                f"Multiple sample rates detected: {unique_srs}. "
                "Resampling will be needed for consistency."
            )
        
        # Check formats
        all_formats = []
        for dataset_name, metrics in self.results.items():
            if 'format' in metrics:
                all_formats.extend(set(metrics['format']))
        
        if len(set(all_formats)) > 1:
            issues['info'].append(
                f"Multiple audio formats detected: {set(all_formats)}. "
                "This is OK as libraries handle format conversion."
            )
        
        # Check for quality issues
        for dataset_name, metrics in self.results.items():
            if 'has_clipping' in metrics:
                clipping_rate = sum(metrics['has_clipping']) / len(metrics['has_clipping'])
                if clipping_rate > 0.1:
                    issues['warnings'].append(
                        f"{dataset_name}: {clipping_rate*100:.1f}% of files have clipping"
                    )
        
        # Check amplitude ranges
        max_amps = []
        for dataset_name, metrics in self.results.items():
            if 'amplitude_max' in metrics:
                max_amps.append((dataset_name, np.mean(metrics['amplitude_max'])))
        
        if max_amps:
            amp_values = [amp for _, amp in max_amps]
            if max(amp_values) / min(amp_values) > 5:
                issues['warnings'].append(
                    "Large amplitude differences between datasets. "
                    "Consider normalizing audio levels."
                )
        
        return issues


def analyze_audio_datasets(unified_dataset_path: str = "unified_dataset_samples.csv", 
                          sample_size: int = 50):
    """Main function to analyze audio datasets"""
    
    print("Loading unified dataset...")
    df = pd.read_csv(unified_dataset_path)
    samples = df.to_dict('records')
    
    # Filter to only samples with valid audio paths
    valid_samples = [s for s in samples if pd.notna(s.get('audio_path')) and 
                     os.path.exists(s.get('audio_path', ''))]
    
    print(f"Found {len(valid_samples)} samples with valid audio paths")
    
    # Create analyzer
    analyzer = AudioDataAnalyzer(valid_samples, sample_size=sample_size)
    
    # Run analysis
    results = analyzer.analyze_datasets()
    
    # Generate report
    report = analyzer.generate_report()
    
    # Print report
    print("\n" + "="*60)
    print("AUDIO DATA ANALYSIS REPORT")
    print("="*60)
    
    for dataset_name, dataset_report in report.items():
        print(f"\n{dataset_name} Dataset:")
        print("-" * 40)
        
        if 'sample_rates' in dataset_report:
            sr_info = dataset_report['sample_rates']
            print(f"Sample Rates: {sr_info['unique_values']}")
            if not sr_info['all_same']:
                print(f"  ⚠️  Multiple sample rates detected!")
        
        if 'formats' in dataset_report:
            format_info = dataset_report['formats']
            print(f"Audio Formats: {format_info['unique_formats']}")
        
        if 'duration' in dataset_report:
            dur = dataset_report['duration']
            print(f"Duration: {dur['mean']:.2f}s ± {dur['std']:.2f}s (range: {dur['min']:.2f}s - {dur['max']:.2f}s)")
        
        if 'amplitude' in dataset_report:
            amp = dataset_report['amplitude']
            print(f"Max Amplitude: {amp['max_values']['mean']:.3f} ± {amp['max_values']['std']:.3f}")
            print(f"RMS Energy: {amp['rms_energy']['mean']:.3f} ± {amp['rms_energy']['std']:.3f}")
        
        if 'quality_issues' in dataset_report:
            quality = dataset_report['quality_issues']
            if quality['files_with_clipping'] > 0:
                print(f"  ⚠️  Clipping detected in {quality['files_with_clipping']} files")
            if quality['high_silence_files'] > 0:
                print(f"  ⚠️  High silence in {quality['high_silence_files']} files")
        
        if 'channels' in dataset_report:
            ch = dataset_report['channels']
            if not ch['all_mono']:
                print(f"Channels: {ch['channel_counts']}")
    
    # Check compatibility
    print("\n" + "="*60)
    print("COMPATIBILITY CHECK")
    print("="*60)
    
    compatibility = analyzer.check_compatibility()
    
    for level in ['critical', 'warnings', 'info']:
        if compatibility[level]:
            print(f"\n{level.upper()}:")
            for issue in compatibility[level]:
                print(f"  - {issue}")
    
    # Generate plots
    print("\nGenerating analysis plots...")
    analyzer.plot_analysis()
    print("Plots saved to audio_analysis_plots/")
    
    # Save detailed report
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj
    
    serializable_report = convert_to_serializable(report)
    with open("audio_analysis_report.json", 'w') as f:
        json.dump(serializable_report, f, indent=2)
    print("\nDetailed report saved to audio_analysis_report.json")
    
    return report, compatibility


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze audio data across datasets")
    parser.add_argument("--csv", default="unified_dataset_samples.csv", 
                       help="Path to unified dataset CSV")
    parser.add_argument("--sample_size", type=int, default=50,
                       help="Number of samples to analyze per dataset")
    
    args = parser.parse_args()
    
    analyze_audio_datasets(args.csv, args.sample_size)