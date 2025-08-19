"""
Unified Dataset Pipeline for US Regional Accent Classification

This module provides a unified interface for loading, processing, and combining
multiple accent datasets into a single training dataset.
"""

import os
import json
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import logging
import tarfile
import zipfile
import shutil
import subprocess
import requests
from urllib.parse import urlparse
import librosa

from region_mappings import get_region_for_state, MediumRegion, MEDIUM_MAPPINGS


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class UnifiedSample:
    """Unified schema for all audio samples across datasets"""
    sample_id: str  # Unique ID across all datasets
    dataset_name: str  # Source dataset (TIMIT, CommonVoice, etc.)
    speaker_id: str
    audio_path: str
    transcript: str
    
    # Regional classification
    region_label: str  # Our 8-region classification
    original_accent_label: str  # Original dataset's accent label
    state: Optional[str] = None
    
    # Speaker metadata
    gender: Optional[str] = None  # M/F/O/U
    age: Optional[str] = None
    native_language: Optional[str] = None
    
    # Audio metadata
    duration: Optional[float] = None
    sample_rate: Optional[int] = None
    
    # Quality/filtering metadata
    is_validated: bool = True
    quality_score: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


class BaseDatasetLoader(ABC):
    """Abstract base class for dataset loaders"""
    
    def __init__(self, data_root: str, cache_dir: str = "~/.cache/accent_datasets"):
        self.data_root = Path(data_root)
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.samples: List[UnifiedSample] = []
        
    @abstractmethod
    def download(self) -> bool:
        """Download dataset if not already cached"""
        pass
    
    @abstractmethod
    def load(self) -> List[UnifiedSample]:
        """Load and process dataset into unified format"""
        pass
    
    @abstractmethod
    def get_dataset_info(self) -> Dict:
        """Return dataset statistics and metadata"""
        pass
    
    def _generate_sample_id(self, dataset_name: str, speaker_id: str, utterance_id: str) -> str:
        """Generate unique sample ID"""
        return f"{dataset_name}_{speaker_id}_{utterance_id}"
    
    
    def _map_to_region(self, state: str = None, city: str = None, 
                      accent_description: str = None) -> Tuple[str, str]:
        """
        Map various location/accent info to our 8-region system
        Returns: (region_label, original_label)
        """
        # If we have state info, use our mapping
        if state:
            region = get_region_for_state(state, classification='medium')
            return region, state
            
        # City-based mapping for datasets like CORAAL
        city_to_region = {
            'detroit': 'Midwest',
            'new york': 'New York Metropolitan',
            'nyc': 'New York Metropolitan',
            'atlanta': 'Deep South',
            'washington': 'Mid-Atlantic',
            'dc': 'Mid-Atlantic',
            'boston': 'New England',
            'chicago': 'Midwest',
            'los angeles': 'West',
            'seattle': 'West',
            'dallas': 'West',  # Texas is in West for medium classification
            'houston': 'West',
        }
        
        if city:
            city_lower = city.lower()
            for city_key, region in city_to_region.items():
                if city_key in city_lower:
                    return region, city
        
        # Parse accent descriptions (for Common Voice)
        if accent_description:
            desc_lower = accent_description.lower()
            
            # Regional keywords mapping
            accent_keywords = {
                'New England': ['new england', 'boston', 'massachusetts', 'maine', 'vermont'],
                'New York Metropolitan': ['new york', 'nyc', 'brooklyn', 'bronx', 'manhattan'],
                'Mid-Atlantic': ['mid-atlantic', 'mid atlantic', 'philadelphia', 'baltimore', 'dc', 'washington'],
                'South Atlantic': ['virginia', 'carolina', 'florida', 'georgia'],
                'Deep South': ['southern', 'deep south', 'alabama', 'mississippi', 'louisiana', 'atlanta'],
                'Midwest': ['midwest', 'upper midwest', 'lower midwest', 'michigan', 'wisconsin', 'minnesota', 'chicago', 'ohio', 'indiana', 'missouri', 'iowa', 'illinois', 'detroit', 'milwaukee', 'kansas', 'nebraska'],
                'West': ['western', 'california', 'pacific', 'texas', 'colorado', 'arizona', 'nevada']
            }
            
            for region, keywords in accent_keywords.items():
                if any(keyword in desc_lower for keyword in keywords):
                    return region, accent_description
        
        # Default to West if we can't determine
        return 'West', accent_description or 'Unknown'


class TIMITLoader(BaseDatasetLoader):
    """Loader for TIMIT dataset"""
    
    def download(self) -> bool:
        """Download TIMIT dataset from Kaggle if not already present"""
        timit_path = self.cache_dir / 'timit'
        
        # Check if TIMIT directories exist
        if (timit_path / 'TRAIN').exists() and (timit_path / 'TEST').exists():
            logger.info("TIMIT dataset found")
            return True
        
        # Try to download from Kaggle
        logger.info("TIMIT not found locally. Attempting to download from Kaggle...")
        
        # Check if kaggle CLI is available and configured
        try:
            result = subprocess.run(['kaggle', '--version'], capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("Kaggle CLI not found. Please install with: pip install kaggle")
                return False
        except FileNotFoundError:
            logger.error("Kaggle CLI not found. Please install with: pip install kaggle")
            return False
        
        # Check for Kaggle API credentials
        kaggle_config = Path.home() / '.kaggle' / 'kaggle.json'
        if not kaggle_config.exists():
            logger.error("Kaggle API credentials not found. Please:")
            logger.error("1. Go to https://www.kaggle.com/account")
            logger.error("2. Create New API Token")
            logger.error(f"3. Place kaggle.json at {kaggle_config}")
            return False
        
        # Download the dataset
        cache_dir = self.cache_dir / 'timit'
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Downloading TIMIT corpus from Kaggle (this may take a few minutes)...")
        download_cmd = [
            'kaggle', 'datasets', 'download',
            '-d', 'mfekadu/darpa-timit-acousticphonetic-continuous-speech',
            '-p', str(cache_dir),
            '--unzip'
        ]
        
        result = subprocess.run(download_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Failed to download TIMIT: {result.stderr}")
            return False
        
        logger.info("TIMIT downloaded successfully. Organizing files...")
        
        # The mfekadu dataset has structure: data/TRAIN and data/TEST
        kaggle_data_path = cache_dir / 'data'
        
        if kaggle_data_path.exists():
            # Move TRAIN and TEST to the data_root
            if (kaggle_data_path / 'TRAIN').exists():
                target_train = timit_path / 'TRAIN'
                if target_train.exists():
                    shutil.rmtree(target_train)
                shutil.move(str(kaggle_data_path / 'TRAIN'), str(timit_path))
                logger.info(f"Moved TRAIN directory to {timit_path}")
            
            if (kaggle_data_path / 'TEST').exists():
                target_test = timit_path / 'TEST'
                if target_test.exists():
                    shutil.rmtree(target_test)
                shutil.move(str(kaggle_data_path / 'TEST'), str(timit_path))
                logger.info(f"Moved TEST directory to {timit_path}")
            
            # Clean up the cache directory
            shutil.rmtree(cache_dir)
            
            # Remove duplicate .WAV.wav files if they exist
            logger.info("Cleaning up duplicate .WAV.wav files...")
            for wav_file in timit_path.rglob("*.WAV.wav"):
                wav_file.unlink()
                
            logger.info("TIMIT dataset ready for use")
            return True
        else:
            logger.error(f"Unexpected dataset structure. Please check {cache_dir}")
            return False
    
    def load(self) -> List[UnifiedSample]:
        """Load TIMIT dataset"""
        if not self.download():
            return []
            
        from loaders.timit_loader import TimitDatasetLoader, TIMIT_REGIONS
        
        logger.info("Loading TIMIT dataset...")
        loader = TimitDatasetLoader(str(self.cache_dir / 'timit'))
        timit_samples = loader.load_dataset()
        
        # Convert to unified format
        unified_samples = []
        for ts in timit_samples:
            # Map TIMIT dialect region to our classification
            region_name = TIMIT_REGIONS.get(ts.dialect_region, 'Unknown')
            
            # Map TIMIT regions to our 8-region system
            timit_to_our_regions = {
                'New England': 'New England',
                'Northern': 'Midwest',
                'North Midland': 'Midwest',
                'South Midland': 'Deep South',
                'Southern': 'Deep South',
                'New York City': 'New York Metropolitan',
                'Western': 'West',
                'Army Brat (moved around)': 'West'  # Default to West
            }
            
            our_region = timit_to_our_regions.get(region_name, 'West')
            
            unified_sample = UnifiedSample(
                sample_id=self._generate_sample_id('TIMIT', ts.speaker_id, ts.sentence_id),
                dataset_name='TIMIT',
                speaker_id=ts.speaker_id,
                audio_path=ts.audio_path,
                transcript=ts.transcript,
                region_label=our_region,
                original_accent_label=region_name,
                gender=ts.gender,
                sample_rate=16000
            )
            unified_samples.append(unified_sample)
        
        self.samples = unified_samples
        logger.info(f"Loaded {len(unified_samples)} samples from TIMIT")
        return unified_samples
    
    def get_dataset_info(self) -> Dict:
        """Get TIMIT dataset statistics"""
        if not self.samples:
            self.load()
            
        df = pd.DataFrame([s.to_dict() for s in self.samples])
        
        return {
            'dataset': 'TIMIT',
            'total_samples': len(self.samples),
            'total_speakers': df['speaker_id'].nunique(),
            'region_distribution': df['region_label'].value_counts().to_dict(),
            'gender_distribution': df['gender'].value_counts().to_dict(),
            'total_duration_hours': len(self.samples) * 3 / 3600  # Approximate 3s per sample
        }


# CommonVoiceLoader has been moved to loaders/commonvoice_loader.py


class CORAALLoader(BaseDatasetLoader):
    """Loader for CORAAL (Corpus of Regional African American Language)"""
    
    CORAAL_COMPONENTS = {
        'ATL': {'city': 'Atlanta', 'year': '2017', 'region': 'Deep South',
                'audio_urls': [
                    'https://lingtools.uoregon.edu/coraal/atl/2020.05/ATL_audio_part01_2020.05.tar.gz',
                    'https://lingtools.uoregon.edu/coraal/atl/2020.05/ATL_audio_part02_2020.05.tar.gz',
                    'https://lingtools.uoregon.edu/coraal/atl/2020.05/ATL_audio_part03_2020.05.tar.gz',
                    'https://lingtools.uoregon.edu/coraal/atl/2020.05/ATL_audio_part04_2020.05.tar.gz'
                ]},
        'DCA': {'city': 'Washington DC', 'year': '1968', 'region': 'Mid-Atlantic',
                'audio_urls': [
                    'https://lingtools.uoregon.edu/coraal/dca/2018.10.06/DCA_audio_part01_2018.10.06.tar.gz',
                    'https://lingtools.uoregon.edu/coraal/dca/2018.10.06/DCA_audio_part02_2018.10.06.tar.gz'
                ]},
        'DCB': {'city': 'Washington DC', 'year': '2016', 'region': 'Mid-Atlantic',
                'audio_urls': [
                    'https://lingtools.uoregon.edu/coraal/dcb/2018.10.06/DCB_audio_part01_2018.10.06.tar.gz'
                ]},
        'PRV': {'city': 'Princeville NC', 'year': '2004', 'region': 'South Atlantic',
                'audio_urls': [
                    'https://lingtools.uoregon.edu/coraal/prv/2018.10.06/PRV_audio_part01_2018.10.06.tar.gz'
                ]},
        'VLD': {'city': 'Valdosta GA', 'year': '2017', 'region': 'Deep South',
                'audio_urls': []},  # No downloadable files found yet
        'ROC': {'city': 'Rochester NY', 'year': '2016-2018', 'region': 'New York Metropolitan',
                'audio_urls': [
                    'https://lingtools.uoregon.edu/coraal/roc/2020.05/ROC_audio_part01_2020.05.tar.gz',
                    'https://lingtools.uoregon.edu/coraal/roc/2020.05/ROC_audio_part02_2020.05.tar.gz'
                ]},
        'LES': {'city': 'Lower East Side NYC', 'year': '2008', 'region': 'New York Metropolitan',
                'audio_urls': [
                    'https://lingtools.uoregon.edu/coraal/les/2021.04/LES_audio_part01_2021.04.tar.gz'
                ]},
        'DTA': {'city': 'Detroit', 'year': '1966', 'region': 'Upper Midwest',
                'audio_urls': [
                    'https://lingtools.uoregon.edu/coraal/dta/2023.06/DTA_audio_part01_2023.06.tar.gz',
                    'https://lingtools.uoregon.edu/coraal/dta/2023.06/DTA_audio_part02_2023.06.tar.gz'
                ]},
    }
    
    def download(self) -> bool:
        """Download CORAAL dataset components"""
        coraal_dir = self.cache_dir / "coraal"
        coraal_dir.mkdir(parents=True, exist_ok=True)
        
        # Components to prioritize for download (focus on weak regions)
        priority_components = ['DCA', 'DCB', 'DCB_se', 'PRV', 'ATL', 'ROC', 'LES', 'VLD']
        
        downloaded_any = False
        for component in priority_components:
            component_dir = coraal_dir / component
            audio_dir = component_dir / "audio"
            
            # Skip if already downloaded
            if audio_dir.exists() and any(audio_dir.glob("*.wav")):
                logger.info(f"CORAAL {component} already downloaded")
                continue
            
            if component not in self.CORAAL_COMPONENTS:
                continue
                
            info = self.CORAAL_COMPONENTS[component]
            logger.info(f"Downloading CORAAL {component} ({info['city']}) - {info['region']}")
            
            try:
                # Download the tar file
                tar_path = coraal_dir / f"{component}.tar"
                # Disable SSL verification for lingtools.uoregon.edu
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                response = requests.get(info['audio_urls'][0], stream=True, verify=False)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                with open(tar_path, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {component}") as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
                
                # Extract the tar file
                logger.info(f"Extracting {component}...")
                with tarfile.open(tar_path, 'r') as tar:
                    # Extract to temp dir first to handle nested structure
                    temp_dir = coraal_dir / f"temp_{component}"
                    tar.extractall(temp_dir)
                
                # Find the actual CORAAL directory and move it
                extracted_dirs = list(temp_dir.glob("CORAAL_*"))
                if extracted_dirs:
                    # Move the extracted content to the right location
                    extracted_dir = extracted_dirs[0]
                    if component_dir.exists():
                        shutil.rmtree(component_dir)
                    shutil.move(str(extracted_dir), str(component_dir))
                    shutil.rmtree(temp_dir)
                
                # Clean up tar file
                tar_path.unlink()
                downloaded_any = True
                logger.info(f"Successfully downloaded and extracted CORAAL {component}")
                
            except Exception as e:
                logger.warning(f"Failed to download CORAAL {component}: {e}")
                # Clean up failed download
                if tar_path.exists():
                    tar_path.unlink()
                continue
        
        # Check if we have at least some components
        if any(coraal_dir.glob("*/audio/*.wav")):
            logger.info(f"CORAAL dataset ready at {coraal_dir}")
            return True
        
        return downloaded_any
    
    def load(self) -> List[UnifiedSample]:
        """Load CORAAL dataset"""
        coraal_dir = self.cache_dir / "coraal"
        
        # Try to download if not present
        if not coraal_dir.exists() or not any(coraal_dir.glob("*/audio/*.wav")):
            logger.info("CORAAL dataset not found. Attempting to download...")
            if not self.download():
                logger.warning("Failed to download CORAAL dataset")
                return []
        
        unified_samples = []
        
        # Process each CORAAL component
        for component_code, info in self.CORAAL_COMPONENTS.items():
            component_dir = coraal_dir / component_code
            audio_dir = component_dir / "audio"
            
            if not audio_dir.exists():
                continue
            
            logger.info(f"Processing CORAAL component: {component_code} ({info['city']})")
            
            # Load metadata if available
            metadata_file = component_dir / f"{component_code}_metadata.txt"
            speaker_info = {}
            
            if metadata_file.exists():
                # Parse CORAAL metadata format
                with open(metadata_file, 'r') as f:
                    for line in f:
                        if '\t' in line:
                            parts = line.strip().split('\t')
                            if len(parts) >= 4:
                                speaker_id = parts[0]
                                speaker_info[speaker_id] = {
                                    'age': parts[1],
                                    'gender': parts[2],
                                    'occupation': parts[3]
                                }
            
            # Process audio files
            for audio_file in audio_dir.glob("*.wav"):
                # CORAAL naming: COMPONENT_speaker_interviewer.wav
                filename = audio_file.stem
                parts = filename.split('_')
                
                if len(parts) >= 2:
                    speaker_id = f"{component_code}_{parts[1]}"
                else:
                    speaker_id = f"{component_code}_{filename}"
                
                # Get speaker metadata
                speaker_data = speaker_info.get(parts[1], {}) if len(parts) >= 2 else {}
                
                # Create base sample with metadata
                base_sample = UnifiedSample(
                    sample_id=self._generate_sample_id('CORAAL', speaker_id, filename),
                    dataset_name='CORAAL',
                    speaker_id=speaker_id,
                    audio_path=str(audio_file),
                    transcript="",  # CORAAL provides separate transcript files
                    region_label=info['region'],
                    original_accent_label=f"{info['city']} African American English",
                    gender=speaker_data.get('gender', 'U'),
                    age=speaker_data.get('age', None),
                    is_validated=True  # CORAAL is professionally curated
                )
                
                # Add single sample - chunking will be handled by prepare_dataset.py
                unified_samples.append(base_sample)
        
        self.samples = unified_samples
        logger.info(f"Loaded {len(unified_samples)} samples from CORAAL")
        
        return unified_samples
    
    def get_dataset_info(self) -> Dict:
        """Get CORAAL dataset statistics"""
        if not self.samples:
            self.load()
        
        df = pd.DataFrame([s.to_dict() for s in self.samples])
        
        if len(df) == 0:
            return {'dataset': 'CORAAL', 'status': 'No samples loaded'}
        
        return {
            'dataset': 'CORAAL',
            'total_samples': len(self.samples),
            'total_speakers': df['speaker_id'].nunique(),
            'region_distribution': df['region_label'].value_counts().to_dict(),
            'gender_distribution': df['gender'].value_counts().to_dict(),
            'cities': df['original_accent_label'].str.extract(r'(.*) African American')[0].value_counts().to_dict()
        }


class UnifiedAccentDataset:
    """
    Main class for managing multiple accent datasets
    """
    
    def __init__(self, data_root: str, cache_dir: str = "~/.cache/accent_datasets"):
        self.data_root = Path(data_root)
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Import SAA loader if available (use Kaggle version for complete dataset)
        try:
            from loaders.saa_kaggle_loader import SAALoader
            saa_loader = SAALoader(data_root, cache_dir)
        except ImportError:
            logger.warning("SAA loader not available")
            saa_loader = None
        
        
        # Import SBCSAE loader if available
        try:
            from loaders.sbcsae_loader import SBCSAELoader
            sbcsae_loader = SBCSAELoader(data_root, cache_dir)
        except ImportError:
            logger.warning("SBCSAE loader not available")
            sbcsae_loader = None
        
        try:
            from loaders.commonvoice_loader import CommonVoiceLoader
            cv_loader = CommonVoiceLoader(data_root, cache_dir)
        except ImportError:
            logger.warning("CommonVoice loader not available")
            cv_loader = None
        
        self.loaders = {
            'TIMIT': TIMITLoader(data_root, cache_dir),
            'CORAAL': CORAALLoader(data_root, cache_dir),
        }
        
        # Add SAA loader if available
        if saa_loader:
            self.loaders['SAA'] = saa_loader
        
        # Add SBCSAE loader if available
        if sbcsae_loader:
            self.loaders['SBCSAE'] = sbcsae_loader
        
        # Add CommonVoice loader if available
        if cv_loader:
            self.loaders['CommonVoice'] = cv_loader
        
        self.all_samples: List[UnifiedSample] = []
        self.metadata_file = self.cache_dir / 'unified_metadata.json'
        
    def load_all_datasets(self, datasets: Optional[List[str]] = None) -> List[UnifiedSample]:
        """Load specified datasets or all available"""
        if datasets is None:
            datasets = list(self.loaders.keys())
            
        all_samples = []
        for dataset_name in datasets:
            if dataset_name in self.loaders:
                logger.info(f"Loading {dataset_name}...")
                samples = self.loaders[dataset_name].load()
                all_samples.extend(samples)
            else:
                logger.warning(f"Unknown dataset: {dataset_name}")
        
        self.all_samples = all_samples
        self._save_metadata()
        return all_samples
    
    def _save_metadata(self):
        """Save metadata for all loaded samples"""
        metadata = {
            'total_samples': len(self.all_samples),
            'datasets': {},
            'region_distribution': {},
            'gender_distribution': {}
        }
        
        df = pd.DataFrame([s.to_dict() for s in self.all_samples])
        
        # Overall statistics (only if samples exist)
        if len(df) > 0:
            metadata['region_distribution'] = df['region_label'].value_counts().to_dict()
            metadata['gender_distribution'] = df['gender'].value_counts().to_dict() if 'gender' in df else {}
        else:
            metadata['region_distribution'] = {}
            metadata['gender_distribution'] = {}
        
        # Per-dataset statistics
        if len(df) > 0:
            for dataset_name in df['dataset_name'].unique():
                dataset_df = df[df['dataset_name'] == dataset_name]
                metadata['datasets'][dataset_name] = {
                    'samples': len(dataset_df),
                    'speakers': dataset_df['speaker_id'].nunique(),
                    'regions': dataset_df['region_label'].value_counts().to_dict()
                }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {self.metadata_file}")
    
    def create_train_val_test_split(self, 
                                   val_ratio: float = 0.1, 
                                   test_ratio: float = 0.1,
                                   stratify_by: str = 'region_label',
                                   seed: int = 42) -> Tuple[List[UnifiedSample], List[UnifiedSample], List[UnifiedSample]]:
        """
        Create train/val/test splits ensuring no speaker overlap
        Uses weighted assignment to achieve target sample ratios
        """
        if not self.all_samples:
            raise ValueError("No samples loaded. Call load_all_datasets() first.")
        
        np.random.seed(seed)
        df = pd.DataFrame([s.to_dict() for s in self.all_samples])
        
        # Calculate target sample counts
        total_samples = len(df)
        target_test_samples = int(total_samples * test_ratio)
        target_val_samples = int(total_samples * val_ratio)
        target_train_samples = total_samples - target_test_samples - target_val_samples
        
        # Get speaker sample counts
        speaker_sample_counts = df.groupby('speaker_id').size().to_dict()
        
        # Group speakers by region for stratification
        speakers_by_region = {}
        for region in df[stratify_by].unique():
            region_df = df[df[stratify_by] == region]
            region_speakers = region_df['speaker_id'].unique().tolist()
            # Sort by sample count (descending) for better greedy assignment
            region_speakers.sort(key=lambda s: speaker_sample_counts[s], reverse=True)
            speakers_by_region[region] = region_speakers
        
        # Calculate target samples per region for each split
        region_totals = df[stratify_by].value_counts().to_dict()
        
        train_speakers, val_speakers, test_speakers = [], [], []
        train_count, val_count, test_count = 0, 0, 0
        
        # Process each region
        for region, speakers in speakers_by_region.items():
            region_total = region_totals[region]
            region_target_test = int(region_total * test_ratio)
            region_target_val = int(region_total * val_ratio)
            
            # Shuffle speakers within region for randomness
            np.random.shuffle(speakers)
            
            region_test_count = 0
            region_val_count = 0
            
            for speaker in speakers:
                speaker_samples = speaker_sample_counts[speaker]
                
                # Assign to test if we still need test samples for this region
                if region_test_count < region_target_test and test_count < target_test_samples:
                    test_speakers.append(speaker)
                    test_count += speaker_samples
                    region_test_count += speaker_samples
                # Assign to val if we still need val samples for this region
                elif region_val_count < region_target_val and val_count < target_val_samples:
                    val_speakers.append(speaker)
                    val_count += speaker_samples
                    region_val_count += speaker_samples
                # Otherwise assign to train
                else:
                    train_speakers.append(speaker)
                    train_count += speaker_samples
        
        # Create splits based on speakers
        train_samples = [s for s in self.all_samples if s.speaker_id in train_speakers]
        val_samples = [s for s in self.all_samples if s.speaker_id in val_speakers]
        test_samples = [s for s in self.all_samples if s.speaker_id in test_speakers]
        
        # Log the actual percentages achieved
        actual_train_pct = len(train_samples) / total_samples * 100
        actual_val_pct = len(val_samples) / total_samples * 100
        actual_test_pct = len(test_samples) / total_samples * 100
        
        logger.info(f"Split sizes - Train: {len(train_samples)} ({actual_train_pct:.1f}%), "
                   f"Val: {len(val_samples)} ({actual_val_pct:.1f}%), "
                   f"Test: {len(test_samples)} ({actual_test_pct:.1f}%)")
        
        # Save split information
        split_info = {
            'train_speakers': train_speakers,
            'val_speakers': val_speakers,
            'test_speakers': test_speakers,
            'split_sizes': {
                'train': len(train_samples),
                'val': len(val_samples),
                'test': len(test_samples)
            }
        }
        
        with open(self.cache_dir / 'split_info.json', 'w') as f:
            json.dump(split_info, f, indent=2)
        
        return train_samples, val_samples, test_samples
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics about the unified dataset"""
        if not self.all_samples:
            return {"error": "No samples loaded"}
        
        df = pd.DataFrame([s.to_dict() for s in self.all_samples])
        
        stats = {
            'total_samples': len(self.all_samples),
            'total_speakers': df['speaker_id'].nunique(),
            'datasets': df['dataset_name'].value_counts().to_dict(),
            'regions': df['region_label'].value_counts().to_dict(),
            'gender': df['gender'].value_counts().to_dict(),
            'samples_per_speaker': {
                'mean': df.groupby('speaker_id').size().mean(),
                'median': df.groupby('speaker_id').size().median(),
                'min': df.groupby('speaker_id').size().min(),
                'max': df.groupby('speaker_id').size().max()
            }
        }
        
        # Regional balance analysis
        region_counts = df['region_label'].value_counts()
        total = len(df)
        stats['filtered_chunk_distribution'] = {
            region: f"{(count/total*100):.1f}%" 
            for region, count in region_counts.items()
        }
        
        return stats
    
    def export_to_csv(self, output_path: str):
        """Export unified dataset to CSV for inspection"""
        if not self.all_samples:
            raise ValueError("No samples loaded")
            
        df = pd.DataFrame([s.to_dict() for s in self.all_samples])
        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(df)} samples to {output_path}")


def create_unified_dataloaders(data_root: str, 
                              processor,
                              datasets: List[str] = None,
                              batch_size: int = 16,
                              val_ratio: float = 0.1,
                              test_ratio: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders from unified dataset
    """
    # Load unified dataset
    unified_dataset = UnifiedAccentDataset(data_root)
    unified_dataset.load_all_datasets(datasets)
    
    # Create splits
    train_samples, val_samples, test_samples = unified_dataset.create_train_val_test_split(
        val_ratio=val_ratio, 
        test_ratio=test_ratio
    )
    
    # Create PyTorch datasets using OptimizedAccentDataset
    from optimized_dataset import OptimizedAccentDataset
    train_dataset = OptimizedAccentDataset(train_samples, processor, cache_size=200)
    val_dataset = OptimizedAccentDataset(val_samples, processor, cache_size=100)
    test_dataset = OptimizedAccentDataset(test_samples, processor, cache_size=100)
    
    # Create DataLoaders with custom collate function
    def unified_collate_fn(batch):
        """Custom collate function for unified dataset"""
        input_values = torch.stack([item['input_values'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        sample_ids = [item['sample_id'] for item in batch]
        
        return {
            'input_values': input_values,
            'attention_mask': attention_mask,
            'labels': labels,
            'sample_ids': sample_ids
        }
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=unified_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=unified_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=unified_collate_fn
    )
    
    # Print statistics
    stats = unified_dataset.get_statistics()
    logger.info(f"Unified dataset statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Accent Dataset Pipeline")
    parser.add_argument("--data_root", type=str, default=".", help="Root directory for datasets")
    parser.add_argument("--datasets", nargs="+", default=None, help="Datasets to load (default: all)")
    parser.add_argument("--export_csv", type=str, help="Export unified dataset to CSV")
    parser.add_argument("--stats_only", action="store_true", help="Only show statistics")
    
    args = parser.parse_args()
    
    # Create unified dataset
    unified = UnifiedAccentDataset(args.data_root)
    
    # Load datasets
    unified.load_all_datasets(args.datasets)
    
    # Show statistics
    stats = unified.get_statistics()
    print("\nUnified Dataset Statistics:")
    print("=" * 50)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Export to CSV if requested
    if args.export_csv:
        unified.export_to_csv(args.export_csv)
        print(f"\nExported to {args.export_csv}")
    
    # Create splits
    if not args.stats_only:
        train, val, test = unified.create_train_val_test_split()
        print(f"\nCreated splits - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")