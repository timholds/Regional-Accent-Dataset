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
            'detroit': 'Upper Midwest',
            'new york': 'New York Metropolitan',
            'nyc': 'New York Metropolitan',
            'atlanta': 'Deep South',
            'washington': 'Mid-Atlantic',
            'dc': 'Mid-Atlantic',
            'boston': 'New England',
            'chicago': 'Upper Midwest',
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
                'Upper Midwest': ['midwest', 'upper midwest', 'michigan', 'wisconsin', 'minnesota', 'chicago'],
                'Lower Midwest': ['ohio', 'indiana', 'missouri', 'iowa'],
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
        timit_path = self.data_root
        
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
            
        from timit_dataset import TimitDatasetLoader, TIMIT_REGIONS
        
        logger.info("Loading TIMIT dataset...")
        loader = TimitDatasetLoader(str(self.data_root))
        timit_samples = loader.load_dataset()
        
        # Convert to unified format
        unified_samples = []
        for ts in timit_samples:
            # Map TIMIT dialect region to our classification
            region_name = TIMIT_REGIONS.get(ts.dialect_region, 'Unknown')
            
            # Map TIMIT regions to our 8-region system
            timit_to_our_regions = {
                'New England': 'New England',
                'Northern': 'Upper Midwest',
                'North Midland': 'Lower Midwest',
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


class CommonVoiceLoader(BaseDatasetLoader):
    """Loader for Mozilla Common Voice dataset"""
    
    CV_VERSION = "cv-corpus-17.0-2024-03-15"  # Latest version as of 2024
    CV_URL = "https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-17.0-2024-03-15/cv-corpus-17.0-2024-03-15-en.tar.gz"
    
    def download(self) -> bool:
        """Download Common Voice dataset if not cached"""
        cv_dir = self.cache_dir / "common_voice" / self.CV_VERSION
        
        if cv_dir.exists() and len(list(cv_dir.glob("*.tsv"))) > 0:
            logger.info(f"Common Voice dataset found at {cv_dir}")
            return True
        
        # Check if we need to download
        tar_path = self.cache_dir / "common_voice" / f"{self.CV_VERSION}-en.tar.gz"
        
        if not tar_path.exists():
            logger.info(f"Downloading Common Voice dataset (~2.5GB)...")
            logger.info("Note: You can also download manually from https://commonvoice.mozilla.org/en/datasets")
            
            # Create directory
            tar_path.parent.mkdir(parents=True, exist_ok=True)
            
            # For now, we'll skip actual download and assume user downloads manually
            logger.warning("Please download Common Voice English dataset manually from:")
            logger.warning("https://commonvoice.mozilla.org/en/datasets")
            logger.warning(f"Place the tar.gz file at: {tar_path}")
            return False
        
        # Extract if needed
        if not cv_dir.exists():
            logger.info(f"Extracting Common Voice dataset...")
            cv_dir.mkdir(parents=True, exist_ok=True)
            
            with tarfile.open(tar_path, 'r:gz') as tar:
                # Extract only the English subset
                for member in tar.getmembers():
                    if member.name.startswith(f"{self.CV_VERSION}/en/"):
                        member.name = member.name.replace(f"{self.CV_VERSION}/en/", "")
                        tar.extract(member, cv_dir)
            
            logger.info("Extraction complete")
        
        return True
    
    def _parse_accent_label(self, accent_str: str) -> Tuple[str, str]:
        """Parse Common Voice accent labels to our regions"""
        if not accent_str or pd.isna(accent_str):
            return 'West', 'Unknown'
        
        accent_lower = accent_str.lower()
        
        # US-specific accent patterns
        us_accent_patterns = {
            'New England': [
                'new england', 'boston', 'maine', 'vermont', 'new hampshire',
                'massachusetts', 'rhode island', 'connecticut'
            ],
            'New York Metropolitan': [
                'new york', 'nyc', 'brooklyn', 'bronx', 'manhattan', 'queens',
                'long island', 'new jersey', 'newark'
            ],
            'Mid-Atlantic': [
                'mid-atlantic', 'mid atlantic', 'philadelphia', 'baltimore',
                'washington', 'dc', 'maryland', 'delaware'
            ],
            'South Atlantic': [
                'virginia', 'north carolina', 'south carolina', 'georgia', 
                'florida', 'charleston', 'richmond'
            ],
            'Deep South': [
                'southern', 'deep south', 'alabama', 'mississippi', 'louisiana',
                'arkansas', 'tennessee', 'kentucky', 'atlanta', 'nashville',
                'new orleans', 'memphis'
            ],
            'Upper Midwest': [
                'midwest', 'upper midwest', 'michigan', 'wisconsin', 'minnesota',
                'chicago', 'detroit', 'milwaukee', 'twin cities', 'great lakes'
            ],
            'Lower Midwest': [
                'ohio', 'indiana', 'illinois', 'missouri', 'iowa', 'nebraska',
                'kansas', 'st louis', 'cincinnati', 'indianapolis'
            ],
            'West': [
                'western', 'west coast', 'california', 'pacific', 'texas',
                'colorado', 'arizona', 'nevada', 'utah', 'new mexico',
                'washington', 'oregon', 'seattle', 'portland', 'los angeles',
                'san francisco', 'denver', 'phoenix', 'las vegas'
            ]
        }
        
        # Check for US patterns
        for region, patterns in us_accent_patterns.items():
            if any(pattern in accent_lower for pattern in patterns):
                return region, accent_str
        
        # Check if it's US English but region unspecified
        us_indicators = ['united states', 'american', 'us english', 'usa']
        if any(ind in accent_lower for ind in us_indicators):
            return 'West', accent_str  # Default to West for unspecified US
        
        # Not US English
        return None, accent_str
    
    def load(self) -> List[UnifiedSample]:
        """Load Common Voice dataset"""
        if not self.download():
            return []
        
        cv_dir = self.cache_dir / "common_voice" / self.CV_VERSION
        validated_tsv = cv_dir / "validated.tsv"
        
        if not validated_tsv.exists():
            logger.error(f"Validated.tsv not found at {validated_tsv}")
            return []
        
        logger.info("Loading Common Voice metadata...")
        df = pd.read_csv(validated_tsv, sep='\t')
        
        # Filter for US accents only
        logger.info(f"Total Common Voice samples: {len(df)}")
        
        unified_samples = []
        clips_dir = cv_dir / "clips"
        
        # Process samples with US accents
        us_samples = 0
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Common Voice"):
            accent = row.get('accent', '')
            region, original_accent = self._parse_accent_label(accent)
            
            if region is None:  # Skip non-US accents
                continue
            
            # Build audio path
            audio_path = clips_dir / row['path']
            if not audio_path.exists():
                audio_path = clips_dir / row['path'].replace('.mp3', '.wav')
                if not audio_path.exists():
                    continue
            
            sample = UnifiedSample(
                sample_id=self._generate_sample_id('CommonVoice', row['client_id'], row['path']),
                dataset_name='CommonVoice',
                speaker_id=row['client_id'],
                audio_path=str(audio_path),
                transcript=row['sentence'],
                region_label=region,
                original_accent_label=original_accent,
                gender=row.get('gender', 'U'),
                age=row.get('age', None),
                duration=row.get('duration', None),
                is_validated=True  # We're using validated.tsv
            )
            
            unified_samples.append(sample)
            us_samples += 1
            
            # Limit for testing (remove in production)
            if us_samples >= 1000:  # Process first 1000 US samples for testing
                break
        
        self.samples = unified_samples
        logger.info(f"Loaded {len(unified_samples)} US samples from Common Voice")
        
        return unified_samples
    
    def get_dataset_info(self) -> Dict:
        """Get Common Voice dataset statistics"""
        if not self.samples:
            self.load()
        
        df = pd.DataFrame([s.to_dict() for s in self.samples])
        
        if len(df) == 0:
            return {'dataset': 'CommonVoice', 'status': 'No samples loaded'}
        
        return {
            'dataset': 'CommonVoice',
            'total_samples': len(self.samples),
            'total_speakers': df['speaker_id'].nunique(),
            'region_distribution': df['region_label'].value_counts().to_dict(),
            'gender_distribution': df['gender'].value_counts().to_dict(),
            'total_duration_hours': df['duration'].sum() / 3600 if 'duration' in df else 0
        }


class CORAALLoader(BaseDatasetLoader):
    """Loader for CORAAL (Corpus of Regional African American Language)"""
    
    CORAAL_COMPONENTS = {
        'DCA': {'city': 'Washington DC', 'year': '1968', 'region': 'Mid-Atlantic'},
        'DCB': {'city': 'Washington DC', 'year': '2016', 'region': 'Mid-Atlantic'},
        'ATL': {'city': 'Atlanta', 'year': '2017', 'region': 'Deep South'},
        'PRV': {'city': 'Princeville NC', 'year': '2004', 'region': 'South Atlantic'},
        'VLD': {'city': 'Valdosta GA', 'year': '2017', 'region': 'Deep South'},
        'ROC': {'city': 'Rochester NY', 'year': '2016-2018', 'region': 'New York Metropolitan'},
        'LES': {'city': 'Lower East Side NYC', 'year': '2008', 'region': 'New York Metropolitan'},
        'DCB_se': {'city': 'Washington DC', 'year': '2018', 'region': 'Mid-Atlantic'},
    }
    
    def download(self) -> bool:
        """Download CORAAL dataset components"""
        coraal_dir = self.cache_dir / "coraal"
        
        # Check if already downloaded
        if coraal_dir.exists() and any(coraal_dir.glob("*/audio/*.wav")):
            logger.info(f"CORAAL dataset found at {coraal_dir}")
            return True
        
        logger.info("CORAAL download instructions:")
        logger.info("1. Visit http://lingtools.uoregon.edu/coraal/")
        logger.info("2. Download the components you want (ATL, DCA, LES, ROC recommended)")
        logger.info("3. Extract each component to: " + str(coraal_dir))
        logger.info("4. Directory structure should be: coraal/COMPONENT/audio/*.wav")
        
        # For now, return False and let user download manually
        return False
    
    def load(self) -> List[UnifiedSample]:
        """Load CORAAL dataset"""
        coraal_dir = self.cache_dir / "coraal"
        
        if not coraal_dir.exists():
            logger.warning("CORAAL dataset not found. Please download it first.")
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
                
                sample = UnifiedSample(
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
                
                unified_samples.append(sample)
        
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
        
        self.loaders = {
            'TIMIT': TIMITLoader(data_root, cache_dir),
            'CommonVoice': CommonVoiceLoader(data_root, cache_dir),
            'CORAAL': CORAALLoader(data_root, cache_dir),
        }
        
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
        
        # Overall statistics
        metadata['region_distribution'] = df['region_label'].value_counts().to_dict()
        metadata['gender_distribution'] = df['gender'].value_counts().to_dict()
        
        # Per-dataset statistics
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
        """
        if not self.all_samples:
            raise ValueError("No samples loaded. Call load_all_datasets() first.")
        
        np.random.seed(seed)
        df = pd.DataFrame([s.to_dict() for s in self.all_samples])
        
        # Get unique speakers per region
        speakers_by_region = {}
        for region in df[stratify_by].unique():
            region_df = df[df[stratify_by] == region]
            speakers_by_region[region] = region_df['speaker_id'].unique().tolist()
        
        train_speakers, val_speakers, test_speakers = [], [], []
        
        # Split speakers per region to maintain distribution
        for region, speakers in speakers_by_region.items():
            np.random.shuffle(speakers)
            n_speakers = len(speakers)
            n_val = int(n_speakers * val_ratio)
            n_test = int(n_speakers * test_ratio)
            
            test_speakers.extend(speakers[:n_test])
            val_speakers.extend(speakers[n_test:n_test + n_val])
            train_speakers.extend(speakers[n_test + n_val:])
        
        # Create splits based on speakers
        train_samples = [s for s in self.all_samples if s.speaker_id in train_speakers]
        val_samples = [s for s in self.all_samples if s.speaker_id in val_speakers]
        test_samples = [s for s in self.all_samples if s.speaker_id in test_speakers]
        
        logger.info(f"Split sizes - Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
        
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
        stats['region_percentages'] = {
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


class UnifiedAccentDatasetTorch(Dataset):
    """PyTorch Dataset wrapper for unified samples"""
    
    def __init__(self, samples: List[UnifiedSample], processor, 
                 target_sr: int = 16000, max_length: int = 160000, 
                 label_mapping: Optional[Dict[str, int]] = None):
        self.samples = samples
        self.processor = processor
        self.target_sr = target_sr
        self.max_length = max_length
        
        # Use provided label mapping or create from all possible regions
        if label_mapping is not None:
            self.region_to_label = label_mapping
        else:
            # Default mapping for all expected regions
            self.region_to_label = {
                'Deep South': 0,
                'Lower Midwest': 1,
                'New England': 2,
                'New York Metropolitan': 3,
                'Upper Midwest': 4,
                'West': 5,
                'Mid-Atlantic': 6,
                'South Atlantic': 7
            }
        
        self.label_to_region = {v: k for k, v in self.region_to_label.items()}
            
        logger.info(f"Using label mapping with {len(self.region_to_label)} regions: {self.region_to_label}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Implementation similar to TimitAccentDataset
        # but using UnifiedSample format
        sample = self.samples[idx]
        
        # Load and process audio
        import librosa
        audio, sr = librosa.load(sample.audio_path, sr=self.target_sr)
        
        # Track original length before padding
        original_length = len(audio)
        
        # Pad or truncate
        if len(audio) > self.max_length:
            audio = audio[:self.max_length]
            original_length = self.max_length
        elif len(audio) < self.max_length:
            audio = np.pad(audio, (0, self.max_length - len(audio)))
        
        # Process with model processor
        inputs = self.processor(
            audio, 
            sampling_rate=self.target_sr, 
            return_tensors="pt",
            padding=False  # We already padded
        )
        
        # Create proper attention mask (1 for real audio, 0 for padding)
        attention_mask = torch.zeros(self.max_length, dtype=torch.float32)
        attention_mask[:original_length] = 1.0
        
        label = self.region_to_label[sample.region_label]
        
        return {
            'input_values': inputs.input_values.squeeze(),
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long),
            'sample_id': sample.sample_id
        }


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
    
    # Create PyTorch datasets
    train_dataset = UnifiedAccentDatasetTorch(train_samples, processor)
    val_dataset = UnifiedAccentDatasetTorch(val_samples, processor)
    test_dataset = UnifiedAccentDatasetTorch(test_samples, processor)
    
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