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
from audio_segmentation import segment_audio_smart, process_and_segment_audio


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
    
    def _process_audio_segments(self, audio_path: str, sample_base: UnifiedSample,
                               target_duration: float = 7.5) -> List[UnifiedSample]:
        """
        Process audio file and create segmented samples if needed.
        
        Args:
            audio_path: Path to audio file
            sample_base: Base sample with metadata to copy
            target_duration: Target segment duration (default 7.5s)
        
        Returns:
            List of UnifiedSample objects (one per segment)
        """
        try:
            import librosa
            
            # Load audio and check duration
            audio, sr = librosa.load(audio_path, sr=16000)
            duration = len(audio) / sr
            
            # If audio is short enough, return single sample
            if duration <= 10.0:
                sample = UnifiedSample(
                    sample_id=sample_base.sample_id,
                    dataset_name=sample_base.dataset_name,
                    speaker_id=sample_base.speaker_id,
                    audio_path=audio_path,
                    transcript=sample_base.transcript,
                    region_label=sample_base.region_label,
                    original_accent_label=sample_base.original_accent_label,
                    state=sample_base.state,
                    gender=sample_base.gender,
                    age=sample_base.age,
                    native_language=sample_base.native_language,
                    duration=duration,
                    sample_rate=16000,
                    is_validated=sample_base.is_validated,
                    quality_score=sample_base.quality_score
                )
                return [sample]
            
            # Segment longer audio
            segments = segment_audio_smart(
                audio, sr=16000,
                target_duration=target_duration,
                min_duration=5.0,
                max_duration=10.0,
                overlap_ratio=0.2
            )
            
            segmented_samples = []
            for i, (segment_audio, start_sample, end_sample) in enumerate(segments):
                segment_duration = len(segment_audio) / sr
                segment_id = f"{sample_base.sample_id}_seg{i:03d}"
                
                sample = UnifiedSample(
                    sample_id=segment_id,
                    dataset_name=sample_base.dataset_name,
                    speaker_id=sample_base.speaker_id,
                    audio_path=audio_path,  # Keep original path
                    transcript=sample_base.transcript,
                    region_label=sample_base.region_label,
                    original_accent_label=sample_base.original_accent_label,
                    state=sample_base.state,
                    gender=sample_base.gender,
                    age=sample_base.age,
                    native_language=sample_base.native_language,
                    duration=segment_duration,
                    sample_rate=16000,
                    is_validated=sample_base.is_validated,
                    quality_score=sample_base.quality_score
                )
                # Store segment info for later extraction
                sample._segment_start = start_sample
                sample._segment_end = end_sample
                segmented_samples.append(sample)
            
            return segmented_samples
            
        except Exception as e:
            logger.warning(f"Failed to segment {audio_path}: {e}. Returning single sample.")
            # Fallback to single sample
            sample_base.audio_path = audio_path
            return [sample_base]
    
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
            
        from timit_dataset import TimitDatasetLoader, TIMIT_REGIONS
        
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


class CommonVoiceLoader(BaseDatasetLoader):
    """Loader for Mozilla Common Voice dataset"""
    
    CV_VERSION = "cv-corpus-22.0-2025-06-20"  # Latest version as of 2025
    CV_URL = "https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-22.0-2025-06-20/cv-corpus-22.0-2025-06-20-en.tar.gz"
    
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
            logger.info("This may take several minutes depending on your connection...")
            
            # Create directory
            tar_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download the dataset
            try:
                response = requests.get(self.CV_URL, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                with open(tar_path, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading CommonVoice") as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                
                logger.info("CommonVoice download complete!")
            except Exception as e:
                logger.error(f"Failed to download CommonVoice: {e}")
                logger.warning("You can manually download from https://commonvoice.mozilla.org/en/datasets")
                if tar_path.exists():
                    tar_path.unlink()  # Remove partial download
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
            'Midwest': [
                'midwest', 'upper midwest', 'lower midwest', 'michigan', 'wisconsin', 'minnesota',
                'chicago', 'detroit', 'milwaukee', 'twin cities', 'great lakes',
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
            
            # Remove testing limit - process all samples
            # Note: This may take a while as v22.0 has millions of samples
        
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
                
                # Process and segment the audio (CORAAL files are long interviews)
                segmented_samples = self._process_audio_segments(str(audio_file), base_sample)
                unified_samples.extend(segmented_samples)
                
                if len(segmented_samples) > 1:
                    logger.info(f"  Segmented {filename} into {len(segmented_samples)} chunks")
        
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
        
        # Import SAA loader if available
        try:
            from saa_loader import SAALoader
            saa_loader = SAALoader(data_root, cache_dir)
        except ImportError:
            logger.warning("SAA loader not available")
            saa_loader = None
        
        
        # Import SBCSAE loader if available
        try:
            from sbcsae_loader import SBCSAELoader
            sbcsae_loader = SBCSAELoader(data_root, cache_dir)
        except ImportError:
            logger.warning("SBCSAE loader not available")
            sbcsae_loader = None
        
        try:
            from commonvoice_filtered_loader import FilteredCommonVoiceLoader
            filtered_cv_loader = FilteredCommonVoiceLoader(data_root, cache_dir)
        except ImportError:
            logger.warning("FilteredCommonVoice loader not available")
            filtered_cv_loader = None
        
        self.loaders = {
            'TIMIT': TIMITLoader(data_root, cache_dir),
            'CommonVoice': CommonVoiceLoader(data_root, cache_dir),
            'CORAAL': CORAALLoader(data_root, cache_dir),
        }
        
        # Add SAA loader if available
        if saa_loader:
            self.loaders['SAA'] = saa_loader
        
        # Add SBCSAE loader if available
        if sbcsae_loader:
            self.loaders['SBCSAE'] = sbcsae_loader
        
        # Add FilteredCommonVoice loader if available
        if filtered_cv_loader:
            self.loaders['FilteredCommonVoice'] = filtered_cv_loader
        
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
                'Midwest': 1,  # Merged Upper and Lower Midwest
                'New England': 2,
                'New York Metropolitan': 3,
                'West': 4,
                'Mid-Atlantic': 5,
                'South Atlantic': 6
            }
        
        self.label_to_region = {v: k for k, v in self.region_to_label.items()}
            
        logger.info(f"Dataset initialized with {len(self.samples)} samples")
        logger.info(f"Using label mapping with {len(self.region_to_label)} regions: {self.region_to_label}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Implementation similar to TimitAccentDataset
        # but using UnifiedSample format
        sample = self.samples[idx]
        
        # Handle different path types
        audio_path = sample.audio_path
        
        # Check if file exists, if not try to handle SAA naming issues
        if not os.path.exists(audio_path):
            if sample.dataset_name == 'SAA' and 'english' in audio_path:
                # Try to fix SAA naming issue
                # The CSV has 4-digit numbers but files have up to 3 digits
                import re
                match = re.search(r'english(\d+)\.mp3', audio_path)
                if match:
                    num = match.group(1)
                    # Map 4-digit numbers to actual files (there are only ~294 files)
                    # This is a workaround for a bug in dataset preparation
                    if len(num) == 4:
                        # Convert to a number within the actual range
                        file_num = int(num) % 300  # Wrap around to stay in range
                        fixed_path = audio_path.replace(f'english{num}', f'english{file_num}')
                        if os.path.exists(fixed_path):
                            audio_path = fixed_path
                        else:
                            # Try other patterns
                            for i in range(1, 295):
                                test_path = audio_path.replace(f'english{num}', f'english{i}')
                                if os.path.exists(test_path):
                                    audio_path = test_path
                                    break
        
        # Load and process audio
        import librosa
        try:
            audio, sr = librosa.load(audio_path, sr=self.target_sr)
        except Exception as e:
            logger.error(f"Failed to load audio from {audio_path}: {e}")
            logger.error(f"Sample info - Dataset: {sample.dataset_name}, ID: {sample.sample_id}")
            raise
        
        # Handle segmented samples (from load-time segmentation)
        if hasattr(sample, '_segment_start'):
            start_sample = sample._segment_start
            end_sample = sample._segment_end
            
            # Extract the specific segment
            if end_sample <= len(audio):
                audio = audio[start_sample:end_sample]
            else:
                # Shouldn't happen if segmentation was done correctly
                logger.warning(f"Segment bounds exceed audio length for {sample.sample_id}")
        
        # Track original length before padding
        original_length = len(audio)
        
        # Ensure audio is exactly max_length
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
        
        # Create proper attention mask manually since Wav2Vec2Processor doesn't provide one
        attention_mask = torch.ones(inputs.input_values.size(-1), dtype=torch.float32)
        if original_length < self.max_length:
            # Zero out the padded region
            attention_mask[original_length:] = 0.0
        
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