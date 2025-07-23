#!/usr/bin/env python3
"""
TIMIT Dataset Loader Module
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging

# Set up logging
logger = logging.getLogger(__name__)

# TIMIT dialect region mapping
TIMIT_REGIONS = {
    'dr1': 'New England',
    'dr2': 'Northern',
    'dr3': 'North Midland',
    'dr4': 'South Midland',
    'dr5': 'Southern',
    'dr6': 'New York City',
    'dr7': 'Western',
    'dr8': 'Army Brat (moved around)'
}


@dataclass
class TimitSample:
    """Represents a single TIMIT sample"""
    speaker_id: str
    sentence_id: str
    audio_path: str
    transcript: str
    dialect_region: str
    gender: str
    split: str  # 'TRAIN' or 'TEST'


class TimitDatasetLoader:
    """Loader for TIMIT dataset"""
    
    def __init__(self, data_root: str):
        """
        Initialize TIMIT loader
        
        Args:
            data_root: Path to TIMIT dataset root (containing TRAIN and TEST directories)
        """
        self.data_root = Path(data_root)
        self.speaker_info = self._load_speaker_info()
        
    def _load_speaker_info(self) -> Dict[str, Dict]:
        """Load speaker information from SPKRINFO.TXT"""
        speaker_info = {}
        
        # Look for SPKRINFO.TXT in parent directory
        spkr_info_path = self.data_root.parent / 'SPKRINFO.TXT'
        if not spkr_info_path.exists():
            # Try in the data_root directory
            spkr_info_path = self.data_root / 'SPKRINFO.TXT'
            
        if not spkr_info_path.exists():
            logger.warning(f"SPKRINFO.TXT not found in {self.data_root.parent} or {self.data_root}")
            return speaker_info
            
        with open(spkr_info_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith(';'):
                    continue
                    
                # Parse speaker info line
                parts = line.split()
                if len(parts) >= 4:
                    speaker_id = parts[0]
                    gender = parts[1]
                    dialect_region = f"dr{parts[2]}"
                    usage = parts[3]
                    
                    speaker_info[speaker_id] = {
                        'gender': gender,
                        'dialect_region': dialect_region,
                        'usage': usage
                    }
                    
        logger.info(f"Loaded {len(speaker_info)} speakers from SPKRINFO.TXT")
        return speaker_info
        
    def load_dataset(self) -> List[TimitSample]:
        """Load all TIMIT samples from TRAIN and TEST directories"""
        samples = []
        
        # Load from both TRAIN and TEST directories
        for split in ['TRAIN', 'TEST']:
            split_dir = self.data_root / split
            if not split_dir.exists():
                logger.warning(f"{split} directory not found at {split_dir}")
                continue
                
            # Iterate through dialect regions (DR1-DR8)
            for dr_dir in split_dir.glob('DR*'):
                if not dr_dir.is_dir():
                    continue
                    
                dialect_region = dr_dir.name.lower()  # e.g., 'dr1'
                
                # Iterate through speakers
                for speaker_dir in dr_dir.iterdir():
                    if not speaker_dir.is_dir():
                        continue
                        
                    speaker_id = speaker_dir.name
                    
                    # Get speaker info
                    gender = 'M'  # Default
                    if speaker_id.upper() in self.speaker_info:
                        gender = self.speaker_info[speaker_id.upper()]['gender']
                    elif speaker_id in self.speaker_info:
                        gender = self.speaker_info[speaker_id]['gender']
                    
                    # Load all .WAV files in speaker directory
                    for wav_file in speaker_dir.glob('*.WAV'):
                        # Skip .WAV.wav files (duplicates)
                        if wav_file.suffix == '.wav' and wav_file.stem.endswith('.WAV'):
                            continue
                            
                        sentence_id = wav_file.stem
                        
                        # Load transcript
                        txt_file = wav_file.with_suffix('.TXT')
                        transcript = ""
                        if txt_file.exists():
                            with open(txt_file, 'r') as f:
                                # TIMIT transcripts have format: "0 46797 She had your dark suit in greasy wash water all year."
                                line = f.readline().strip()
                                parts = line.split(maxsplit=2)
                                if len(parts) >= 3:
                                    transcript = parts[2]
                                    
                        sample = TimitSample(
                            speaker_id=speaker_id,
                            sentence_id=sentence_id,
                            audio_path=str(wav_file),
                            transcript=transcript,
                            dialect_region=dialect_region,
                            gender=gender,
                            split=split
                        )
                        samples.append(sample)
                        
        logger.info(f"Loaded {len(samples)} samples from TIMIT dataset")
        return samples