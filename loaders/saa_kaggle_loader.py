"""
Speech Accent Archive (SAA) Loader - Kaggle Complete Dataset Version

Loads the complete SAA dataset from Kaggle with 2,138 samples.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, List
import logging
import re

from unified_dataset import BaseDatasetLoader, UnifiedSample
from region_mappings import get_region_for_state

logger = logging.getLogger(__name__)


class SAALoader(BaseDatasetLoader):
    """Loader for complete Speech Accent Archive from Kaggle"""
    
    def download(self) -> bool:
        """Check if SAA Kaggle data exists"""
        saa_dir = self.cache_dir / "saa"
        kaggle_csv = saa_dir / "kaggle_metadata.csv"
        audio_dir = saa_dir / "audio"
        
        if not kaggle_csv.exists():
            logger.warning(f"Kaggle SAA metadata not found at {kaggle_csv}")
            logger.info("Run: kaggle datasets download -d rtatman/speech-accent-archive")
            return False
        
        if not audio_dir.exists() or len(list(audio_dir.glob("*.mp3"))) == 0:
            logger.warning(f"No audio files found in {audio_dir}")
            return False
            
        logger.info(f"Found Kaggle SAA dataset at {saa_dir}")
        return True
    
    def load(self) -> List[UnifiedSample]:
        """Load SAA dataset from Kaggle data"""
        if not self.download():
            return []
            
        saa_dir = self.cache_dir / "saa"
        kaggle_csv = saa_dir / "kaggle_metadata.csv"
        audio_dir = saa_dir / "audio"
        
        # Load Kaggle metadata
        logger.info(f"Loading Kaggle SAA data from {kaggle_csv}")
        df = pd.read_csv(kaggle_csv)
        
        # Filter for samples with audio files
        df_with_audio = df[df['file_missing?'] == False].copy()
        logger.info(f"Found {len(df_with_audio)} samples with audio out of {len(df)} total")
        
        # Filter for US samples only
        us_df = df_with_audio[df_with_audio['country'] == 'usa'].copy()
        logger.info(f"Found {len(us_df)} US speakers with audio")
        
        # Extract state from birthplace (format: "city, state, usa")
        def extract_state(birthplace):
            if pd.isna(birthplace):
                return None
            # Match pattern like "miami, florida, usa"
            match = re.search(r',\s*([a-z\s]+),\s*usa', birthplace.lower())
            if match:
                state_name = match.group(1).strip()
                # Convert state name to abbreviation
                state_map = {
                    'alabama': 'AL', 'alaska': 'AK', 'arizona': 'AZ', 'arkansas': 'AR',
                    'california': 'CA', 'colorado': 'CO', 'connecticut': 'CT',
                    'delaware': 'DE', 'district of columbia': 'DC', 'florida': 'FL',
                    'georgia': 'GA', 'hawaii': 'HI', 'idaho': 'ID', 'illinois': 'IL',
                    'indiana': 'IN', 'iowa': 'IA', 'kansas': 'KS', 'kentucky': 'KY',
                    'louisiana': 'LA', 'maine': 'ME', 'maryland': 'MD',
                    'massachusetts': 'MA', 'michigan': 'MI', 'minnesota': 'MN',
                    'mississippi': 'MS', 'missouri': 'MO', 'montana': 'MT',
                    'nebraska': 'NE', 'nevada': 'NV', 'new hampshire': 'NH',
                    'new jersey': 'NJ', 'new mexico': 'NM', 'new york': 'NY',
                    'north carolina': 'NC', 'north dakota': 'ND', 'ohio': 'OH',
                    'oklahoma': 'OK', 'oregon': 'OR', 'pennsylvania': 'PA',
                    'rhode island': 'RI', 'south carolina': 'SC', 'south dakota': 'SD',
                    'tennessee': 'TN', 'texas': 'TX', 'utah': 'UT', 'vermont': 'VT',
                    'virginia': 'VA', 'washington': 'WA', 'west virginia': 'WV',
                    'wisconsin': 'WI', 'wyoming': 'WY'
                }
                return state_map.get(state_name, None)
            return None
        
        us_df['state_abbr'] = us_df['birthplace'].apply(extract_state)
        
        # Map states to regions using canonical mappings
        us_df['region_mapped'] = us_df['state_abbr'].apply(
            lambda state: get_region_for_state(state, classification='medium') if state else 'Unknown'
        )
        
        # Show distribution
        region_counts = us_df['region_mapped'].value_counts()
        logger.info("SAA Kaggle Regional distribution:")
        for region, count in region_counts.items():
            if region != 'Unknown':
                logger.info(f"  {region}: {count}")
        
        # Count unknowns
        unknown_count = len(us_df[us_df['region_mapped'] == 'Unknown'])
        if unknown_count > 0:
            logger.warning(f"  {unknown_count} samples with unknown/unmapped regions")
        
        unified_samples = []
        
        # Common SAA transcript (all speakers read the same passage)
        saa_transcript = ("Please call Stella. Ask her to bring these things with her " + 
                         "from the store: Six spoons of fresh snow peas, five thick " + 
                         "slabs of blue cheese, and maybe a snack for her brother Bob.")
        
        # Process each US speaker with valid audio
        samples_loaded = 0
        samples_skipped = 0
        
        for idx, row in us_df.iterrows():
            filename = row['filename']
            
            # Check if audio file exists (filename might already have .mp3 or not)
            if filename.endswith('.mp3'):
                audio_file = audio_dir / filename
            else:
                audio_file = audio_dir / f"{filename}.mp3"
            if not audio_file.exists() or audio_file.stat().st_size < 1000:
                samples_skipped += 1
                continue
            
            # Map gender format
            gender_map = {'male': 'M', 'female': 'F'}
            gender = gender_map.get(row.get('sex', '').lower(), 'U')
            
            # Create unified sample
            sample = UnifiedSample(
                sample_id=self._generate_sample_id('SAA', filename, 'full'),
                dataset_name='SAA',
                speaker_id=str(row.get('speakerid', filename)),
                audio_path=str(audio_file),
                transcript=saa_transcript,
                region_label=row['region_mapped'],
                original_accent_label=f"{row.get('birthplace', 'Unknown')} - {row.get('native_language', 'English')}",
                state=row.get('state_abbr'),
                gender=gender,
                age=row.get('age'),
                native_language=row.get('native_language', 'english'),
                is_validated=True
            )
            
            unified_samples.append(sample)
            samples_loaded += 1
        
        self.samples = unified_samples
        logger.info(f"Loaded {samples_loaded} SAA samples from Kaggle dataset")
        if samples_skipped > 0:
            logger.info(f"Skipped {samples_skipped} samples without audio files")
        
        return unified_samples
    
    def get_dataset_info(self) -> Dict:
        """Get SAA dataset statistics"""
        if not self.samples:
            self.load()
        
        if not self.samples:
            return {'dataset': 'SAA', 'status': 'No samples loaded'}
        
        df = pd.DataFrame([s.to_dict() for s in self.samples])
        
        return {
            'dataset': 'SAA',
            'total_samples': len(self.samples),
            'total_speakers': df['speaker_id'].nunique(),
            'region_distribution': df['region_label'].value_counts().to_dict(),
            'state_distribution': df['state'].value_counts().to_dict() if 'state' in df else {},
            'native_language_distribution': df['native_language'].value_counts().to_dict() if 'native_language' in df else {},
            'description': 'Complete Speech Accent Archive from Kaggle (2,138 samples)'
        }


if __name__ == "__main__":
    # Test the loader
    import logging
    logging.basicConfig(level=logging.INFO)
    
    loader = SAALoader('.', '~/.cache/accent_datasets')
    samples = loader.load()
    info = loader.get_dataset_info()
    
    print("\nSAA Kaggle Dataset Info:")
    for key, value in info.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")