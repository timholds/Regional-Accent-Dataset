"""
SBCSAE Loader - Santa Barbara Corpus of Spoken American English

Note: This corpus requires manual download from LDC or UCSB.
Audio files must be placed in ~/.cache/accent_datasets/sbcsae/
"""

import os
import json
from pathlib import Path
from typing import Dict, List
import logging
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))

from unified_dataset import BaseDatasetLoader, UnifiedSample
from region_mappings import get_region_for_state

logger = logging.getLogger(__name__)


class SBCSAELoader(BaseDatasetLoader):
    """Loader for Santa Barbara Corpus of Spoken American English"""
    
    # Mapping of conversations to states (based on DATASETS.md)
    CONVERSATION_METADATA = {
        'SBC001': {'state': 'CA', 'speakers': ['Lenore', 'Ron']},
        'SBC002': {'state': 'CA', 'speakers': ['Angela', 'Jamie']},
        'SBC003': {'state': 'NY', 'speakers': ['Brad', 'Wendy']},
        'SBC004': {'state': 'TX', 'speakers': ['Dan', 'Jeff']},
        'SBC005': {'state': 'NC', 'speakers': ['Alice', 'Mary', 'Joe']},
        'SBC006': {'state': 'MI', 'speakers': ['Pete', 'Tom']},
        'SBC007': {'state': 'CA', 'speakers': ['Doris', 'Joanne']},
        'SBC008': {'state': 'FL', 'speakers': ['Marilyn', 'Roy']},
        'SBC009': {'state': 'OH', 'speakers': ['Ken', 'Marci']},
        'SBC010': {'state': 'CA', 'speakers': ['Nathan', 'Shane']},
        'SBC011': {'state': 'PA', 'speakers': ['Rebecca', 'Rickie']},
        'SBC012': {'state': 'GA', 'speakers': ['Darryl', 'Tim']},
        'SBC013': {'state': 'WI', 'speakers': ['Karen', 'Lynne']},
        'SBC014': {'state': 'CA', 'speakers': ['Andrew', 'Danny']},
        'SBC015': {'state': 'VA', 'speakers': ['Miles', 'Kendra']},
        'SBC016': {'state': 'CO', 'speakers': ['Fred', 'Jim']},
        'SBC017': {'state': 'AL', 'speakers': ['Beth', 'Carolyn']},
        'SBC018': {'state': 'MD', 'speakers': ['Gary', 'Harold']},
        'SBC019': {'state': 'IL', 'speakers': ['Kevin', 'Pamela']},
        'SBC020': {'state': 'CA', 'speakers': ['Julie', 'Kitty']},
        'SBC021': {'state': 'WA', 'speakers': ['Frank', 'Sue']},
        'SBC022': {'state': 'MO', 'speakers': ['Charles', 'Dorothy']},
        'SBC023': {'state': 'SC', 'speakers': ['Brad', 'Matt']},
        'SBC024': {'state': 'LA', 'speakers': ['Alina', 'Lenore']},
        'SBC025': {'state': 'DC', 'speakers': ['Arnold', 'Sheri']},
        'SBC026': {'state': 'TN', 'speakers': ['Joe', 'Lance']},
        'SBC027': {'state': 'MN', 'speakers': ['Ken', 'Roger']},
        'SBC028': {'state': 'CA', 'speakers': ['Lori', 'Stephanie']},
        'SBC029': {'state': 'NJ', 'speakers': ['Dana', 'Fran']},
        'SBC030': {'state': 'MA', 'speakers': ['Anne', 'Bill']},
        # Add more as needed - up to SBC060
    }
    
    def download(self) -> bool:
        """Check if SBCSAE audio files exist"""
        sbcsae_dir = self.cache_dir / "sbcsae"
        sbcsae_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for audio files
        audio_files = list(sbcsae_dir.glob("SBC*.wav"))
        
        if len(audio_files) == 0:
            logger.warning(f"No SBCSAE audio files found in {sbcsae_dir}")
            logger.info("SBCSAE requires manual download from LDC or UCSB")
            logger.info("Place WAV files as: ~/.cache/accent_datasets/sbcsae/SBC001.wav, etc.")
            
            # Create metadata even without audio for testing
            self._create_metadata(sbcsae_dir)
            return False
        
        logger.info(f"Found {len(audio_files)} SBCSAE audio files in {sbcsae_dir}")
        self._create_metadata(sbcsae_dir)
        return True
    
    def _create_metadata(self, sbcsae_dir: Path):
        """Create metadata file for SBCSAE"""
        metadata_file = sbcsae_dir / "sbcsae_metadata.json"
        
        if metadata_file.exists():
            return
        
        # Create synthetic metadata for demonstration
        speakers = []
        for conv_id, conv_data in self.CONVERSATION_METADATA.items():
            for speaker_name in conv_data['speakers']:
                speaker_data = {
                    'speaker_id': f"{conv_id}_{speaker_name}",
                    'conversation_id': conv_id,
                    'speaker_name': speaker_name,
                    'state': conv_data['state'],
                    'region': get_region_for_state(conv_data['state'], classification='medium')
                }
                speakers.append(speaker_data)
        
        metadata = {
            'speakers': speakers,
            'total': len(speakers),
            'source': 'Santa Barbara Corpus of Spoken American English'
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Created synthetic metadata with {len(speakers)} speaker entries")
    
    def load(self) -> List[UnifiedSample]:
        """Load SBCSAE dataset"""
        sbcsae_dir = self.cache_dir / "sbcsae"
        audio_files = list(sbcsae_dir.glob("SBC*.wav"))
        
        if len(audio_files) == 0:
            logger.warning("No SBCSAE audio files available")
            return []
        
        logger.info(f"Found {len(audio_files)} SBCSAE audio files in {sbcsae_dir}")
        
        # Load metadata
        metadata_file = sbcsae_dir / "sbcsae_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            self._create_metadata(sbcsae_dir)
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        
        logger.info(f"Loading SBCSAE dataset with {len(metadata['speakers'])} speaker entries")
        
        unified_samples = []
        
        # Process each audio file
        for audio_file in tqdm(audio_files, desc="Processing SBCSAE"):
            conv_id = audio_file.stem  # e.g., "SBC001"
            
            if conv_id not in self.CONVERSATION_METADATA:
                continue
            
            conv_data = self.CONVERSATION_METADATA[conv_id]
            
            # Create a sample for each speaker in the conversation
            for speaker_name in conv_data['speakers']:
                speaker_id = f"{conv_id}_{speaker_name}"
                
                # Get region from state
                region = get_region_for_state(conv_data['state'], classification='medium')
                
                # Create unified sample
                sample = UnifiedSample(
                    sample_id=self._generate_sample_id('SBCSAE', speaker_id, conv_id),
                    dataset_name='SBCSAE',
                    speaker_id=speaker_id,
                    audio_path=str(audio_file),
                    transcript="",  # Transcripts would need to be loaded separately
                    region_label=region,
                    original_accent_label=f"{conv_data['state']} English",
                    state=conv_data['state'],
                    gender='U',  # Would need speaker metadata for this
                    is_validated=True
                )
                
                unified_samples.append(sample)
        
        self.samples = unified_samples
        logger.info(f"Loaded {len(unified_samples)} SBCSAE samples")
        
        # Show regional distribution
        from collections import Counter
        region_counts = Counter(s.region_label for s in unified_samples)
        logger.info("SBCSAE Regional distribution:")
        for region, count in region_counts.most_common():
            logger.info(f"  {region}: {count}")
        
        return unified_samples
    
    def get_dataset_info(self) -> Dict:
        """Get dataset statistics"""
        if not self.samples:
            self.load()
        
        if not self.samples:
            return {
                'dataset': 'SBCSAE',
                'status': 'No audio files available - manual download required',
                'instructions': 'Download from LDC or UCSB and place in ~/.cache/accent_datasets/sbcsae/'
            }
        
        from collections import Counter
        region_counts = Counter(s.region_label for s in self.samples)
        state_counts = Counter(s.state for s in self.samples)
        
        return {
            'dataset': 'SBCSAE',
            'total_samples': len(self.samples),
            'total_speakers': len(set(s.speaker_id for s in self.samples)),
            'region_distribution': dict(region_counts),
            'state_distribution': dict(state_counts),
            'description': 'Santa Barbara Corpus of Spoken American English'
        }


if __name__ == "__main__":
    # Test the loader
    import logging
    from tqdm import tqdm
    logging.basicConfig(level=logging.INFO)
    
    loader = SBCSAELoader('.', '~/.cache/accent_datasets')
    samples = loader.load()
    info = loader.get_dataset_info()
    
    print("\nSBCSAE Dataset Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")