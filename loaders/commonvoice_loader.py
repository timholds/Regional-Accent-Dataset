"""
CommonVoice Loader - Extracts US regional samples using geographic indicators

This loader filters CommonVoice samples to only include those with clear
geographic indicators (states, cities, regions) while rejecting subjective
linguistic or cultural descriptors for maximum label accuracy.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import re

from unified_dataset import BaseDatasetLoader, UnifiedSample
from region_mappings import get_region_for_state

logger = logging.getLogger(__name__)


class CommonVoiceLoader(BaseDatasetLoader):
    """CommonVoice loader using geographic indicators for high-confidence regional labels"""
    
    # ONLY explicit geographic locations - no linguistic/cultural markers
    REGIONAL_PATTERNS = {
        'New England': {
            'states': ['massachusetts', 'maine', 'vermont', 'new hampshire', 
                      'rhode island', 'connecticut'],
            'cities': ['boston', 'cambridge', 'providence', 'hartford', 
                      'burlington', 'portland maine', 'worcester', 'springfield mass'],
            'regions': ['new england', 'northeast', 'northeastern'],
            'abbreviations': [' ma ', ' me ', ' vt ', ' nh ', ' ri ', ' ct ']
        },
        'New York Metropolitan': {
            'states': ['new york', 'new jersey'],
            'cities': ['new york city', 'nyc', 'brooklyn', 'manhattan', 'queens', 
                      'bronx', 'staten island', 'long island', 'newark', 
                      'jersey city', 'buffalo', 'rochester', 'albany', 'syracuse'],
            'regions': ['new york metropolitan', 'upstate new york', 'downstate new york',
                       'tri-state', 'tristate'],
            'abbreviations': [' ny ', ' nj ', 'n.y.', 'n.j.']
        },
        'Mid-Atlantic': {
            'states': ['pennsylvania', 'maryland', 'delaware', 'washington dc', 
                      'district of columbia'],
            'cities': ['philadelphia', 'pittsburgh', 'baltimore', 'harrisburg', 
                      'wilmington', 'dover', 'allentown', 'erie', 'scranton'],
            'regions': ['mid-atlantic', 'mid atlantic', 'dc area', 'dmv area', 
                       'delmarva', 'philly area'],
            'abbreviations': [' pa ', ' md ', ' de ', ' dc ', 'd.c.']
        },
        'South Atlantic': {
            'states': ['virginia', 'north carolina', 'south carolina', 'georgia', 'florida'],
            'cities': ['richmond', 'norfolk', 'virginia beach', 'charlotte', 'raleigh', 
                      'charleston', 'columbia', 'atlanta', 'savannah', 'augusta',
                      'miami', 'orlando', 'tampa', 'jacksonville', 'tallahassee'],
            'regions': ['south atlantic', 'carolinas', 'southeast', 'southeastern',
                       'south east', 'panhandle'],
            'abbreviations': [' va ', ' nc ', ' sc ', ' ga ', ' fl ', 'n.c.', 's.c.']
        },
        'Deep South': {
            'states': ['alabama', 'mississippi', 'louisiana', 'tennessee', 
                      'kentucky', 'arkansas'],
            'cities': ['birmingham', 'montgomery', 'jackson', 'new orleans', 'baton rouge',
                      'memphis', 'nashville', 'knoxville', 'louisville', 'lexington',
                      'little rock', 'mobile', 'shreveport'],
            'regions': ['deep south', 'southern united states', 'gulf coast', 'dixie',
                       'bible belt', 'southern', 'the south', 'gulf states', 'bayou',
                       'cajun', 'creole', 'appalachian', 'appalachia', 'ozark', 'bluegrass'],
            'abbreviations': [' al ', ' ms ', ' la ', ' tn ', ' ky ', ' ar ']
        },
        'Midwest': {
            'states': ['ohio', 'michigan', 'indiana', 'illinois', 'wisconsin', 
                      'minnesota', 'iowa', 'missouri', 'north dakota', 'south dakota',
                      'nebraska', 'kansas'],
            'cities': ['chicago', 'detroit', 'cleveland', 'cincinnati', 'columbus',
                      'indianapolis', 'milwaukee', 'minneapolis', 'st paul', 'twin cities',
                      'des moines', 'omaha', 'kansas city', 'st louis', 'madison',
                      'grand rapids', 'dayton', 'toledo', 'akron'],
            'regions': ['midwest', 'midwestern', 'upper midwest', 'lower midwest', 
                       'great lakes', 'rust belt', 'unite states midwest',  # typo in data
                       'midwestern united states', 'midwestern usa', 'heartland',
                       'corn belt', 'plains', 'great plains', 'prairie'],
            'abbreviations': [' oh ', ' mi ', ' in ', ' il ', ' wi ', ' mn ', ' ia ',
                            ' mo ', ' nd ', ' sd ', ' ne ', ' ks ']
        },
        'West': {
            'states': ['california', 'texas', 'washington', 'oregon', 'nevada', 
                      'arizona', 'colorado', 'utah', 'new mexico', 'idaho', 
                      'montana', 'wyoming', 'alaska', 'hawaii', 'oklahoma'],
            'cities': ['los angeles', 'san francisco', 'san diego', 'sacramento', 
                      'san jose', 'oakland', 'fresno', 'long beach',
                      'houston', 'dallas', 'austin', 'san antonio', 'fort worth', 'el paso',
                      'seattle', 'portland oregon', 'spokane', 'tacoma',
                      'phoenix', 'tucson', 'las vegas', 'reno', 'denver', 'colorado springs',
                      'salt lake city', 'albuquerque', 'boise', 'anchorage', 'honolulu',
                      'oklahoma city', 'tulsa'],
            'regions': ['west coast', 'pacific northwest', 'southwest', 'southwestern',
                       'western united states', 'california', 'southern california', 
                       'northern california', 'bay area', 'silicon valley',
                       'texas', 'west texas', 'south texas', 'pacific', 'pnw',
                       'socal', 'norcal', 'mountain west', 'rocky mountains',
                       'desert southwest', 'four corners', 'frontier'],
            'abbreviations': [' ca ', ' tx ', ' wa ', ' or ', ' nv ', ' az ', ' co ',
                            ' ut ', ' nm ', ' id ', ' mt ', ' wy ', ' ak ', ' hi ', ' ok ',
                            'l.a.', 's.f.', 'n.m.']
        }
    }
    
    # Reject patterns for clearly non-US or non-native speakers
    # More lenient to allow ethnic-American combinations
    REJECT_PATTERNS = [
        # Non-US English varieties (clear indicators)
        'england english', 'british english', 'uk english', 'united kingdom', 
        'australian english', 'canadian english', 'scottish english', 'irish english', 
        'welsh english', 'new zealand english', 'south african english',
        # Non-English as primary language (clear indicators)
        'india and south asia', 'filipino english', 'caribbean english',
        'african english', 'asian english', 'european english',
        'malaysian english', 'singaporean english', 'hong kong english',
        # Specific foreign languages/nationalities (only when clearly foreign)
        'finnish', 'swedish', 'norwegian', 'danish', 'icelandic',
        'german english', 'french english', 'russian english', 
        'polish english', 'czech english', 'hungarian english',
        'turkish english', 'arabic english', 'hebrew english',
        'chinese english', 'japanese english', 'korean english',
        'vietnamese english', 'thai english', 'hindi english',
        # Clear non-native indicators
        'non-native', 'non native', 'second language', 'esl', 'foreign accent',
        'learning english', 'english learner',
        # Only reject if clearly not US-based
        'international student', 'recently immigrated', 'new to english'
    ]
    
    def download(self) -> bool:
        """Check if CommonVoice dataset exists"""
        possible_paths = [
            Path.home() / ".cache" / "accent_datasets" / "common_voice" / "cv-corpus-22.0-2025-06-20" / "en",
            self.cache_dir / "common_voice" / "cv-corpus-22.0-2025-06-20" / "en",
            Path.home() / ".cache" / "accent_datasets" / "CommonVoice" / "cv-corpus-22.0-2025-06-20" / "en",
        ]
        
        cv_path = None
        for path in possible_paths:
            if path.exists():
                cv_path = path
                break
        
        if not cv_path:
            logger.warning(f"CommonVoice dataset not found in any expected location")
            logger.info("Download from: https://commonvoice.mozilla.org/en/datasets")
            return False
            
        self.cv_path = cv_path
        return True
    
    def _parse_accent_to_region(self, accent: str) -> Optional[str]:
        """
        Parse accent string to determine US region using ONLY geographic indicators.
        Returns None if not confident or if non-geographic descriptors are present.
        """
        if not accent or pd.isna(accent):
            return None
            
        accent_lower = accent.lower().strip()
        
        # Skip pure generic without geographic qualifier
        if accent_lower == 'united states english':
            return None
        
        # Special handling for compound labels like "United States English,Geographic"
        # Split by comma and check each part
        parts = [p.strip() for p in accent_lower.split(',')]
        
        # Check each part for reject patterns
        for part in parts:
            # Skip "united states english" itself
            if part == 'united states english':
                continue
            # Reject if any part contains non-geographic descriptors
            if any(pattern in part for pattern in self.REJECT_PATTERNS):
                return None
        
        # Count matches for each region
        region_scores = {}
        
        for region, geographic_data in self.REGIONAL_PATTERNS.items():
            score = 0
            matches = []
            
            # Check against all parts of the accent string
            for part in parts:
                # Skip generic US part
                if part == 'united states english':
                    continue
                    
                # Check states (highest confidence)
                for state in geographic_data['states']:
                    # Use word boundaries for better matching (avoid "finnish" matching "fin" in regions)
                    if state in part and not any(foreign in part for foreign in ['finnish', 'spanish', 'polish', 'danish']):
                        score += 3  # States get highest weight
                        matches.append(f"state:{state}")
                
                # Check cities (high confidence)
                for city in geographic_data['cities']:
                    if city in part:
                        score += 2  # Cities get medium weight
                        matches.append(f"city:{city}")
                
                # Check regions (medium confidence)  
                for region_name in geographic_data['regions']:
                    if region_name in part:
                        score += 1  # Regional names get lower weight
                        matches.append(f"region:{region_name}")
                
                # Check abbreviations (medium-high confidence)
                if 'abbreviations' in geographic_data:
                    for abbrev in geographic_data['abbreviations']:
                        if abbrev in part or abbrev.strip() in part:
                            score += 2  # Abbreviations get medium-high weight
                            matches.append(f"abbrev:{abbrev.strip()}")
            
            if score > 0:
                region_scores[region] = (score, matches)
        
        # No geographic matches found
        if not region_scores:
            return None
        
        # Get the best match
        best_region = max(region_scores.keys(), key=lambda k: region_scores[k][0])
        best_score, best_matches = region_scores[best_region]
        
        # Check for conflicts - if multiple regions have high scores, it's ambiguous
        second_best_score = max([s for r, (s, _) in region_scores.items() 
                                if r != best_region] + [0])
        
        # Require clear winner (best score at least 2 points higher than second)
        # For compound labels, be more lenient (1 point difference is OK)
        min_difference = 1 if ',' in accent_lower else 2
        if best_score - second_best_score < min_difference:
            return None
        
        # Require minimum score of 1 (at least one geographic match)
        if best_score < 1:
            return None
        
        return best_region
    
    def _quality_check(self, row: pd.Series) -> bool:
        """Check if sample meets quality criteria"""
        # Require good quality: upvotes > downvotes
        if row['up_votes'] >= 2 and row['down_votes'] == 0:
            return True
        # Allow medium quality if geographic confidence is high
        if row['up_votes'] >= 1 and row['down_votes'] == 0:
            return True
        return False
    
    def load(self) -> List[UnifiedSample]:
        """Load CommonVoice samples using only geographic indicators"""
        if not self.download():
            return []
        
        # Load validated TSV
        validated_tsv = self.cv_path / "validated.tsv"
        if not validated_tsv.exists():
            logger.error(f"Validated TSV not found at {validated_tsv}")
            return []
        
        logger.info(f"Loading CommonVoice from {self.cv_path}")
        logger.info("Using GEOGRAPHIC indicators only (no linguistic markers)...")
        
        # Read the TSV file - only load necessary columns to reduce memory
        needed_columns = ['client_id', 'path', 'sentence', 'up_votes', 'down_votes', 
                         'accents', 'gender', 'age']
        
        logger.info("Loading CommonVoice samples...")
        df = pd.read_csv(validated_tsv, sep='\t', usecols=needed_columns,
                        dtype={'accents': str})
        logger.info(f"Total CommonVoice samples: {len(df):,}")
        
        logger.info("Pre-filtering CommonVoice samples for geographic indicators...")
        
        # Build a mask for rows that might have geographic content
        # This dramatically reduces the dataset we need to process (from 1.8M to ~6k)
        geographic_keywords = []
        for region_data in self.REGIONAL_PATTERNS.values():
            geographic_keywords.extend([s.lower() for s in region_data['states']])
            geographic_keywords.extend([c.lower() for c in region_data['cities']])
            geographic_keywords.extend([r.lower() for r in region_data['regions']])
            if 'abbreviations' in region_data:
                geographic_keywords.extend([a.lower().strip() for a in region_data['abbreviations']])
        
        # Create a regex pattern for fast pre-filtering
        pattern = '|'.join(geographic_keywords)
        
        # Pre-filter for potential geographic content
        accent_mask = df['accents'].notna() & df['accents'].str.lower().str.contains(
            pattern, na=False, regex=True
        )
        
        # Filter to potentially relevant rows
        filtered_df = df[accent_mask].copy()
        logger.info(f"Found {len(filtered_df):,} potentially relevant rows out of {len(df):,}")
        
        # Now apply quality filtering to the pre-filtered set
        quality_mask = filtered_df.apply(self._quality_check, axis=1)
        quality_df = filtered_df[quality_mask]
        logger.info(f"After quality filtering: {len(quality_df):,} samples")
        
        unified_samples = []
        clips_dir = self.cv_path / "clips"
        
        # Track statistics
        stats = {
            'total_processed': 0,
            'geographic_found': 0,
            'rejected_non_geographic': 0,
            'no_region_match': 0,
            'generic_us_skipped': 0,
            'no_audio_skipped': 0,
            'by_region': {region: 0 for region in self.REGIONAL_PATTERNS.keys()}
        }
        
        # Track examples for debugging
        examples = {
            'accepted': [],
            'rejected': []
        }
        
        for idx, row in tqdm(quality_df.iterrows(), total=len(quality_df), 
                            desc="Processing CommonVoice"):
            stats['total_processed'] += 1
            
            accent = str(row.get('accents', ''))
            
            # Check for reject patterns first
            if any(p in accent.lower() for p in self.REJECT_PATTERNS):
                stats['rejected_non_geographic'] += 1
                if len(examples['rejected']) < 10 and accent:
                    examples['rejected'].append(accent)
                continue
            
            # Try to parse region
            region = self._parse_accent_to_region(accent)
            
            if not region:
                if accent.lower() == 'united states english':
                    stats['generic_us_skipped'] += 1
                else:
                    stats['no_region_match'] += 1
                continue
            
            # Check if audio file exists
            audio_file = clips_dir / row['path']
            if not audio_file.exists():
                stats['no_audio_skipped'] += 1
                continue
            
            # Create unified sample
            sample = UnifiedSample(
                sample_id=self._generate_sample_id('CommonVoice', 
                                                  row['client_id'], row['path']),
                dataset_name='CommonVoice',
                speaker_id=row['client_id'],
                audio_path=str(audio_file),
                transcript=row['sentence'],
                region_label=region,
                original_accent_label=accent,
                gender=row.get('gender', 'U'),
                age=row.get('age', ''),
                is_validated=True,
                quality_score=row['up_votes'] / (row['up_votes'] + row['down_votes'] + 1)
            )
            
            unified_samples.append(sample)
            stats['geographic_found'] += 1
            stats['by_region'][region] += 1
            
            # Track examples
            if len(examples['accepted']) < 10:
                examples['accepted'].append(f"{accent} -> {region}")
            
            if stats['geographic_found'] % 1000 == 0:
                logger.info(f"Found {stats['geographic_found']:,} geographic samples...")
        
        self.samples = unified_samples
        
        # Log final statistics
        logger.info(f"\n{'='*60}")
        logger.info("CommonVoice Loading Complete:")
        logger.info(f"  Total processed: {stats['total_processed']:,}")
        logger.info(f"  Geographic samples found: {stats['geographic_found']:,}")
        logger.info(f"  Rejected (non-geographic): {stats['rejected_non_geographic']:,}")
        logger.info(f"  No region match: {stats['no_region_match']:,}")
        logger.info(f"  Generic US skipped: {stats['generic_us_skipped']:,}")
        logger.info(f"  No audio file: {stats['no_audio_skipped']:,}")
        
        logger.info("\nRegional distribution:")
        for region, count in sorted(stats['by_region'].items(), 
                                   key=lambda x: x[1], reverse=True):
            if count > 0:
                pct = count / stats['geographic_found'] * 100 if stats['geographic_found'] > 0 else 0
                logger.info(f"  {region}: {count:,} ({pct:.1f}%)")
        
        # Show examples
        if examples['accepted']:
            logger.info("\nExample accepted patterns:")
            for ex in examples['accepted'][:5]:
                logger.info(f"  ✓ {ex}")
        
        if examples['rejected']:
            logger.info("\nExample rejected patterns:")
            for ex in examples['rejected'][:5]:
                logger.info(f"  ✗ {ex}")
        
        logger.info(f"{'='*60}\n")
        
        return unified_samples
    
    def get_dataset_info(self) -> Dict:
        """Get dataset statistics"""
        if not self.samples:
            self.load()
        
        if not self.samples:
            return {'dataset': 'CommonVoice', 'status': 'No samples loaded'}
        
        df = pd.DataFrame([s.to_dict() for s in self.samples])
        
        return {
            'dataset': 'CommonVoice',
            'total_samples': len(self.samples),
            'total_speakers': df['speaker_id'].nunique(),
            'region_distribution': df['region_label'].value_counts().to_dict(),
            'avg_quality_score': df['quality_score'].mean() if 'quality_score' in df else None,
            'description': 'CommonVoice with geographic indicators only (no linguistic markers)'
        }


if __name__ == "__main__":
    # Test the geographic loader
    import logging
    logging.basicConfig(level=logging.INFO)
    
    loader = CommonVoiceLoader('.', '~/.cache/accent_datasets')
    samples = loader.load()
    info = loader.get_dataset_info()
    
    print("\nCommonVoice Dataset Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")