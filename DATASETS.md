# Dataset Documentation

## New Dataset Acquisition
New datasets should have samples that are native english speakers. The data sources should be free of copyright restrictions and have licenses that are compatible with use in this project.

## Priority Datasets for Future Acquisition

Based on analysis of current dataset limitations (birthplace-based labeling in SAA, limited samples ~10k total, 500k trainable parameters requiring more data), these are the **top 3 datasets** to prioritize for incorporation:

### 1. **Nationwide Speech Project (NSP)** - HIGHEST PRIORITY
- **Size**: 60 speakers × ~1 hour each = 60 hours of speech
- **Regional Coverage**: 6 US dialect regions with explicit mapping (New England, Mid-Atlantic, North, Midland, South, West)
- **Reliability**: ✅ **Excellent** - Speakers were carefully selected based on their regional background, recorded in controlled conditions
- **Key Advantage**: Unlike SAA's birthplace-based labeling, NSP verified speakers' current regional dialect through linguistic analysis
- **Access**: Through Linguistic Data Consortium (LDC2007S15) or direct contact with Ohio State University (Cynthia Clopper)
- **Why critical**: Provides ground-truth regional labels with 10 speakers per region, addressing the accent drift issue where birthplace ≠ current accent
- **Recording Quality**: High-quality digital recordings in sound-attenuated booth
- **Materials**: Words, sentences, passages, and interviews

### 2. **DARE Audio Project** - FREE ACCESS
- **Size**: 1,843 recordings, ~900 hours of conversation (massive scale)
- **Regional Coverage**: 1,002 communities across all US states
- **Reliability**: ✅ **Excellent** - Recordings include geographic location where interview took place (not just birthplace)
- **Key Advantage**: Verified recording locations, includes both casual conversation and standardized reading passages ("Arthur the Rat")
- **Access**: **FREE** through University of Wisconsin Digital Collections (https://dare.wisc.edu/audio/)
- **Why critical**: Recordings from 1965-1970 capture authentic regional dialects before modern accent leveling, with actual recording location metadata
- **Unique Value**: Historical preservation of accents that may have since evolved or disappeared
- **Searchable**: Database searchable by topics, biographical, or geographical characteristics

### 3. **SLAAP North Carolina Collection** - ADDRESSES SOUTH ATLANTIC GAP
- **Size**: Multiple NC datasets including:
  - Asheville: 46 recordings (1974, 19 African American speakers)
  - Roanoke Island: 35 recordings (2003)
  - Robeson County: 23+ recordings
- **Regional Coverage**: Specifically targets South Atlantic region (currently only 0.6% of our dataset)
- **Reliability**: ✅ **Excellent** - Sociolinguistic interviews with verified local speakers
- **Key Advantage**: Directly addresses the South Atlantic gap with 100+ North Carolina speakers
- **Access**: Through NC State University SLAAP archive (https://slaap.chass.ncsu.edu/) - password-protected but accessible
- **Why critical**: Would increase South Atlantic representation from 50 to ~500+ samples after chunking
- **Additional Value**: Professional sociolinguistic curation with demographic metadata

### Why These Three Over Others:
- **Not More CommonVoice**: Self-reported accent labels remain unreliable regardless of volume
- **Not VoxCeleb**: No US regional metadata, only nationality-level labels
- **Not IARPA BABEL**: Focuses on non-English underserved languages
- **Not More SAA Data**: Still relies on birthplace rather than current accent location

### Expected Impact:
- **Combined addition**: 1000+ hours of verified regional speech data
- **South Atlantic coverage**: Increase from 0.6% to ~10-15% of dataset
- **Ground truth improvement**: Recording location data instead of birthplace assumptions
- **Historical diversity**: DARE provides 1960s accents, complementing modern recordings

### Audio Chunking for Long-Form Content
The pipeline automatically handles long-form audio (interviews, conversations) through intelligent chunking:
- **Default chunk size**: 7.5 seconds (120,000 samples at 16kHz - optimal for Wav2Vec2)
- **Overlap**: 2.5 seconds between chunks (40,000 samples - preserves context at boundaries)
- **Lazy chunking**: Audio lengths are checked only when accessed, not at initialization
- **Automatic processing**: Long audio from CORAAL, SBCSAE, etc. is chunked during training
- **Efficient**: Uses soundfile to read audio metadata without loading full files
- **Random selection**: For multi-chunk files, randomly selects chunks each epoch for variety

`--chunk_duration 7.5 (default)`
  - Perfect for wav2vec2: 7.5s × 16kHz = 120k samples fits well in model context
  - Training efficiency: Consistent batch sizes, good gradient updates
  - Not too short: Captures enough accent features
  - Not too long: Avoids memory issues

`--chunk_overlap 2.5 (default)`
- 33% overlap: Ensures no accent features are lost at boundaries
- More training data: Creates more diverse samples from long audio
- Smooth transitions: Helps model learn continuous speech patterns

`--min_chunk_duration 5.0 (default)`
- Minimum viable: 5 seconds is enough for accent identification
- Filters noise: Removes very short, likely low-quality segments

`--max_chunk_duration 10.0 (default)`
- Memory limit: Prevents extremely long samples that cause OOM
- Forces chunking: Files >10s get split into manageable pieces


## Overview

This document describes all datasets supported by the unified accent classification pipeline. Each dataset can be included via the `prepare_dataset.py` script.

### Supported Datasets
1. **SAA** - Speech Accent Archive (George Mason University)
2. **TIMIT** - Texas Instruments/MIT Acoustic-Phonetic Corpus
3. **CommonVoice** - Mozilla Common Voice
4. **CORAAL** - Corpus of Regional African American Language
5. **PNC** - Philadelphia Neighborhood Corpus *(if available)*
6. **SBCSAE** - Santa Barbara Corpus of Spoken American English

### Usage
```bash
# Single dataset
python prepare_dataset.py --datasets SAA

# Multiple datasets
python prepare_dataset.py --datasets TIMIT SAA

# All datasets (default)
python prepare_dataset.py --datasets TIMIT CommonVoice CORAAL SAA PNC SBCSAE
```

## Unified Sample Schema

All datasets are converted to this unified format:

| Field | Type | Description | Required |
|-------|------|-------------|----------|
| `sample_id` | str | Unique ID across all datasets | ✓ |
| `dataset_name` | str | Source dataset (TIMIT, SAA, etc.) | ✓ |
| `speaker_id` | str | Unique speaker identifier | ✓ |
| `audio_path` | str | Path to audio file | ✓ |
| `transcript` | str | Text transcription | ✓ |
| `region_label` | str | Our 8-region classification | ✓ |
| `original_accent_label` | str | Original dataset's label | ✓ |
| `state` | str | US state code (if available) | |
| `gender` | str | M/F/O/U | |
| `age` | str | Age or age range | |
| `native_language` | str | Speaker's native language | |
| `duration` | float | Audio duration in seconds | |
| `sample_rate` | int | Audio sample rate in Hz | |
| `is_validated` | bool | Quality validation flag | |
| `quality_score` | float | Audio quality metric | |

## Regional Classification

All datasets map to these 8 US regions:

1. **New England** - ME, NH, VT, MA, RI, CT
2. **New York Metropolitan** - NY, NJ (northern)
3. **Mid-Atlantic** - PA, DE, MD, DC
4. **South Atlantic** - VA, NC, SC, GA, FL
5. **Deep South** - AL, MS, LA, AR, TN, KY
6. **Upper Midwest** - WI, MI, MN, ND, SD, IL (northern)
7. **Lower Midwest** - OH, IN, MO, IA, NE, KS
8. **West** - CA, TX, WA, OR, CO, AZ, NV, UT, ID, WY, MT, NM, AK, HI, OK, WV

---

## 1. SAA - Speech Accent Archive

### Overview
- **Source**: George Mason University
- **Size**: ~2,500 speakers (1,197 US native English)
- **Focus**: Global English accents, filtered to US native speakers
- **Audio**: Single ~20-second reading of "Please call Stella" passage
- **Format**: MP3, 44.1kHz → converted to 16kHz WAV

### Data Structure
```
~/.cache/accent_datasets/saa/
├── saa_speakers.csv       # Metadata
└── audio/
    ├── english1.mp3
    ├── english2.mp3
    └── ...
```

### Metadata Fields
| Field | Example | Notes |
|-------|---------|-------|
| `speaker_id` | english123 | Unique SAA identifier |
| `birthplace_city` | Baltimore | City name |
| `birthplace_state` | MD | State code |
| `birthplace_country` | USA | Country |
| `native_language` | English | Filtered to English only |
| `age` | 26-35 | Age range |
| `gender` | female | male/female |
| `audio_file` | english123.mp3 | Audio filename |

### ⚠️ IMPORTANT WARNING: Birthplace vs. Accent Origin
**This dataset uses BIRTHPLACE for region labeling, NOT where speakers were raised or acquired their accent.** This is a fundamental limitation because:
- Accents are primarily acquired during childhood (ages 3-12), not at birth
- Many people move during childhood and develop accents different from their birthplace
- A person born in Texas but raised in Boston will likely have a Boston accent, not a Texas accent
- This mislabeling can introduce significant noise into accent classification models

**Recommendation**: Use SAA samples with caution and consider them as supplementary data rather than ground truth. When possible, prefer datasets like TIMIT or CORAAL that have more reliable regional associations.

### Regional Mapping
- Maps `birthplace_state` to our 8 regions using `MEDIUM_MAPPINGS`
- Filters: `birthplace_country == 'USA'` AND `native_language == 'English'`
- Primary contribution: Mid-Atlantic (381) and South Atlantic (707) samples
- **Note**: These regional labels may not reflect actual accent characteristics due to birthplace limitation

### Loader Implementation
- **File**: `saa_loader.py`
- **Class**: `SAALoader`
- **Auto-download**: Creates mock data for testing; real data from accent.gmu.edu

---

## 2. TIMIT - Texas Instruments/MIT Corpus

### Overview
- **Source**: LDC/Kaggle
- **Size**: 630 speakers, 6,300 utterances
- **Focus**: US regional dialects
- **Audio**: 10 phonetically-rich sentences per speaker
- **Format**: 16kHz WAV

### Data Structure
```
./TIMIT/
├── TRAIN/
│   ├── DR1/           # Dialect Region 1
│   │   ├── FCJF0/     # Speaker directory
│   │   │   ├── SA1.WAV
│   │   │   ├── SA1.TXT
│   │   │   └── ...
│   │   └── ...
│   └── ...
└── TEST/
    └── ...
```

### Dialect Regions
| TIMIT Code | Name | Our Region |
|------------|------|------------|
| DR1 | New England | New England |
| DR2 | Northern | Upper Midwest |
| DR3 | North Midland | Lower Midwest |
| DR4 | South Midland | Deep South |
| DR5 | Southern | Deep South |
| DR6 | New York City | New York Metropolitan |
| DR7 | Western | West |
| DR8 | Army Brat | West (default) |

### Metadata
- **Gender**: M/F from speaker ID
- **Sentences**: SA (dialect), SI (compact), SX (phonetically-diverse)
- **No direct mapping** to Mid-Atlantic or South Atlantic regions

### Loader Implementation
- **File**: `unified_dataset.py`
- **Class**: `TIMITLoader`
- **Auto-download**: From Kaggle (requires API key)

---

## 3. CommonVoice - Mozilla Common Voice

### ✅ FILTERED APPROACH NOW AVAILABLE

**FilteredCommonVoice Loader**: A new filtered loader (`commonvoice_filtered_loader.py`) has been implemented that:
- **Extracts only the ~5,558 samples with specific regional labels** 
- **Filters out all 441,218 generic "United States English" samples**
- **Applies quality filtering**: min_upvotes=2, max_downvotes=0
- **Successfully contributes to training** with verified regional samples

#### Original Dataset Analysis (v22.0):
- **Total samples**: 1,855,218 validated clips
- **US-labeled samples**: 441,218 (23.8%)
- **Regionally-specific samples**: 5,558 after filtering (0.3%)
- **Generic labels filtered out**: 441,218 samples

#### Why the Filtered Approach Works:
1. **Strict regional matching** - Only accepts samples with explicit regional indicators
2. **Quality thresholds** - Uses upvote/downvote data to filter quality
3. **No generic defaults** - Rejects all "United States English" without regional info
4. **Verified in training** - Successfully used in latest prepared datasets

### Overview
- **Source**: Mozilla Foundation  
- **Size**: 85+ GB for English subset
- **Total samples**: 1.85M validated clips
- **US Coverage**: 441K labeled as US (but 99%+ lack regional info)
- **Focus**: Crowdsourced global speech
- **Audio**: Variable length sentences
- **Format**: MP3 → 16kHz WAV

### Data Structure
```
~/.cache/accent_datasets/common_voice/cv-corpus-22.0-2025-06-20/
├── validated.tsv      # Validated samples
├── train.tsv
├── dev.tsv
├── test.tsv
└── clips/
    ├── common_voice_en_123.mp3
    └── ...
```

### Metadata Fields
| Field | Description | Reliability |
|-------|-------------|-------------|
| `client_id` | Speaker ID | ✓ Reliable |
| `path` | Audio filename | ✓ Reliable |
| `sentence` | Transcript | ✓ Reliable |
| `accent` | Self-reported accent | ❌ UNRELIABLE |
| `age` | Age range | ~ Partial |
| `gender` | male/female/other | ~ Partial |

**Note**: No location metadata (state/city) available

### Regional Mapping Solution
- **FilteredCommonVoice approach**: Strict pattern matching for regional keywords
- **Filtered samples by region** (from latest run):
  - Midwest: ~2,200 samples
  - Deep South: ~1,500 samples  
  - West: ~1,200 samples
  - New York Metropolitan: ~400 samples
  - Mid-Atlantic: ~150 samples
  - New England: ~80 samples
  - South Atlantic: ~28 samples
- **Key improvements**:
  - No arbitrary defaults - unmappable samples are excluded
  - Strict regional keyword matching
  - Quality filtering via voting data

### Real Examples from v22.0 Data
- "United States English" → West (441,218 samples with zero regional info)
- "United States English,Midwestern,Low,Demure" → Upper Midwest (1,470 samples)
- Empty/missing → West (831,985 samples without any accent label)
- "United States English,southern United States" → Deep South (only 551 samples)
- 42 different "Midwest" variants showing extreme inconsistency

### Loader Implementations

#### FilteredCommonVoiceLoader (RECOMMENDED)
- **File**: `commonvoice_filtered_loader.py`
- **Class**: `FilteredCommonVoiceLoader`
- **Dataset location**: `.cache/CommonVoice/cv-corpus-22.0-2025-06-20/en/`
- **Processing**: Fast - only processes ~5,558 regional samples
- **Usage in prepare_dataset.py**: `--datasets FilteredCommonVoice`

#### Original CommonVoiceLoader (NOT RECOMMENDED)
- **File**: `unified_dataset.py`
- **Class**: `CommonVoiceLoader`
- **Issues**: Includes 441K generic samples that corrupt training

---

## 4. CORAAL - Corpus of Regional African American Language

### Overview
- **Source**: University of Oregon
- **Size**: ~220+ speakers (v2023.06)
- **Focus**: African American English varieties
- **Audio**: Long sociolinguistic interviews (30-60 min)
- **Format**: 44.1kHz WAV
- **Latest Version**: 2023.06 (includes Detroit component)

### Data Structure
```
~/.cache/accent_datasets/coraal/
├── DCA/               # Washington DC
│   ├── audio/
│   │   ├── DCA_se1_ag1_m_01.wav
│   │   └── ...
│   └── DCA_metadata.txt
├── ATL/               # Atlanta
├── ROC/               # Rochester
└── ...
```

### Components
| Code | City | State | Our Region | Audio Files | Status |
|------|------|-------|------------|-------------|--------|
| DCA | Washington DC | DC | Mid-Atlantic | Multiple | ✅ Available |
| DCB | Washington DC | DC | Mid-Atlantic | Part 1 | ✅ Available |
| ATL | Atlanta | GA | Deep South | Multiple | ✅ Available |
| PRV | Princeville | NC | South Atlantic | Part 1 | ✅ **Critical for NC coverage** |
| VLD | Valdosta | GA | Deep South | 0 | ❌ Not found |
| ROC | Rochester | NY | New York Metropolitan | Multiple | ✅ Available |
| LES | Lower East Side | NY | New York Metropolitan | Multiple | ✅ Available |
| DTA | Detroit | MI | Upper Midwest | Part 1 | ✅ Available |

#### Important Discovery: Missing CORAAL Components
During analysis, we discovered that PRV (Princeville, NC) and DCB (Washington DC) components were not included in the original CORAAL_COMPONENTS dictionary despite being available. These have now been added:

**PRV Component (Critical for South Atlantic):**
- URL: `https://lingtools.uoregon.edu/coraal/prv/2018.10.06/PRV_audio_part01_2018.10.06.tar.gz`
- Impact: Adds 30-50 speakers from North Carolina
- Expected samples after chunking: +500-1000 for South Atlantic region
- This more than doubles South Atlantic coverage (from 707 to ~1,700 samples)

**DCB Component (Enhances Mid-Atlantic):**
- URL: `https://lingtools.uoregon.edu/coraal/dcb/2018.10.06/DCB_audio_part01_2018.10.06.tar.gz`
- Impact: Additional Washington DC speakers to complement DCA

### Metadata
- **Naming**: `COMPONENT_speaker_interviewer.wav`
- **Demographics**: Age, gender, occupation in metadata files
- **Transcripts**: Separate files (not included in pipeline)

### Loader Implementation
- **File**: `unified_dataset.py`
- **Class**: `CORAALLoader`
- **Auto-download**: Attempts download from lingtools.uoregon.edu

---

## 5. PNC - Philadelphia Neighborhood Corpus

### Overview
- **Source**: University of Pennsylvania
- **Size**: TBD
- **Focus**: Philadelphia regional accents
- **Audio**: Sociolinguistic interviews
- **Format**: WAV

### Status
- **Implementation**: `pnc_loader.py` (if available)
- **Region**: Maps to Mid-Atlantic
- **Note**: Optional dataset, may not be available

---

## 6. SBCSAE - Santa Barbara Corpus of Spoken American English ⚠️ AUDIO NOT AVAILABLE

### ⚠️ STATUS - TRANSCRIPTS ONLY, NO AUDIO
**Current Status: Transcripts downloaded successfully, audio requires manual acquisition**

#### Current State:
- **Transcripts available**: 60 transcript files (.trn) ✅
- **Audio files available**: 0 audio files ❌
- **Metadata generated**: Yes, for 120 speakers ✅
- **Contributing to training**: 0 samples (no audio)

### Overview
- **Source**: UC Santa Barbara Linguistics Department
- **Size**: ~60 hours of natural conversation (WHEN AVAILABLE)
- **Conversations**: 60 recorded conversations
- **Speakers**: ~250 American English speakers
- **Focus**: Natural conversational American English
- **Audio**: High-quality recordings of unscripted conversations (REQUIRES MANUAL ACQUISITION)
- **Format**: WAV files with time-aligned transcriptions

### Data Structure
```
~/.cache/accent_datasets/sbcsae/
├── sbcsae_metadata.csv    # Speaker and conversation metadata
├── SBC001.wav             # Audio file for conversation 1
├── SBC001.txt             # Transcript for conversation 1
├── SBC002.wav
├── SBC002.txt
└── ...
```

### Metadata Fields
| Field | Example | Notes |
|-------|---------|-------|
| `conversation_id` | SBC001 | Unique conversation identifier |
| `speaker_id` | SBC001_Lenore | Conversation + speaker name |
| `speaker_name` | Lenore | Speaker's pseudonym |
| `state` | CA | Speaker's home state |
| `conversation_title` | Actual Blacksmithing | Topic/title of conversation |

### Regional Coverage
Based on the 30 conversations currently mapped in the loader:
- **West**: 8 conversations (CA, WA, OR, CO, TX)
- **Mid-Atlantic**: 4 conversations (PA, MD, DC, NJ)
- **South Atlantic**: 5 conversations (VA, NC, SC, GA, FL)
- **Deep South**: 4 conversations (AL, LA, TN)
- **Upper Midwest**: 5 conversations (MI, WI, IL, MN)
- **Lower Midwest**: 3 conversations (OH, IN, MO)
- **New England**: 1 conversation (MA, CT)

### Key Features
- **Natural speech**: Unscripted, spontaneous conversations
- **Diverse contexts**: Face-to-face, telephone, various social settings
- **Multiple speakers**: Many conversations have 2-5 participants
- **Geographic metadata**: Each speaker's state is documented
- **Professional quality**: Linguistically curated and annotated

### Loader Implementation
- **File**: `sbcsae_loader.py`
- **Class**: `SBCSAELoader`
- **Manual download**: Currently requires manual placement of files
- **Auto-metadata**: Creates synthetic metadata for demonstration

### Current Issues
1. **No automatic download available** - Original Box.com URLs return HTML, not audio
2. **TalkBank URLs don't work** - Expected paths return 404 errors
3. **Requires paid LDC access** - Professional corpus licensing required
4. **Manual acquisition only** - Users must obtain files themselves

### Download Instructions (MANUAL ONLY)
**Option 1: Linguistic Data Consortium (LDC) - PAID ACCESS**
1. Visit: https://www.ldc.upenn.edu/
2. Search for "Santa Barbara Corpus"
3. Purchase Parts 1-4: LDC2000S85, LDC2003S06, LDC2004S09, LDC2005S25
4. Place WAV files as: `~/.cache/accent_datasets/sbcsae/SBC001.wav`, etc.

**Option 2: UCSB Direct (if available)**
1. Visit: https://www.linguistics.ucsb.edu/research/santa-barbara-corpus
2. Check for direct download links (availability varies)
3. Download audio files manually
4. Place WAV files in: `~/.cache/accent_datasets/sbcsae/`

### Contributing 0 Samples Currently
**Until audio files are manually placed, SBCSAE contributes 0 samples to training.**

The loader will:
- ✅ Create metadata for all 120 speakers
- ✅ Show clear instructions for obtaining audio
- ✅ Not crash when audio files are missing
- ❌ Actually provide any training data

---

## Creating a Custom Dataset Loader

To add a new dataset, create a loader class inheriting from `BaseDatasetLoader`:

```python
from unified_dataset import BaseDatasetLoader, UnifiedSample

class MyDatasetLoader(BaseDatasetLoader):
    def download(self) -> bool:
        """Download dataset if needed"""
        # Check if exists
        # Download if not
        return True
    
    def load(self) -> List[UnifiedSample]:
        """Load and convert to unified format"""
        samples = []
        
        # Load your data
        for item in your_data:
            sample = UnifiedSample(
                sample_id=self._generate_sample_id('MyDataset', speaker_id, utterance_id),
                dataset_name='MyDataset',
                speaker_id=speaker_id,
                audio_path=audio_path,
                transcript=transcript,
                region_label=self._map_to_region(state=state),  # Use helper
                original_accent_label=original_label,
                # ... other fields
            )
            samples.append(sample)
        
        return samples
    
    def get_dataset_info(self) -> Dict:
        """Return dataset statistics"""
        # Return stats dict
```

### Integration Steps

1. **Create loader file**: `my_dataset_loader.py`
2. **Add to unified_dataset.py**:
```python
from my_dataset_loader import MyDatasetLoader
self.loaders['MyDataset'] = MyDatasetLoader(data_root, cache_dir)
```
3. **Update prepare_dataset.py choices**:
```python
choices=["TIMIT", "CommonVoice", "CORAAL", "SAA", "PNC", "MyDataset"]
```

---

## Dataset Statistics

### Current Distribution (Latest Prepared Dataset - accent_dataset_balanced)

**Total samples after chunking and filtering: 8,860**

| Region | Samples | % of Dataset | Primary Sources |
|--------|---------|--------------|----------------|
| Midwest* | 2,934 | 33.1% | TIMIT, FilteredCommonVoice, CORAAL |
| Deep South | 2,151 | 24.3% | TIMIT, FilteredCommonVoice, CORAAL |
| West | 1,751 | 19.8% | TIMIT, FilteredCommonVoice, SAA |
| New York Metropolitan | 811 | 9.2% | TIMIT, FilteredCommonVoice, CORAAL |
| Mid-Atlantic | 603 | 6.8% | SAA, CORAAL (DCA, DCB), FilteredCommonVoice |
| New England | 560 | 6.3% | TIMIT, FilteredCommonVoice |
| South Atlantic | 50 | 0.6% | CORAAL (PRV), FilteredCommonVoice |

*Upper Midwest and Lower Midwest have been consolidated into a single "Midwest" region

#### ⚠️ Regional Imbalance Issue
**South Atlantic region severely underrepresented (0.6% of dataset)**
- Current: Only 50 samples from limited sources
- After PRV integration: Expected to increase to ~1,700 samples (15% of dataset)
- This represents a 140% increase in South Atlantic coverage

**Key improvements from latest dataset:**
- FilteredCommonVoice successfully contributes 5,558 regional samples
- CORAAL chunks provide extensive coverage after audio chunking
- Speaker capping at 50 samples prevents memorization
- All samples have chunk metadata for efficient loading

### Key Insights
- SAA fills critical gaps in Mid-Atlantic and South Atlantic
- TIMIT provides strong coverage of other regions
- **PRV component (Princeville, NC) adds crucial North Carolina data for South Atlantic**
- **DCB component doubles Mid-Atlantic coverage from DC**
- **DTA component provides Detroit representation for Upper Midwest**
- Combined dataset has representation for all 8 regions
- No speaker overlap between train/val/test splits

### Action Items for Regional Balance
1. **Priority #1**: Ensure PRV component is downloaded and integrated
   ```bash
   python download_coraal.py PRV
   ```
2. **Alternative South Atlantic sources** if PRV issues persist:
   - SLAAP (NC State) - Asheville dataset with 46 NC recordings
   - Re-map ATL component from Deep South to South Atlantic (Georgia is technically South Atlantic)
   - Expand SAA collection for VA, NC, SC, GA, FL speakers

---

## Quality Considerations

### Audio Quality Factors
1. **Recording Environment**: Studio (TIMIT) vs. Web (SAA) vs. Field (CORAAL)
2. **Sample Rate**: Standardized to 16kHz
3. **Duration**: Single utterance (SAA) vs. Multiple (TIMIT) vs. Long-form (CORAAL)
4. **Background Noise**: Varies by dataset

### Filtering Applied
1. **Native Speakers**: SAA filters to native English only
2. **US Only**: CommonVoice filtered to US accents
3. **Validated**: Using validated splits where available
4. **Quality Threshold**: Can filter by SNR or other metrics

---

## Training Recommendations

### Best Practices
1. **Start with TIMIT**: Cleanest, most consistent labels
2. **Add SAA**: For Mid-Atlantic and South Atlantic coverage
3. **Add CORAAL**: For diversity and city-specific samples
4. **Use FilteredCommonVoice**: Adds 5,558 quality regional samples
5. **AVOID original CommonVoice**: Generic labels corrupt training

### Recommended Dataset Combinations

#### High Quality Training (RECOMMENDED)
```bash
# Best quality with filtered CommonVoice
python prepare_dataset.py --datasets TIMIT FilteredCommonVoice CORAAL SAA

# With speaker capping to prevent memorization
python prepare_dataset.py --datasets TIMIT FilteredCommonVoice CORAAL SAA --max_samples_per_speaker 50
```

#### Baseline Testing
```bash
# Start with just TIMIT for debugging
python prepare_dataset.py --datasets TIMIT
```

#### Regional Coverage
```bash
# Add SAA for missing regions
python prepare_dataset.py --datasets TIMIT SAA
```

### ⚠️ Dataset Quality Issues

| Dataset | Label Quality | Location Data | Samples | Recommendation |
|---------|--------------|---------------|---------|----------------|
| TIMIT | ✅ Excellent | Dialect regions | 6,300 | **Use as primary** |
| SAA | ✅ Good | State-level | 1,197 | **Use for coverage** |
| CORAAL | ✅ Excellent | City-specific | ~380 chunks | **Use for diversity** |
| FilteredCommonVoice | ✅ Good | Regional keywords | 5,558 | **Use filtered version** |
| CommonVoice (original) | ❌ Poor | None | 441K | **AVOID** |
| SBCSAE | ✅ Good | State-level | 0 (no audio) | **Manual setup only** |
| PNC | ✅ Good | City-specific | 0 | Not available |

### Why CommonVoice Fails for Regional Accents

1. **No ground truth**: Self-reported labels are unreliable
2. **No location data**: Can't verify claims about accent origin
3. **Inconsistent taxonomy**: Users report accents differently
4. **Default mapping bias**: Unknown samples arbitrarily assigned to 'West'
5. **Label noise**: Corrupts clean training signal from other datasets

### Sample Commands
```bash
# RECOMMENDED: High-quality training with all working datasets
python prepare_dataset.py \
    --datasets TIMIT FilteredCommonVoice CORAAL SAA \
    --max_samples_per_speaker 50 \
    --output_dir prepared_dataset \
    --force \
    --min_chunk_duration 5.0 \
    --max_chunk_duration 10.0

# Debugging: Start simple
python prepare_dataset.py --datasets TIMIT

# Test subset with chunking
python prepare_dataset.py --datasets TIMIT SAA --max_samples_per_dataset 100 \
    --chunk_duration 7.5 --chunk_overlap 2.5

# NOT RECOMMENDED: Original CommonVoice
# python prepare_dataset.py --datasets TIMIT CommonVoice  # Will hurt performance!
```

---

*Last updated: 2025-08-18*
*Pipeline version: 1.1*

### Summary of Key Updates (Aug 18):
- ✅ **FilteredCommonVoice** successfully extracts 5,558 regional samples
- ✅ **Chunking** now applied to all datasets by default (7.5s chunks, 2.5s overlap)
- ✅ **Regional consolidation**: Upper/Lower Midwest merged into single "Midwest" region
- ✅ **Speaker capping**: Max 50 samples per speaker prevents memorization
- ✅ **CORAAL components**: PRV (4 files), DTA (8 files) confirmed working
- ⚠️ **SBCSAE**: Transcripts available but audio still requires manual acquisition