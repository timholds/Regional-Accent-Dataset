# Dataset Documentation

## New Dataset Acquisition
New datasets should have samples that are native english speakers. The data sources should be free of copyright restrictions and have licenses that are compatible with use in this project.

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

### Regional Mapping
- Maps `birthplace_state` to our 8 regions using `MEDIUM_MAPPINGS`
- Filters: `birthplace_country == 'USA'` AND `native_language == 'English'`
- Primary contribution: Mid-Atlantic (381) and South Atlantic (707) samples

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

### ⚠️ CRITICAL DATA QUALITY ISSUES - NOT RECOMMENDED

**Based on analysis of REAL CommonVoice v22.0 data (1.85M validated samples):**

#### The Numbers Don't Lie:
- **Total samples**: 1,855,218 
- **US-labeled samples**: 441,218 (23.8%)
- **Samples mappable to specific US regions**: 5,893 (0.32%)
- **Samples with generic "United States English"**: 450,080 (99%+ of US samples)

#### Fatal Issues for Regional Accent Classification:
1. **No location metadata** - Zero state/city data, only user-reported accent labels
2. **99%+ generic labels** - Almost all US samples just say "United States English"
3. **0.32% regional specificity** - Only 5,893 of 1.85M samples map to specific regions
4. **Extreme label inconsistency** - "Midwest" alone has 42 different label variants
5. **No verification possible** - Cannot validate self-reported labels without location data

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

### Regional Mapping Problems
- **Current approach**: Parses `accent` field for US location keywords
- **Major issues**:
  - Generic labels ("American", "US English") default to 'West' region
  - No verification of self-reported accents
  - Inconsistent labeling (e.g., "US", "united states", "american" all different)
  - Many speakers don't provide accent information
  - When provided, accent labels often don't map to specific regions

### Real Examples from v22.0 Data
- "United States English" → West (441,218 samples with zero regional info)
- "United States English,Midwestern,Low,Demure" → Upper Midwest (1,470 samples)
- Empty/missing → West (831,985 samples without any accent label)
- "United States English,southern United States" → Deep South (only 551 samples)
- 42 different "Midwest" variants showing extreme inconsistency

### Loader Implementation
- **File**: `unified_dataset.py`
- **Class**: `CommonVoiceLoader`
- **Manual download**: Required from commonvoice.mozilla.org (v22.0)
- **Processing**: No sample limit - processes all US samples (may take significant time)

---

## 4. CORAAL - Corpus of Regional African American Language

### Overview
- **Source**: University of Oregon
- **Size**: ~150 speakers
- **Focus**: African American English varieties
- **Audio**: Long sociolinguistic interviews (30-60 min)
- **Format**: 44.1kHz WAV

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
| Code | City | State | Our Region |
|------|------|-------|------------|
| DCA | Washington DC | DC | Mid-Atlantic |
| DCB | Washington DC | DC | Mid-Atlantic |
| ATL | Atlanta | GA | Deep South |
| PRV | Princeville | NC | South Atlantic |
| VLD | Valdosta | GA | Deep South |
| ROC | Rochester | NY | New York Metropolitan |
| LES | Lower East Side | NY | New York Metropolitan |

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

## 6. SBCSAE - Santa Barbara Corpus of Spoken American English

### Overview
- **Source**: UC Santa Barbara Linguistics Department
- **Size**: ~60 hours of natural conversation
- **Conversations**: 60 recorded conversations
- **Speakers**: ~250 American English speakers
- **Focus**: Natural conversational American English
- **Audio**: High-quality recordings of unscripted conversations
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

### Download Instructions
1. Visit: https://www.linguistics.ucsb.edu/research/santa-barbara-corpus
2. Download the audio files and transcripts
3. Place WAV files in: `~/.cache/accent_datasets/sbcsae/`
4. Place transcript files alongside audio files

### Future Enhancements Needed
- Implement automatic download from UCSB servers
- Parse full 60-conversation corpus metadata
- Extract speaker segments from multi-speaker conversations
- Integrate time-aligned transcriptions
- Add speaker diarization for multi-speaker files

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

### Current Distribution (TIMIT + SAA)

| Region | TIMIT | SAA | Total | % of Dataset |
|--------|-------|-----|-------|--------------|
| Deep South | 1,980 | 0 | 1,980 | 26.4% |
| West | 1,330 | 25 | 1,355 | 18.1% |
| Upper Midwest | 1,020 | 3 | 1,023 | 13.6% |
| Lower Midwest | 1,020 | 0 | 1,020 | 13.6% |
| **South Atlantic** | 0 | 707 | 707 | 9.4% |
| New York Metropolitan | 460 | 67 | 527 | 7.0% |
| New England | 490 | 14 | 504 | 6.7% |
| **Mid-Atlantic** | 0 | 381 | 381 | 5.1% |
| **Total** | 6,300 | 1,197 | 7,497 | 100% |

### Key Insights
- SAA fills critical gaps in Mid-Atlantic and South Atlantic
- TIMIT provides strong coverage of other regions
- Combined dataset has representation for all 8 regions
- No speaker overlap between train/val/test splits

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
4. **AVOID CommonVoice**: Unreliable labels corrupt training

### Recommended Dataset Combinations

#### High Quality Training (RECOMMENDED)
```bash
# Best quality - reliable labels only
python prepare_dataset.py --datasets TIMIT SAA CORAAL
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

| Dataset | Label Quality | Location Data | Recommendation |
|---------|--------------|---------------|----------------|
| TIMIT | ✅ Excellent | Dialect regions | **Use as primary** |
| SAA | ✅ Good | State-level | **Use for coverage** |
| CORAAL | ✅ Excellent | City-specific | **Use for diversity** |
| CommonVoice | ❌ Poor | None | **AVOID** |
| PNC | ✅ Good | City-specific | Use if available |

### Why CommonVoice Fails for Regional Accents

1. **No ground truth**: Self-reported labels are unreliable
2. **No location data**: Can't verify claims about accent origin
3. **Inconsistent taxonomy**: Users report accents differently
4. **Default mapping bias**: Unknown samples arbitrarily assigned to 'West'
5. **Label noise**: Corrupts clean training signal from other datasets

### Sample Commands
```bash
# RECOMMENDED: High-quality training
python prepare_dataset.py --datasets TIMIT SAA CORAAL

# Debugging: Start simple
python prepare_dataset.py --datasets TIMIT

# Test subset
python prepare_dataset.py --datasets TIMIT SAA --max_samples_per_dataset 100

# NOT RECOMMENDED: Including CommonVoice
# python prepare_dataset.py --datasets TIMIT CommonVoice  # Will hurt performance!
```

---

*Last updated: 2025-08-17*
*Pipeline version: 1.0*