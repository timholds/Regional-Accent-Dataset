#!/usr/bin/env python3
"""
Helper script to download/prepare additional datasets for testing
"""

import os
import sys
from pathlib import Path
import requests
import zipfile
import tarfile
from tqdm import tqdm


def download_file(url: str, dest_path: Path, desc: str = "Downloading"):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


def setup_common_voice_sample():
    """Download a small sample of Common Voice for testing"""
    print("\n" + "="*60)
    print("Mozilla Common Voice Setup")
    print("="*60)
    
    print("\nFor the full Common Voice dataset (~2.5GB):")
    print("1. Visit: https://commonvoice.mozilla.org/en/datasets")
    print("2. Download the English dataset")
    print("3. Place the tar.gz file in: ~/.cache/accent_datasets/common_voice/")
    
    print("\nFor testing, we'll create a mock structure...")
    
    # Create mock directory structure
    cache_dir = Path.home() / ".cache" / "accent_datasets"
    cv_dir = cache_dir / "common_voice" / "cv-corpus-17.0-2024-03-15"
    cv_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a sample validated.tsv file with US accent entries
    sample_tsv = """client_id\tpath\tsentence\tup_votes\tdown_votes\tage\tgender\taccent\tlocale\tsegment\tvariant
abc123\tsample1.mp3\tThe quick brown fox jumps over the lazy dog.\t2\t0\tfourties\tmale\tus english california\ten\ttrain\t
def456\tsample2.mp3\tShe sells seashells by the seashore.\t1\t0\ttwenties\tfemale\tnew york city\ten\ttrain\t
ghi789\tsample3.mp3\tHow much wood would a woodchuck chuck.\t3\t0\tthirties\tmale\tsouthern united states\ten\ttrain\t
jkl012\tsample4.mp3\tPeter Piper picked a peck of pickled peppers.\t2\t0\tfifties\tfemale\tmidwestern united states\ten\ttrain\t
mno345\tsample5.mp3\tThe rain in Spain stays mainly in the plain.\t1\t0\ttwenties\tmale\tboston massachusetts\ten\ttrain\t
"""
    
    with open(cv_dir / "validated.tsv", 'w') as f:
        f.write(sample_tsv)
    
    print(f"✓ Created sample Common Voice structure at: {cv_dir}")
    print("  Note: This is just for testing. Download the real dataset for training.")


def setup_coraal_sample():
    """Setup CORAAL sample structure"""
    print("\n" + "="*60)
    print("CORAAL Setup")
    print("="*60)
    
    print("\nCORAAL is a professionally curated corpus.")
    print("To download:")
    print("1. Visit: http://lingtools.uoregon.edu/coraal/")
    print("2. Register for access (free for research)")
    print("3. Download components: ATL, DCA, LES, ROC recommended")
    print("4. Extract to: ~/.cache/accent_datasets/coraal/")
    
    print("\nExpected structure:")
    print("  coraal/")
    print("    ATL/")
    print("      audio/")
    print("        ATL_se0_ag1_f_01.wav")
    print("        ATL_se0_ag2_m_01.wav")
    print("        ...")
    print("      ATL_metadata.txt")
    
    # Create mock structure for testing
    cache_dir = Path.home() / ".cache" / "accent_datasets"
    coraal_dir = cache_dir / "coraal"
    
    # Create sample component
    atl_dir = coraal_dir / "ATL" / "audio"
    atl_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n✓ Created CORAAL directory structure at: {coraal_dir}")
    print("  Note: You'll need to download actual audio files from the CORAAL website.")


def check_datasets():
    """Check which datasets are available"""
    print("\n" + "="*60)
    print("Dataset Status Check")
    print("="*60)
    
    cache_dir = Path.home() / ".cache" / "accent_datasets"
    
    # Check TIMIT
    timit_path = Path(".") / "data" / "data"
    if timit_path.exists() and (timit_path / "TRAIN").exists():
        print("✓ TIMIT: Found")
    else:
        print("✗ TIMIT: Not found at ./data/data/")
    
    # Check Common Voice
    cv_path = cache_dir / "common_voice"
    if cv_path.exists() and any(cv_path.glob("*/validated.tsv")):
        print("✓ Common Voice: Found (or sample created)")
    else:
        print("✗ Common Voice: Not found")
    
    # Check CORAAL
    coraal_path = cache_dir / "coraal"
    if coraal_path.exists() and any(coraal_path.glob("*/audio")):
        print("✓ CORAAL: Directory structure found")
        audio_files = list(coraal_path.glob("*/audio/*.wav"))
        if audio_files:
            print(f"  - {len(audio_files)} audio files found")
        else:
            print("  - No audio files found (download from CORAAL website)")
    else:
        print("✗ CORAAL: Not found")


def main():
    print("Regional Accent Dataset Setup Helper")
    print("====================================")
    
    # Setup sample datasets
    setup_common_voice_sample()
    setup_coraal_sample()
    
    # Check status
    check_datasets()
    
    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    print("1. For Common Voice: Download the full dataset from Mozilla")
    print("2. For CORAAL: Register and download from the official website")
    print("3. Run: python test_unified_dataset.py")
    print("   to test the unified pipeline with all datasets")


if __name__ == "__main__":
    main()