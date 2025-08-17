#!/usr/bin/env python3
"""
Download CORAAL dataset components with proper error handling and SSL bypass
"""

import os
import sys
import requests
import tarfile
from pathlib import Path
from tqdm import tqdm
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# CORAAL components with working URLs
CORAAL_COMPONENTS = {
    'ATL': {
        'city': 'Atlanta', 
        'region': 'Deep South',
        'audio_urls': [
            'https://lingtools.uoregon.edu/coraal/atl/2020.05/ATL_audio_part01_2020.05.tar.gz',
            'https://lingtools.uoregon.edu/coraal/atl/2020.05/ATL_audio_part02_2020.05.tar.gz',
            'https://lingtools.uoregon.edu/coraal/atl/2020.05/ATL_audio_part03_2020.05.tar.gz',
            'https://lingtools.uoregon.edu/coraal/atl/2020.05/ATL_audio_part04_2020.05.tar.gz'
        ]
    },
    'DCA': {
        'city': 'Washington DC', 
        'region': 'Mid-Atlantic',
        'audio_urls': [
            'https://lingtools.uoregon.edu/coraal/dca/2018.10.06/DCA_audio_part01_2018.10.06.tar.gz',
            'https://lingtools.uoregon.edu/coraal/dca/2018.10.06/DCA_audio_part02_2018.10.06.tar.gz'
        ]
    },
    'ROC': {
        'city': 'Rochester NY', 
        'region': 'New York Metropolitan',
        'audio_urls': [
            'https://lingtools.uoregon.edu/coraal/roc/2020.05/ROC_audio_part01_2020.05.tar.gz',
            'https://lingtools.uoregon.edu/coraal/roc/2020.05/ROC_audio_part02_2020.05.tar.gz'
        ]
    },
    'LES': {
        'city': 'Lower East Side NYC', 
        'region': 'New York Metropolitan',
        'audio_urls': [
            'https://lingtools.uoregon.edu/coraal/les/2021.04/LES_audio_part01_2021.04.tar.gz'
        ]
    },
    'DTA': {
        'city': 'Detroit', 
        'region': 'Midwest',
        'audio_urls': [
            'https://lingtools.uoregon.edu/coraal/dta/2023.06/DTA_audio_part01_2023.06.tar.gz',
            'https://lingtools.uoregon.edu/coraal/dta/2023.06/DTA_audio_part02_2023.06.tar.gz'
        ]
    },
}


def download_file(url, dest_path, description="Downloading"):
    """Download a file with progress bar"""
    try:
        # Stream download with SSL verification disabled
        response = requests.get(url, stream=True, verify=False, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=description) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        return True
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Download failed: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False
    except KeyboardInterrupt:
        print("\n  ⚠️  Download interrupted by user")
        if dest_path.exists():
            dest_path.unlink()
        return False


def extract_audio_files(tar_path, audio_dir):
    """Extract only WAV files from tar.gz"""
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            # Get all WAV files
            wav_members = [m for m in tar.getmembers() if m.name.endswith('.wav')]
            
            print(f"  📦 Extracting {len(wav_members)} WAV files...")
            
            for member in tqdm(wav_members, desc="Extracting"):
                # Extract with flattened path (just filename)
                member.name = os.path.basename(member.name)
                tar.extract(member, audio_dir)
        
        return len(wav_members)
    except Exception as e:
        print(f"  ❌ Extraction failed: {e}")
        return 0


def download_component(component_name, component_info, base_dir):
    """Download a single CORAAL component"""
    print(f"\n{'='*60}")
    print(f"📍 {component_name}: {component_info['city']} ({component_info['region']})")
    print(f"{'='*60}")
    
    # Create directories
    component_dir = base_dir / component_name
    audio_dir = component_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    existing_wavs = list(audio_dir.glob("*.wav"))
    if existing_wavs:
        print(f"  ✅ Already have {len(existing_wavs)} WAV files, skipping...")
        return len(existing_wavs)
    
    total_wavs = 0
    
    # Download each part
    for i, url in enumerate(component_info['audio_urls'], 1):
        print(f"\n  Part {i}/{len(component_info['audio_urls'])}:")
        
        # Download
        tar_path = base_dir / f"{component_name}_part{i:02d}.tar.gz"
        filename = os.path.basename(url)
        
        if not download_file(url, tar_path, f"  Downloading {filename}"):
            continue
        
        # Extract
        wav_count = extract_audio_files(tar_path, audio_dir)
        total_wavs += wav_count
        
        # Clean up tar file
        tar_path.unlink()
        print(f"  ✅ Extracted {wav_count} WAV files")
    
    return total_wavs


def main():
    # Setup paths
    cache_dir = Path.home() / ".cache" / "accent_datasets" / "coraal"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("🎯 CORAAL Dataset Downloader")
    print(f"📂 Download directory: {cache_dir}")
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            components_to_download = list(CORAAL_COMPONENTS.keys())
        else:
            components_to_download = [arg.upper() for arg in sys.argv[1:]]
    else:
        # Interactive mode
        print("\nAvailable components:")
        for i, (name, info) in enumerate(CORAAL_COMPONENTS.items(), 1):
            print(f"  {i}. {name}: {info['city']} ({info['region']})")
        
        print("\nOptions:")
        print("  - Press Enter to download ALL components")
        print("  - Enter component names (e.g., 'ATL DCA') to download specific ones")
        print("  - Enter 'q' to quit")
        
        try:
            choice = input("\nYour choice: ").strip()
        except EOFError:
            # Non-interactive mode, download priority components
            print("\nNon-interactive mode detected. Downloading priority components...")
            components_to_download = ['DCA', 'ROC', 'LES']
        else:
            if choice.lower() == 'q':
                print("Exiting...")
                return
            
            if choice == "":
                components_to_download = list(CORAAL_COMPONENTS.keys())
            else:
                components_to_download = choice.upper().split()
    
    # Download selected components
    total_files = 0
    for component in components_to_download:
        if component not in CORAAL_COMPONENTS:
            print(f"⚠️  Unknown component: {component}")
            continue
        
        wav_count = download_component(
            component, 
            CORAAL_COMPONENTS[component], 
            cache_dir
        )
        total_files += wav_count
    
    print(f"\n{'='*60}")
    print(f"✅ Download complete!")
    print(f"📊 Total WAV files: {total_files}")
    print(f"📂 Location: {cache_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()