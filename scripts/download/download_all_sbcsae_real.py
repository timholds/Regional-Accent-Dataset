#!/usr/bin/env python3
"""
Download ALL real SBCSAE files using the working Box.com pattern
"""

import subprocess
import os
from pathlib import Path

def download_sbcsae_file(conv_id, box_id, sbcsae_dir):
    """Download a single SBCSAE file using the working Box pattern"""
    output_file = sbcsae_dir / f"{conv_id}.wav"
    
    if output_file.exists() and output_file.stat().st_size > 1000000:  # At least 1MB
        print(f"✓ {conv_id}.wav already exists ({output_file.stat().st_size:,} bytes)")
        return True
    
    url = f"https://app.box.com/shared/static/{box_id}.wav"
    print(f"Downloading {conv_id}.wav...")
    
    try:
        result = subprocess.run([
            'wget', '--timeout=60', '--tries=2', '-q',
            '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            '-O', str(output_file), url
        ], capture_output=True, text=True)
        
        if result.returncode == 0 and output_file.exists():
            # Verify it's a real WAV file
            file_result = subprocess.run(['file', str(output_file)], capture_output=True, text=True)
            if 'RIFF' in file_result.stdout and 'WAVE' in file_result.stdout:
                size = output_file.stat().st_size
                print(f"✓ Downloaded {conv_id}.wav ({size:,} bytes)")
                return True
            else:
                output_file.unlink()
                print(f"✗ {conv_id}.wav - not a valid WAV file")
        else:
            if output_file.exists():
                output_file.unlink()
            print(f"✗ Failed to download {conv_id}.wav")
        
        return False
        
    except Exception as e:
        print(f"✗ Error downloading {conv_id}.wav: {e}")
        if output_file.exists():
            output_file.unlink()
        return False

def main():
    # Box.com IDs extracted from the UCSB website
    box_files = {
        'SBC001': 'c5ho8x0kuncvwoysxvtqqqxp0j1vdeu2',
        'SBC002': 'ue89ws26hho83ei0vuzfn6gjqymsyfrh', 
        'SBC003': 'gpwue7abbd502r4hrqqa6rbum3p5s6o1',
        'SBC004': 'vykixuesz9w4ag2qg097wvv3psub5nzm',
        'SBC005': '0wadkhacf960exb0kc59pn4lh159pcmb',
        'SBC006': 'kfcqqhfzlo8flttut6tu71vbj4ov3736',
        'SBC007': 't3umsdhaeet0clisfmqxpftkqt5my5qw',
        'SBC008': 'gs7ekaere42rfmopb8cz3u0wdlkozrak',
        'SBC009': 'ofuyr8mgy9iwwavgkyyyjz9sktxqhmku',
        'SBC010': 'pe8rkk8nmt9g8jvmgikbjk6fwggesf83',
    }
    
    sbcsae_dir = Path.home() / ".cache/accent_datasets/sbcsae"
    sbcsae_dir.mkdir(parents=True, exist_ok=True)
    
    print("REAL SBCSAE Downloader")
    print("=" * 40)
    print(f"Downloading to: {sbcsae_dir}")
    print()
    
    success_count = 0
    
    for conv_id, box_id in box_files.items():
        if download_sbcsae_file(conv_id, box_id, sbcsae_dir):
            success_count += 1
    
    print("\n" + "=" * 40)
    print(f"Download Results: {success_count}/{len(box_files)} files")
    
    if success_count > 0:
        print(f"\n🎉 SUCCESS! Downloaded {success_count} real SBCSAE audio files!")
        
        # Show file info
        wav_files = list(sbcsae_dir.glob("SBC*.wav"))
        if wav_files:
            total_size = sum(f.stat().st_size for f in wav_files)
            print(f"📊 Total size: {total_size/1024/1024:.1f} MB")
            
            # Check audio properties of first file
            first_file = wav_files[0]
            file_result = subprocess.run(['file', str(first_file)], capture_output=True, text=True)
            print(f"📡 Audio format: {file_result.stdout.strip().split(': ')[1]}")
        
        print(f"\n📝 Next steps:")
        print("1. These are long conversations (20-60 minutes each)")
        print("2. Need to segment them into shorter clips (5-10 seconds)")
        print("3. Map segments to speakers using metadata")
        print("\nTest current loader:")
        print("  python prepare_dataset.py --datasets SBCSAE --max_samples_per_dataset 5")
    else:
        print("\n❌ No files downloaded successfully")

if __name__ == "__main__":
    main()