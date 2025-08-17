#!/usr/bin/env python3
"""
Download REAL SBCSAE audio files using multiple methods
"""

import requests
import subprocess
from pathlib import Path
from urllib.parse import urlparse

def try_box_download(conv_id, box_url, output_file):
    """Try to download from Box.com by parsing the HTML for actual download link"""
    try:
        # Get the Box.com page
        response = requests.get(box_url, timeout=30)
        response.raise_for_status()
        
        # Look for download patterns in the HTML
        html = response.text
        
        # Box.com often has download links in JavaScript or as data attributes
        import re
        
        # Common Box.com download URL patterns
        patterns = [
            r'download_url["\']:\s*["\']([^"\']+)["\']',
            r'href=["\']([^"\']*download[^"\']*)["\']',
            r'data-download-url=["\']([^"\']+)["\']',
            r'"download_url":"([^"]+)"',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, html, re.IGNORECASE)
            if matches:
                download_url = matches[0]
                print(f"Found potential download URL: {download_url[:100]}...")
                
                # Try to download from this URL
                dl_response = requests.get(download_url, stream=True, timeout=60)
                if dl_response.status_code == 200:
                    content_type = dl_response.headers.get('content-type', '')
                    if 'audio' in content_type or 'octet-stream' in content_type:
                        # This looks like real audio!
                        with open(output_file, 'wb') as f:
                            for chunk in dl_response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                        return True
        
        return False
        
    except Exception as e:
        print(f"Box.com download failed for {conv_id}: {e}")
        return False

def try_wget_methods(conv_id, output_file):
    """Try various wget methods to download files"""
    
    # Possible URLs to try
    urls_to_try = [
        f"https://media.talkbank.org/ca/SBC/{conv_id}.wav",
        f"https://talkbank.org/media/ca/SBC/{conv_id}.wav", 
        f"https://www.talkbank.org/data/SBC/{conv_id}.wav",
        f"https://childes.talkbank.org/media/SBC/{conv_id}.wav",
        f"https://sla.talkbank.org/TBB/ca/SBC/{conv_id}.wav",
    ]
    
    for url in urls_to_try:
        try:
            print(f"Trying: {url}")
            result = subprocess.run([
                'wget', '--timeout=30', '--tries=2', '-q', 
                '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                '-O', str(output_file), url
            ], capture_output=True, text=True)
            
            if result.returncode == 0 and output_file.exists():
                # Check if it's actually audio (not HTML)
                if output_file.stat().st_size > 10000:  # At least 10KB
                    with open(output_file, 'rb') as f:
                        header = f.read(12)
                        # Check for WAV header (RIFF...WAVE)
                        if header.startswith(b'RIFF') and b'WAVE' in header:
                            print(f"✓ Successfully downloaded {conv_id}.wav from {url}")
                            return True
                
                # Remove if not valid audio
                output_file.unlink()
            
        except Exception as e:
            print(f"Failed {url}: {e}")
            if output_file.exists():
                output_file.unlink()
    
    return False

def main():
    sbcsae_dir = Path.home() / ".cache/accent_datasets/sbcsae"
    sbcsae_dir.mkdir(parents=True, exist_ok=True)
    
    print("REAL SBCSAE Audio Downloader")
    print("=" * 40)
    print("Attempting to download actual audio files (not synthetic)")
    print()
    
    # Clean up any HTML files first
    html_files = [f for f in sbcsae_dir.glob("SBC*.wav") if f.stat().st_size < 1000]
    if html_files:
        print(f"Removing {len(html_files)} HTML files...")
        for f in html_files:
            f.unlink()
    
    # Box.com URLs (from the website)
    box_urls = {
        'SBC001': 'https://ucsb.box.com/s/c5ho8x0kuncvwoysxvtqqqxp0j1vdeu2',
        'SBC002': 'https://ucsb.box.com/s/ue89ws26hho83ei0vuzfn6gjqymsyfrh',
        'SBC003': 'https://ucsb.box.com/s/gpwue7abbd502r4hrqqa6rbum3p5s6o1',
        # Add more as needed
    }
    
    success_count = 0
    
    # Try first 5 files with different methods
    for i in range(1, 6):
        conv_id = f"SBC{i:03d}"
        output_file = sbcsae_dir / f"{conv_id}.wav"
        
        if output_file.exists() and output_file.stat().st_size > 10000:
            print(f"✓ {conv_id}.wav already exists")
            success_count += 1
            continue
        
        print(f"\nDownloading {conv_id}...")
        
        # Method 1: Try wget with various URLs
        if try_wget_methods(conv_id, output_file):
            success_count += 1
            continue
        
        # Method 2: Try Box.com parsing (if URL available)
        if conv_id in box_urls:
            if try_box_download(conv_id, box_urls[conv_id], output_file):
                success_count += 1
                continue
        
        print(f"✗ Failed to download {conv_id}.wav")
    
    print(f"\n" + "=" * 40)
    print(f"Results: {success_count}/5 files downloaded")
    
    if success_count == 0:
        print("\n❌ UNABLE TO DOWNLOAD REAL AUDIO FILES")
        print("\nThe SBCSAE audio files are not freely available for automated download.")
        print("Options:")
        print("1. Contact UC Santa Barbara for access")
        print("2. Purchase from Linguistic Data Consortium (LDC)")
        print("3. Use a different dataset (SAA, TIMIT, CORAAL work fine)")
        print("\nFor now, remove SBCSAE from your dataset list:")
        print("  python prepare_dataset.py --datasets TIMIT SAA CORAAL")
    else:
        print(f"\n✓ Successfully downloaded {success_count} files!")
        print("SBCSAE is ready to use (with available files)")

if __name__ == "__main__":
    main()