#!/bin/bash

# CORAAL Full Dataset Downloader
# Downloads all components with resume support

CACHE_DIR="$HOME/.cache/accent_datasets/coraal"
mkdir -p "$CACHE_DIR"

echo "🎯 CORAAL Full Dataset Downloader"
echo "📂 Target directory: $CACHE_DIR"
echo "⚡ Using parallel downloads with resume support"
echo ""

# Define all CORAAL components and their URLs
declare -A COMPONENTS=(
    ["ATL"]="Atlanta (Deep South)"
    ["DCA"]="Washington DC (Mid-Atlantic)" 
    ["ROC"]="Rochester NY (New York Metropolitan)"
    ["LES"]="Lower East Side NYC (New York Metropolitan)"
    ["DTA"]="Detroit (Upper Midwest)"
)

# Download function with resume support
download_component() {
    local component=$1
    local description=$2
    local base_url=$3
    
    echo "========================================"
    echo "📍 Downloading $component: $description"
    echo "========================================"
    
    mkdir -p "$CACHE_DIR/$component/audio"
    cd "$CACHE_DIR"
    
    # Download all parts for this component
    for part_url in $base_url; do
        filename=$(basename "$part_url")
        echo "⬇️  Downloading $filename..."
        
        # Use wget with resume, retry, and no certificate check
        wget -c --no-check-certificate --tries=5 --timeout=30 \
             --progress=bar:force "$part_url" \
             -O "$filename" 2>&1 | grep -E "%" || true
        
        if [ -f "$filename" ]; then
            echo "📦 Extracting $filename..."
            # Extract only WAV files, flatten directory structure
            tar -xzf "$filename" --wildcards "*.wav" --transform='s/.*\///' \
                -C "$component/audio/" 2>/dev/null || \
            tar -xf "$filename" --wildcards "*.wav" --transform='s/.*\///' \
                -C "$component/audio/" 2>/dev/null || \
            echo "⚠️  Failed to extract $filename"
            
            # Count extracted files
            wav_count=$(ls -1 "$component/audio/"*.wav 2>/dev/null | wc -l)
            echo "✅ $component now has $wav_count WAV files"
        fi
    done
}

# ATL component (4 parts)
download_component "ATL" "${COMPONENTS[ATL]}" \
    "https://lingtools.uoregon.edu/coraal/atl/2020.05/ATL_audio_part01_2020.05.tar.gz \
     https://lingtools.uoregon.edu/coraal/atl/2020.05/ATL_audio_part02_2020.05.tar.gz \
     https://lingtools.uoregon.edu/coraal/atl/2020.05/ATL_audio_part03_2020.05.tar.gz \
     https://lingtools.uoregon.edu/coraal/atl/2020.05/ATL_audio_part04_2020.05.tar.gz"

# DCA component (2 parts)  
download_component "DCA" "${COMPONENTS[DCA]}" \
    "https://lingtools.uoregon.edu/coraal/dca/2018.10.06/DCA_audio_part01_2018.10.06.tar.gz \
     https://lingtools.uoregon.edu/coraal/dca/2018.10.06/DCA_audio_part02_2018.10.06.tar.gz"

# ROC component (2 parts)
download_component "ROC" "${COMPONENTS[ROC]}" \
    "https://lingtools.uoregon.edu/coraal/roc/2020.05/ROC_audio_part01_2020.05.tar.gz \
     https://lingtools.uoregon.edu/coraal/roc/2020.05/ROC_audio_part02_2020.05.tar.gz"

# LES component (1 part) - Note: URL might be different
download_component "LES" "${COMPONENTS[LES]}" \
    "https://lingtools.uoregon.edu/coraal/les/2021.07/LES_audio_2021.07.tar.gz"

# DTA component (2 parts)
download_component "DTA" "${COMPONENTS[DTA]}" \
    "https://lingtools.uoregon.edu/coraal/dta/2023.06/DTA_audio_part01_2023.06.tar.gz \
     https://lingtools.uoregon.edu/coraal/dta/2023.06/DTA_audio_part02_2023.06.tar.gz"

# Final summary
echo ""
echo "========================================"
echo "📊 FINAL SUMMARY"
echo "========================================"
total_wavs=0
for component in "${!COMPONENTS[@]}"; do
    count=$(ls -1 "$CACHE_DIR/$component/audio/"*.wav 2>/dev/null | wc -l)
    echo "$component: $count WAV files"
    total_wavs=$((total_wavs + count))
done
echo "----------------------------------------"
echo "TOTAL: $total_wavs WAV files"
echo "========================================"