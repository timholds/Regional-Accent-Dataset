#!/usr/bin/env python3
"""
Encoder output caching for frozen wav2vec2 models

This caches the outputs of the frozen wav2vec2 encoder,
which provides massive speedup when only training the classifier head.
"""

import os
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional
import torch
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime

from unified_dataset import UnifiedAccentDatasetTorch

logger = logging.getLogger(__name__)


class EncoderCachedDataset(UnifiedAccentDatasetTorch):
    """
    Dataset that caches wav2vec2 encoder outputs for frozen encoder training.
    
    This is much faster than SimpleCachedDataset because it caches the 
    encoder outputs (after the expensive transformer layers) rather than
    just the preprocessed audio.
    """
    
    def __init__(self, *args, 
                 model=None,  # The wav2vec2 model to extract features
                 cache_dir: str = "encoder_cache",
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create cache key
        cache_key = self._compute_cache_key()
        self.cache_path = self.cache_dir / cache_key
        self.cache_path.mkdir(parents=True, exist_ok=True)
        
        # Pre-cache all features if model provided
        if self.model is not None:
            self._ensure_all_cached()
        
        logger.info(f"Encoder cache at: {self.cache_path}")
    
    def _compute_cache_key(self) -> str:
        """Generate cache key based on model and dataset"""
        key_components = {
            'target_sr': self.target_sr,
            'max_length': self.max_length,
            'processor': str(self.processor.feature_extractor.to_dict())[:100],
            'num_samples': len(self.samples),
            'sample_ids_hash': hashlib.md5(
                ''.join(sorted(s.sample_id for s in self.samples)).encode()
            ).hexdigest()[:8]
        }
        
        # Include model info if available
        if self.model is not None:
            key_components['model_config'] = str(self.model.config)[:100]
        
        key_str = str(key_components)
        return hashlib.md5(key_str.encode()).hexdigest()[:12]
    
    def _ensure_all_cached(self):
        """Pre-extract and cache all encoder outputs"""
        if self.model is None:
            return
        
        # Check how many are already cached
        cached_count = sum(
            1 for s in self.samples 
            if (self.cache_path / f"{s.sample_id}.pt").exists()
        )
        
        if cached_count == len(self.samples):
            logger.info(f"All {cached_count} samples already cached")
            return
        
        logger.info(f"Caching encoder outputs: {cached_count}/{len(self.samples)} already cached")
        
        # Put model in eval mode
        self.model.eval()
        device = next(self.model.parameters()).device
        
        with torch.no_grad():
            for idx, sample in enumerate(tqdm(self.samples, desc="Caching encoder outputs")):
                cache_file = self.cache_path / f"{sample.sample_id}.pt"
                
                if cache_file.exists():
                    continue
                
                try:
                    # Get preprocessed audio from parent class
                    item = super().__getitem__(idx)
                    
                    # Move to device and add batch dimension
                    input_values = item['input_values'].unsqueeze(0).to(device)
                    attention_mask = item['attention_mask'].unsqueeze(0).to(device)
                    
                    # Run through encoder only
                    outputs = self.model.wav2vec2(
                        input_values=input_values,
                        attention_mask=attention_mask
                    )
                    
                    # Cache the encoder output
                    cache_data = {
                        'encoder_output': outputs.last_hidden_state.cpu(),
                        'attention_mask': attention_mask.cpu(),
                        'labels': item['labels'],
                        'sample_id': sample.sample_id
                    }
                    
                    torch.save(cache_data, cache_file)
                    
                except Exception as e:
                    logger.error(f"Failed to cache {sample.sample_id}: {e}")
                    continue
                
                # Periodically clear GPU memory
                if (idx + 1) % 100 == 0 and device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        logger.info(f"Caching complete!")
    
    def __getitem__(self, idx):
        """Return cached encoder outputs or compute on the fly"""
        sample = self.samples[idx]
        cache_file = self.cache_path / f"{sample.sample_id}.pt"
        
        if cache_file.exists():
            try:
                cache_data = torch.load(cache_file, map_location='cpu')
                # Return encoder outputs directly
                return {
                    'input_values': cache_data['encoder_output'].squeeze(0),  # Remove batch dim
                    'attention_mask': cache_data['attention_mask'].squeeze(0),
                    'labels': cache_data['labels'],
                    'sample_id': cache_data['sample_id'],
                    'is_encoder_output': True  # Flag to indicate these are encoder outputs
                }
            except Exception as e:
                logger.warning(f"Failed to load cache for {sample.sample_id}: {e}")
        
        # Fall back to parent implementation
        item = super().__getitem__(idx)
        item['is_encoder_output'] = False
        return item