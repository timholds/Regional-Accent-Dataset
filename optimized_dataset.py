#!/usr/bin/env python3
"""
Optimized Dataset with Audio Caching for Regional Accent Classification

This dataset provides audio caching capabilities to speed up training
when using pre-chunked data.
"""

import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
from typing import Dict, List, Optional, Union
from collections import OrderedDict
import logging

from unified_dataset import UnifiedSample

logger = logging.getLogger(__name__)


class AudioCache:
    """LRU cache for audio files"""
    
    def __init__(self, cache_size: int = 200):
        self.cache_size = cache_size
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def get(self, audio_path: str, chunk_start: Optional[int] = None, chunk_end: Optional[int] = None) -> Optional[np.ndarray]:
        """Get audio from cache or return None if not cached"""
        cache_key = self._make_key(audio_path, chunk_start, chunk_end)
        
        if cache_key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(cache_key)
            self.hits += 1
            return self.cache[cache_key]
        
        self.misses += 1
        return None
    
    def put(self, audio_path: str, audio: np.ndarray, chunk_start: Optional[int] = None, chunk_end: Optional[int] = None):
        """Put audio in cache"""
        cache_key = self._make_key(audio_path, chunk_start, chunk_end)
        
        # Remove oldest if cache is full
        if len(self.cache) >= self.cache_size:
            self.cache.popitem(last=False)
        
        self.cache[cache_key] = audio.copy()  # Store a copy to avoid modifications
    
    def _make_key(self, audio_path: str, chunk_start: Optional[int], chunk_end: Optional[int]) -> str:
        """Create cache key"""
        if chunk_start is not None and chunk_end is not None:
            return f"{audio_path}:{chunk_start}:{chunk_end}"
        return audio_path
    
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            'size': len(self.cache),
            'capacity': self.cache_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate
        }
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0


class OptimizedAccentDataset(Dataset):
    """
    Optimized dataset with audio caching for pre-chunked accent classification data.
    
    This dataset provides significant speedup when:
    1. Using pre-chunked data where chunk boundaries are pre-calculated
    2. Multiple epochs are being trained (cache helps on subsequent epochs)
    3. Audio files are accessed multiple times during training
    """
    
    def __init__(
        self,
        samples: List[UnifiedSample],
        processor,
        label_mapping: Optional[Dict[str, int]] = None,
        pre_chunked: bool = False,
        cache_size: int = 200,
        target_sample_rate: int = 16000
    ):
        self.samples = samples
        self.processor = processor
        self.pre_chunked = pre_chunked
        self.target_sample_rate = target_sample_rate
        
        # Setup audio cache
        self.audio_cache = AudioCache(cache_size)
        
        # Create or use provided label mapping
        if label_mapping is None:
            unique_labels = sorted(set(sample.region_label for sample in samples))
            self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.label_mapping = label_mapping
        
        # Log dataset info
        logger.info(f"OptimizedAccentDataset initialized:")
        logger.info(f"  Samples: {len(self.samples)}")
        logger.info(f"  Pre-chunked: {self.pre_chunked}")
        logger.info(f"  Audio cache size: {cache_size}")
        logger.info(f"  Labels: {len(self.label_mapping)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load audio with caching
        audio = self._load_audio_cached(sample)
        
        # Ensure consistent audio length (pad or truncate to max_length)
        max_length_samples = int(self.target_sample_rate * 10)  # 10 seconds max (matches chunking strategy)
        
        if len(audio) > max_length_samples:
            audio = audio[:max_length_samples]  # Truncate
        elif len(audio) < max_length_samples:
            audio = np.pad(audio, (0, max_length_samples - len(audio)))  # Pad with zeros
        
        # Process audio (now guaranteed to be consistent length)
        inputs = self.processor(
            audio,
            sampling_rate=self.target_sample_rate,
            return_tensors="pt",
            padding=False,  # No need to pad since we already did it
            truncation=False  # No need to truncate since we already did it
        )
        
        # Create attention mask manually (Wav2Vec2Processor doesn't provide one)
        attention_mask = torch.ones(inputs.input_values.size(-1), dtype=torch.float32)
        
        # Get label
        label = self.label_mapping[sample.region_label]
        
        return {
            'input_values': inputs.input_values.squeeze(),
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long),
            'sample_id': sample.sample_id
        }
    
    def _load_audio_cached(self, sample: UnifiedSample) -> np.ndarray:
        """Load audio with caching support"""
        
        # Determine if this is a chunked sample
        chunk_start = None
        chunk_end = None
        
        if self.pre_chunked and hasattr(sample, 'chunk_start_sample') and hasattr(sample, 'chunk_end_sample'):
            chunk_start = sample.chunk_start_sample
            chunk_end = sample.chunk_end_sample
        
        # Try to get from cache first
        cached_audio = self.audio_cache.get(sample.audio_path, chunk_start, chunk_end)
        if cached_audio is not None:
            return cached_audio
        
        # Cache miss - load from disk
        try:
            if chunk_start is not None and chunk_end is not None:
                # Load specific chunk
                audio, sr = librosa.load(
                    sample.audio_path,
                    sr=self.target_sample_rate,
                    offset=chunk_start / self.target_sample_rate,
                    duration=(chunk_end - chunk_start) / self.target_sample_rate
                )
            else:
                # Load full audio
                audio, sr = librosa.load(sample.audio_path, sr=self.target_sample_rate)
            
            # Cache the loaded audio
            self.audio_cache.put(sample.audio_path, audio, chunk_start, chunk_end)
            
            return audio
            
        except Exception as e:
            logger.error(f"Error loading audio {sample.audio_path}: {e}")
            # Return silence as fallback
            return np.zeros(int(self.target_sample_rate * 1.0))  # 1 second of silence
    
    def get_cache_stats(self) -> Dict[str, Union[int, float]]:
        """Get audio cache statistics"""
        return self.audio_cache.get_stats()
    
    def clear_cache(self):
        """Clear audio cache"""
        self.audio_cache.clear()
    
    def log_cache_stats(self):
        """Log cache statistics"""
        stats = self.get_cache_stats()
        logger.info(f"Audio cache stats: {stats['hits']} hits, {stats['misses']} misses, "
                   f"{stats['hit_rate']:.1%} hit rate, {stats['size']}/{stats['capacity']} capacity")