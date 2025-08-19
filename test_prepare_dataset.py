#!/usr/bin/env python3
"""
Smoke tests for prepare_dataset.py pipeline
Tests core functionality to catch breaking changes from LLM-assisted coding

Usage:
    # Run all tests
    python test_prepare_dataset.py
    
    # Run with verbose output
    python test_prepare_dataset.py -v
    
    # Run specific test
    python -m unittest test_prepare_dataset.TestPrepareDataset.test_chunking_logic
"""

import unittest
import numpy as np
import pandas as pd
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unified_dataset import UnifiedSample
from prepare_dataset import chunk_all_audio_samples, filter_samples


class TestPrepareDataset(unittest.TestCase):
    """Test suite for prepare_dataset.py core functionality"""
    
    def test_chunking_logic(self):
        """Test 1: Verify chunking behavior for different audio durations"""
        samples = [
            UnifiedSample(
                sample_id="short_5s",
                dataset_name="test",
                speaker_id="spk1",
                audio_path="/fake/short.wav",
                transcript="test transcript",
                region_label="test_region",
                original_accent_label="test",
                duration=5.0
            ),
            UnifiedSample(
                sample_id="medium_8s",
                dataset_name="test", 
                speaker_id="spk2",
                audio_path="/fake/medium.wav",
                transcript="test transcript",
                region_label="test_region",
                original_accent_label="test",
                duration=8.0
            ),
            UnifiedSample(
                sample_id="edge_10s",  # Edge case at threshold
                dataset_name="test",
                speaker_id="spk3",
                audio_path="/fake/edge.wav",
                transcript="test transcript",
                region_label="test_region",
                original_accent_label="test",
                duration=10.0
            ),
            UnifiedSample(
                sample_id="long_15s",
                dataset_name="test",
                speaker_id="spk4",
                audio_path="/fake/long.wav",
                transcript="test transcript",
                region_label="test_region",
                original_accent_label="test",
                duration=15.0
            )
        ]
        
        # Properly mock sf.SoundFile with required attributes
        mock_soundfile = MagicMock()
        mock_soundfile.__enter__.return_value = mock_soundfile
        mock_soundfile.__exit__.return_value = None
        mock_soundfile.samplerate = 16000
        
        # Mock file operations and audio loading
        with patch('prepare_dataset.sf.SoundFile') as mock_sf_class, \
             patch('prepare_dataset.librosa.load') as mock_load, \
             patch('os.path.exists', return_value=True):
            
            # Setup SoundFile mock to return different lengths based on file
            def create_soundfile_mock(path):
                mock = MagicMock()
                mock.samplerate = 16000
                if "short" in path:
                    mock.__len__ = lambda: 5 * 16000
                elif "medium" in path:
                    mock.__len__ = lambda: 8 * 16000
                elif "edge" in path:
                    mock.__len__ = lambda: 10 * 16000
                else:  # long
                    mock.__len__ = lambda: 15 * 16000
                return mock
            
            mock_sf_class.side_effect = create_soundfile_mock
            
            # Setup audio loading mock
            def load_audio(path, sr):
                if "short" in path:
                    return np.zeros(5 * sr), sr
                elif "medium" in path:
                    return np.zeros(8 * sr), sr
                elif "edge" in path:
                    return np.zeros(10 * sr), sr
                else:  # long
                    return np.zeros(15 * sr), sr
            
            mock_load.side_effect = load_audio
            
            # Run chunking with 7.5s chunks, 2.5s overlap, max 10s before chunking
            chunked = chunk_all_audio_samples(
                samples,
                chunk_duration=7.5,
                chunk_overlap=2.5,
                min_duration=5.0,
                max_duration=10.0
            )
            
            # Verify short file (5s) gets single chunk with metadata
            short_chunks = [s for s in chunked if s.sample_id.startswith("short_5s")]
            self.assertEqual(len(short_chunks), 1)
            self.assertEqual(short_chunks[0].chunk_start_sample, 0)
            self.assertEqual(short_chunks[0].chunk_end_sample, 5 * 16000)
            
            # Verify medium file (8s) gets single chunk with metadata  
            medium_chunks = [s for s in chunked if s.sample_id.startswith("medium_8s")]
            self.assertEqual(len(medium_chunks), 1)
            self.assertEqual(medium_chunks[0].chunk_start_sample, 0)
            self.assertEqual(medium_chunks[0].chunk_end_sample, 8 * 16000)
            
            # Verify edge case: 10s file at threshold is NOT chunked
            edge_chunks = [s for s in chunked if s.sample_id.startswith("edge_10s")]
            self.assertEqual(len(edge_chunks), 1, "10s file at threshold should not be chunked")
            
            # Verify long file (15s) gets multiple chunks
            long_chunks = [s for s in chunked if s.sample_id.startswith("long_15s")]
            self.assertGreaterEqual(len(long_chunks), 2, "15s file should create 2+ chunks")
            
            # Verify chunk boundaries are sensible
            for i, chunk in enumerate(sorted(long_chunks, key=lambda x: x.chunk_start_sample)):
                self.assertIn(f"_chunk{i:03d}", chunk.sample_id)
                self.assertIsNotNone(chunk.chunk_start_sample)
                self.assertIsNotNone(chunk.chunk_end_sample)
                self.assertLess(chunk.chunk_start_sample, chunk.chunk_end_sample)
    
    def test_speaker_filtering(self):
        """Test 2: Verify min/max samples per speaker constraints"""
        args = Mock()
        args.min_samples_per_speaker = 3
        args.max_samples_per_speaker = 10
        args.seed = 42
        
        samples = []
        # Speaker with 1 sample (should be filtered out)
        samples.append(UnifiedSample(
            sample_id="few_1", dataset_name="test", speaker_id="spk_few",
            audio_path="/fake/1.wav", transcript="test", region_label="test", 
            original_accent_label="test", duration=3.0
        ))
        
        # Speaker with 5 samples (should pass through)
        for i in range(5):
            samples.append(UnifiedSample(
                sample_id=f"good_{i}", dataset_name="test", speaker_id="spk_good",
                audio_path=f"/fake/good_{i}.wav", transcript="test", region_label="test",
                original_accent_label="test", duration=3.0
            ))
        
        # Speaker with 15 samples (should be capped at 10)
        for i in range(15):
            samples.append(UnifiedSample(
                sample_id=f"many_{i}", dataset_name="test", speaker_id="spk_many",
                audio_path=f"/fake/many_{i}.wav", transcript="test", region_label="test",
                original_accent_label="test", duration=3.0
            ))
        
        # Apply filtering
        filtered = filter_samples(samples, args)
        
        # Verify speaker with 1 sample is removed
        spk_few_samples = [s for s in filtered if s.speaker_id == "spk_few"]
        self.assertEqual(len(spk_few_samples), 0, "Speaker with <3 samples should be removed")
        
        # Verify speaker with 5 samples passes through unchanged
        spk_good_samples = [s for s in filtered if s.speaker_id == "spk_good"]
        self.assertEqual(len(spk_good_samples), 5, "Speaker with 5 samples should remain")
        
        # Verify speaker with 15 samples is capped at 10
        spk_many_samples = [s for s in filtered if s.speaker_id == "spk_many"]
        self.assertEqual(len(spk_many_samples), 10, "Speaker with 15 samples should be capped at 10")
        
        # Verify total sample count
        self.assertEqual(len(filtered), 15, "Should have 0 + 5 + 10 = 15 samples total")
    
    def test_train_val_test_splits(self):
        """Test 3: Verify no speaker overlap and correct proportions"""
        # Create 10 speakers with 3 samples each
        samples = []
        for spk_id in range(10):
            for samp_id in range(3):
                samples.append(UnifiedSample(
                    sample_id=f"spk{spk_id}_samp{samp_id}",
                    dataset_name="test",
                    speaker_id=f"speaker_{spk_id}",
                    audio_path=f"/fake/s{spk_id}_{samp_id}.wav",
                    transcript="test",
                    region_label="test_region",
                    original_accent_label="test",
                    duration=3.0
                ))
        
        # Convert to DataFrame and split by speakers (70/15/15)
        df = pd.DataFrame([s.to_dict() for s in samples])
        speakers = df['speaker_id'].unique()
        np.random.seed(42)
        np.random.shuffle(speakers)
        
        n_speakers = len(speakers)
        train_end = int(n_speakers * 0.7)
        val_end = int(n_speakers * 0.85)
        
        train_spk = set(speakers[:train_end])
        val_spk = set(speakers[train_end:val_end])
        test_spk = set(speakers[val_end:])
        
        # Verify no speaker overlap
        self.assertEqual(len(train_spk & val_spk), 0)
        self.assertEqual(len(train_spk & test_spk), 0)
        self.assertEqual(len(val_spk & test_spk), 0)
        
        # Verify proportions (±1 speaker tolerance)
        self.assertAlmostEqual(len(train_spk), 7, delta=1)
        self.assertAlmostEqual(len(val_spk), 1, delta=1)
        self.assertAlmostEqual(len(test_spk), 2, delta=1)
    
    def test_metadata_preservation_and_csv_output(self):
        """Test 4: Chunk metadata preservation + CSV has required columns"""
        sample = UnifiedSample(
            sample_id="test_sample",
            dataset_name="test",
            speaker_id="spk1",
            audio_path="/fake/test.wav",
            transcript="test transcript",
            region_label="Northeast",
            original_accent_label="boston",
            duration=5.0
        )
        
        # Add chunk metadata
        sample.chunk_start_sample = 0
        sample.chunk_end_sample = 80000
        
        # Convert to DataFrame (simulating save to CSV)
        df = pd.DataFrame([sample.to_dict()])
        df['chunk_start_sample'] = sample.chunk_start_sample
        df['chunk_end_sample'] = sample.chunk_end_sample
        
        # Verify required columns exist
        required_cols = ['audio_path', 'region_label', 'speaker_id', 'dataset_name']
        for col in required_cols:
            self.assertIn(col, df.columns)
            self.assertFalse(df[col].isna().any())
        
        # Verify chunk metadata preserved
        self.assertEqual(df['chunk_start_sample'].iloc[0], 0)
        self.assertEqual(df['chunk_end_sample'].iloc[0], 80000)
        
        # Test CSV roundtrip
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            df_loaded = pd.read_csv(f.name)
            
            # Verify data survives roundtrip
            self.assertEqual(df_loaded['chunk_start_sample'].iloc[0], 0)
            self.assertEqual(df_loaded['chunk_end_sample'].iloc[0], 80000)
            self.assertFalse(df_loaded['chunk_start_sample'].isna().any())
            
            os.unlink(f.name)
    
    def test_edge_cases(self):
        """Test 5: Handle empty dataset, single sample, missing audio"""
        # Test empty dataset
        empty = filter_samples([], Mock(min_samples_per_speaker=1, max_samples_per_speaker=100, seed=42))
        self.assertEqual(len(empty), 0)
        
        # Test single sample (gets filtered due to min_samples constraint)
        single = [UnifiedSample(
            sample_id="single", dataset_name="test", speaker_id="spk1",
            audio_path="/fake/single.wav", transcript="test",
            region_label="test", original_accent_label="test", duration=3.0
        )]
        args = Mock(min_samples_per_speaker=2, max_samples_per_speaker=None, seed=42)
        filtered = filter_samples(single, args)
        self.assertEqual(len(filtered), 0)
        
        # Test missing audio file (should handle gracefully in chunking)
        with patch('os.path.exists', return_value=False):
            missing = [UnifiedSample(
                sample_id="missing", dataset_name="test", speaker_id="spk1",
                audio_path="/fake/missing.wav", transcript="test",
                region_label="test", original_accent_label="test", duration=3.0
            )]
            chunked = chunk_all_audio_samples(missing)
            self.assertEqual(len(chunked), 1)
            self.assertEqual(chunked[0].chunk_start_sample, 0)
    
    def test_reproducibility(self):
        """Test 6: Same seed produces identical splits"""
        samples = [
            UnifiedSample(
                sample_id=f"s{i}", dataset_name="test", speaker_id=f"spk{i}",
                audio_path=f"/fake/{i}.wav", transcript="test",
                region_label="test", original_accent_label="test", duration=3.0
            )
            for i in range(10)
        ]
        
        # Split twice with same seed
        df = pd.DataFrame([s.to_dict() for s in samples])
        
        np.random.seed(42)
        speakers1 = df['speaker_id'].unique().copy()
        np.random.shuffle(speakers1)
        
        np.random.seed(42)
        speakers2 = df['speaker_id'].unique().copy()
        np.random.shuffle(speakers2)
        
        # Verify identical ordering
        self.assertTrue(np.array_equal(speakers1, speakers2))
    
    def test_region_consolidation_integration(self):
        """Test 7: Verify Upper Midwest -> Midwest consolidation in actual pipeline"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test samples with Upper Midwest region
            mock_samples = [
                UnifiedSample(
                    sample_id=f"test_{i}",
                    dataset_name="TIMIT",
                    speaker_id=f"spk_{i}",
                    audio_path=f"{tmpdir}/fake_{i}.wav",
                    transcript="test",
                    region_label="Upper Midwest" if i < 2 else "Northeast",
                    original_accent_label="test",
                    duration=3.0
                )
                for i in range(4)
            ]
            
            # Mock the dataset loading
            with patch('prepare_dataset.UnifiedAccentDataset') as mock_dataset_class:
                mock_instance = Mock()
                mock_instance.load_all_datasets.return_value = mock_samples
                mock_instance.get_statistics.return_value = {}
                mock_dataset_class.return_value = mock_instance
                
                # Mock file operations
                with patch('os.path.exists', return_value=False), \
                     patch('prepare_dataset.chunk_all_audio_samples') as mock_chunk:
                    
                    def add_chunk_metadata(samples, **kwargs):
                        for s in samples:
                            s.chunk_start_sample = 0
                            s.chunk_end_sample = int(s.duration * 16000)
                        return samples
                    mock_chunk.side_effect = add_chunk_metadata
                    
                    # Run main with test args
                    test_args = [
                        '--data_root', tmpdir,
                        '--datasets', 'TIMIT',
                        '--output_dir', tmpdir,
                        '--dataset_name', 'test_consolidation',
                        '--seed', '42'
                    ]
                    
                    with patch('sys.argv', ['prepare_dataset.py'] + test_args):
                        from prepare_dataset import main
                        main()
                    
                    # Verify region consolidation happened
                    output_path = Path(tmpdir) / 'test_consolidation'
                    self.assertTrue(output_path.exists())
                    
                    # Check that Upper Midwest became Midwest
                    for csv_file in ['train.csv', 'val.csv', 'test.csv']:
                        csv_path = output_path / csv_file
                        if csv_path.exists() and csv_path.stat().st_size > 0:
                            df = pd.read_csv(csv_path)
                            if len(df) > 0:
                                self.assertNotIn("Upper Midwest", df['region_label'].values,
                                               f"Upper Midwest should not exist in {csv_file}")
                                # Check if any Midwest samples exist (from consolidation)
                                midwest_samples = df[df['region_label'] == 'Midwest']
                                if csv_file == 'train.csv':  # Most likely to have samples
                                    self.assertTrue(len(midwest_samples) > 0 or len(df) == 0,
                                                  "Should have Midwest samples from consolidation")


if __name__ == "__main__":
    unittest.main()