#!/usr/bin/env python3
"""
Regional Accent Classifier Web App - BASELINE VERSION
This version uses an untrained Wav2Vec2 model to show baseline performance
"""

import os
import tempfile
import uuid
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import librosa
import soundfile as sf
from pydub import AudioSegment
import numpy as np
import io

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Model configuration
REGIONS = [
    "New England",
    "New York Metropolitan", 
    "Mid-Atlantic",
    "South Atlantic",
    "Deep South",
    "Upper Midwest",
    "Lower Midwest",
    "West"
]

# Global model variables
model = None
processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_baseline_model():
    """Load untrained Wav2Vec2 model with random classification head"""
    global model, processor
    if model is None:
        print("Loading BASELINE Wav2Vec2 model (not fine-tuned)...")
        print("WARNING: This model has a randomly initialized classification head!")
        print("Results will be essentially random and serve as a baseline.")
        
        # Load the base Wav2Vec2 model and processor
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        
        # Create model with classification head for 6 regions
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            "facebook/wav2vec2-base",
            num_labels=6
        )
        
        # The classification head is randomly initialized by default
        model.eval()
        model.to(device)
        
        print("Baseline model loaded successfully")
        print("Classification head parameters are RANDOM - expect poor performance!")


def convert_to_wav(audio_data, format_hint=None):
    """Convert various audio formats to WAV"""
    # Create a temporary file for the input audio
    with tempfile.NamedTemporaryFile(suffix=f'.{format_hint or "webm"}', delete=False) as tmp_input:
        tmp_input.write(audio_data)
        tmp_input_path = tmp_input.name
    
    # Create a temporary file for the output WAV
    tmp_output_path = tempfile.mktemp(suffix='.wav')
    
    try:
        # Try to load with pydub (handles WebM, MP3, etc.)
        audio = AudioSegment.from_file(tmp_input_path)
        # Convert to mono and 16kHz
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        # Export as WAV
        audio.export(tmp_output_path, format="wav")
        
        # Load the WAV file
        audio_array, sr = librosa.load(tmp_output_path, sr=16000)
        
    except Exception as e:
        print(f"Error with pydub conversion: {e}")
        # Fallback: try librosa directly
        audio_array, sr = librosa.load(tmp_input_path, sr=16000)
    
    finally:
        # Clean up temporary files
        if os.path.exists(tmp_input_path):
            os.unlink(tmp_input_path)
        if os.path.exists(tmp_output_path):
            os.unlink(tmp_output_path)
    
    return audio_array


def classify_audio(audio_array):
    """Classify audio using the baseline model"""
    load_baseline_model()
    
    # Process audio
    inputs = processor(
        audio_array, 
        sampling_rate=16000, 
        return_tensors="pt",
        padding=True
    )
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        pred_idx = torch.argmax(logits, dim=-1).item()
        confidence = probs[0, pred_idx].item()
    
    # Get top 3 predictions
    top3_probs, top3_indices = torch.topk(probs[0], 3)
    top3_results = [
        {
            "region": REGIONS[idx.item()],
            "probability": prob.item()
        }
        for prob, idx in zip(top3_probs, top3_indices)
    ]
    
    return {
        "predicted_region": REGIONS[pred_idx],
        "confidence": confidence,
        "top3_predictions": top3_results,
        "all_probabilities": {
            REGIONS[i]: probs[0, i].item() for i in range(len(REGIONS))
        },
        "warning": "This is a BASELINE model with random weights. Results are not meaningful!"
    }


@app.route('/')
def index():
    """Main page with baseline warning"""
    return render_template('index_baseline.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze uploaded audio with baseline model"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400
        
        # Read audio data
        audio_data = audio_file.read()
        
        # Get format hint from filename or content type
        format_hint = audio_file.filename.split('.')[-1].lower()
        if format_hint not in ['webm', 'mp3', 'wav', 'ogg', 'm4a']:
            # Try to guess from content type
            content_type = audio_file.content_type
            if 'webm' in content_type:
                format_hint = 'webm'
            elif 'mp' in content_type:
                format_hint = 'mp3'
            else:
                format_hint = 'webm'  # Default for browser recordings
        
        # Convert to WAV and classify
        audio_array = convert_to_wav(audio_data, format_hint)
        
        # Ensure minimum length (0.5 seconds)
        if len(audio_array) < 8000:  # 0.5 seconds at 16kHz
            return jsonify({'error': 'Audio too short. Please record at least 1 second.'}), 400
        
        # Classify audio
        results = classify_audio(audio_array)
        
        # Generate shareable ID
        share_id = str(uuid.uuid4())[:8]
        results['share_id'] = share_id
        
        return jsonify(results)
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'mode': 'baseline'})


if __name__ == '__main__':
    # Ensure model is loaded before starting
    print("\n" + "="*60)
    print("STARTING BASELINE VERSION - NO FINE-TUNING")
    print("="*60)
    print("This version uses an untrained classification head.")
    print("Expect random/poor performance!")
    print("="*60 + "\n")
    
    print("Preloading baseline model...")
    load_baseline_model()
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5001)