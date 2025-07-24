let mediaRecorder;
let audioChunks = [];
let audioBlob;

const recordButton = document.getElementById('recordButton');
const audioPlayback = document.getElementById('audioPlayback');
const analyzeButton = document.getElementById('analyzeButton');
const resultsSection = document.getElementById('results');
const loadingSection = document.getElementById('loading');
const tryAgainButton = document.getElementById('tryAgainButton');
const shareButton = document.getElementById('shareButton');

// Region colors for visualization
const regionColors = {
    'New England': '#FF6B6B',
    'New York Metropolitan': '#4ECDC4',
    'Mid-Atlantic': '#45B7D1',
    'South Atlantic': '#96CEB4',
    'Deep South': '#F7DC6F',
    'Upper Midwest': '#BB8FCE',
    'Lower Midwest': '#F8B739',
    'West': '#52C3F1'
};

recordButton.addEventListener('click', toggleRecording);
analyzeButton.addEventListener('click', analyzeAudio);
tryAgainButton.addEventListener('click', resetApp);
shareButton.addEventListener('click', shareResults);

async function toggleRecording() {
    if (!mediaRecorder || mediaRecorder.state === 'inactive') {
        startRecording();
    } else {
        stopRecording();
    }
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        // Determine the best format for the browser
        const options = { mimeType: 'audio/webm' };
        if (!MediaRecorder.isTypeSupported('audio/webm')) {
            options.mimeType = 'audio/mp4';
        }
        
        mediaRecorder = new MediaRecorder(stream, options);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = () => {
            audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType });
            const audioUrl = URL.createObjectURL(audioBlob);
            audioPlayback.src = audioUrl;
            
            // Show audio preview
            document.querySelector('.audio-preview').style.display = 'block';
            document.querySelector('.recording-indicator').style.display = 'none';
        };

        mediaRecorder.start();
        
        // Update UI
        recordButton.classList.add('recording');
        recordButton.querySelector('.btn-text').textContent = 'Stop Recording';
        document.querySelector('.recording-indicator').style.display = 'flex';
        
    } catch (error) {
        console.error('Error accessing microphone:', error);
        alert('Please allow microphone access to use this feature.');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
        
        // Update UI
        recordButton.classList.remove('recording');
        recordButton.querySelector('.btn-text').textContent = 'Start Recording';
    }
}

async function analyzeAudio() {
    if (!audioBlob) {
        alert('Please record audio first.');
        return;
    }

    // Show loading
    loadingSection.style.display = 'block';
    document.querySelector('.record-section').style.display = 'none';

    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.webm');

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const results = await response.json();
        
        if (results.error) {
            throw new Error(results.error);
        }

        displayResults(results);
        
    } catch (error) {
        console.error('Error analyzing audio:', error);
        alert('Error analyzing audio: ' + error.message);
        resetApp();
    }
}

function displayResults(results) {
    // Hide loading
    loadingSection.style.display = 'none';
    
    // Show results
    resultsSection.style.display = 'block';
    
    // Display main result
    document.querySelector('.region-name').textContent = results.predicted_region;
    document.querySelector('.confidence').textContent = `${(results.confidence * 100).toFixed(1)}% confidence`;
    
    // Create probability chart
    const chartContainer = document.querySelector('.chart-container');
    chartContainer.innerHTML = '';
    
    // Sort regions by probability
    const sortedRegions = Object.entries(results.all_probabilities)
        .sort(([,a], [,b]) => b - a);
    
    sortedRegions.forEach(([region, probability]) => {
        const probBar = document.createElement('div');
        probBar.className = 'prob-bar';
        
        const label = document.createElement('div');
        label.className = 'prob-label';
        label.innerHTML = `
            <span>${region}</span>
            <span>${(probability * 100).toFixed(1)}%</span>
        `;
        
        const track = document.createElement('div');
        track.className = 'prob-track';
        
        const fill = document.createElement('div');
        fill.className = 'prob-fill';
        fill.style.width = '0%';
        fill.style.backgroundColor = regionColors[region] || '#5B5FCF';
        
        track.appendChild(fill);
        probBar.appendChild(label);
        probBar.appendChild(track);
        chartContainer.appendChild(probBar);
        
        // Animate the bar
        setTimeout(() => {
            fill.style.width = `${probability * 100}%`;
        }, 100);
    });
    
    // Store results for sharing
    window.currentResults = results;
}

function resetApp() {
    // Reset UI
    resultsSection.style.display = 'none';
    document.querySelector('.record-section').style.display = 'block';
    document.querySelector('.audio-preview').style.display = 'none';
    
    // Clear audio
    audioChunks = [];
    audioBlob = null;
    audioPlayback.src = '';
    
    // Reset button
    recordButton.classList.remove('recording');
    recordButton.querySelector('.btn-text').textContent = 'Start Recording';
}

function shareResults() {
    if (!window.currentResults) return;
    
    const shareText = `🗣️ I took the Regional Accent Detector test!\n\nMy accent is: ${window.currentResults.predicted_region} (${(window.currentResults.confidence * 100).toFixed(1)}% match)\n\nTry it yourself:`;
    const shareUrl = window.location.href;
    
    if (navigator.share) {
        // Use native share API on mobile
        navigator.share({
            title: 'Regional Accent Detector',
            text: shareText,
            url: shareUrl
        }).catch(err => console.log('Error sharing:', err));
    } else {
        // Fallback: copy to clipboard
        const fullText = `${shareText} ${shareUrl}`;
        navigator.clipboard.writeText(fullText).then(() => {
            alert('Results copied to clipboard!');
        }).catch(err => {
            console.error('Error copying to clipboard:', err);
        });
    }
}

// Check for browser support
if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    alert('Your browser does not support audio recording. Please use a modern browser like Chrome, Firefox, or Safari.');
}