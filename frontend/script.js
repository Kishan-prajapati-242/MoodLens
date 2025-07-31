// API Configuration
const API_URL = window.location.origin;

// DOM Elements
const textInput = document.getElementById('textInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const micBtn = document.getElementById('micBtn');
const resultsSection = document.getElementById('results');
const spinner = analyzeBtn.querySelector('.spinner');

// Speech Recognition Setup
let recognition = null;
if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'en-US';
    
    recognition.onstart = () => {
        micBtn.classList.add('recording');
        textInput.placeholder = 'Listening...';
    };
    
    recognition.onend = () => {
        micBtn.classList.remove('recording');
        textInput.placeholder = 'Type or speak about what happened...';
    };
    
    recognition.onresult = (event) => {
        const transcript = Array.from(event.results)
            .map(result => result[0].transcript)
            .join('');
        textInput.value = transcript;
    };
    
    recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        alert('Speech recognition error: ' + event.error);
    };
}

// Event Listeners
analyzeBtn.addEventListener('click', analyzeText);
micBtn.addEventListener('click', toggleSpeechRecognition);
textInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        analyzeText();
    }
});

// Functions
async function analyzeText() {
    const text = textInput.value.trim();
    if (!text) {
        alert('Please enter some text to analyze');
        return;
    }
    
    // Show loading state
    analyzeBtn.disabled = true;
    analyzeBtn.querySelector('span').textContent = 'Analyzing...';
    spinner.classList.remove('hidden');
    resultsSection.classList.add('hidden');
    
    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });
        
        if (!response.ok) {
            throw new Error('Failed to analyze text');
        }
        
        const data = await response.json();
        displayResults(data);
        
    } catch (error) {
        console.error('Error:', error);
        alert('Error analyzing text. Make sure the backend is running!');
    } finally {
        // Reset button state
        analyzeBtn.disabled = false;
        analyzeBtn.querySelector('span').textContent = 'Analyze';
        spinner.classList.add('hidden');
    }
}

function displayResults(data) {
    // Show results section
    resultsSection.classList.remove('hidden');
    
    // Primary Event
    document.getElementById('primaryEventName').textContent = data.primary_event.event;
    const primaryConfidence = Math.round(data.primary_event.confidence * 100);
    document.getElementById('primaryConfidence').textContent = `${primaryConfidence}%`;
    document.getElementById('primaryConfidenceBar').style.width = `${primaryConfidence}%`;
    document.getElementById('interpretation').textContent = data.interpretation;
    
    // Event Group
    document.getElementById('eventGroup').textContent = data.event_group.prediction;
    document.getElementById('eventGroupConf').textContent = 
        `${Math.round(data.event_group.confidence * 100)}% confident`;
    
    // Emotion
    document.getElementById('emotion').textContent = data.emotion.prediction;
    document.getElementById('emotionConf').textContent = 
        `${Math.round(data.emotion.confidence * 100)}% confident`;
    updateEmotionIcon(data.emotion.prediction);
    
    // Tense
    document.getElementById('tense').textContent = data.tense.prediction;
    document.getElementById('tenseConf').textContent = 
        `${Math.round(data.tense.confidence * 100)}% confident`;
    
    // Sarcasm
    const isSarcastic = data.sarcasm.prediction === 'TRUE';
    document.getElementById('sarcasm').textContent = isSarcastic ? 'Detected' : 'Not detected';
    document.getElementById('sarcasmConf').textContent = 
        `${Math.round(data.sarcasm.confidence * 100)}% confident`;
    
    // Sentiment
    const sentimentPercent = Math.round(data.sentiment * 100);
    document.getElementById('sentimentBar').style.width = `${sentimentPercent}%`;
    document.getElementById('sentimentValue').textContent = `${sentimentPercent}%`;
    
    // Alternative Events
    if (data.multiple_events && data.top_events.length > 1) {
        const altEventsSection = document.getElementById('alternativeEvents');
        const altEventsList = document.getElementById('alternativeEventsList');
        
        altEventsSection.classList.remove('hidden');
        altEventsList.innerHTML = '';
        
        // Skip the first event (already shown as primary)
        data.top_events.slice(1).forEach(event => {
            const eventDiv = document.createElement('div');
            eventDiv.className = 'alternative-event';
            eventDiv.innerHTML = `
                <span>${event.event}</span>
                <span>${Math.round(event.confidence * 100)}%</span>
            `;
            altEventsList.appendChild(eventDiv);
        });
    } else {
        document.getElementById('alternativeEvents').classList.add('hidden');
    }
    
    // Smooth scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function updateEmotionIcon(emotion) {
    const emotionIcons = {
        'joy': '😊',
        'sadness': '😢',
        'anger': '😠',
        'fear': '😨',
        'surprise': '😲',
        'disgust': '🤢',
        'neutral': '😐',
        'anxiety': '😰',
        'hope': '🤗',
        'pride': '😌',
        'disappointment': '😞',
        'relief': '😌'
    };
    
    const icon = emotionIcons[emotion.toLowerCase()] || '💭';
    document.getElementById('emotionIcon').textContent = icon;
}

function toggleSpeechRecognition() {
    if (!recognition) {
        alert('Speech recognition is not supported in your browser.');
        return;
    }
    
    if (micBtn.classList.contains('recording')) {
        recognition.stop();
    } else {
        recognition.start();
    }
}

// Example texts for quick testing
const exampleTexts = [
    "I just got promoted to senior manager!",
    "They laid me off after 10 years with the company",
    "We're getting married next month!",
    "Oh great, another pay cut. Just what I needed.",
    "Starting my freelance journey next month!"
];

// Add example on double-click of placeholder
textInput.addEventListener('dblclick', () => {
    if (!textInput.value) {
        textInput.value = exampleTexts[Math.floor(Math.random() * exampleTexts.length)];
    }
});