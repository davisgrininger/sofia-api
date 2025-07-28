# Updated Sofia.ai Flask App with GoEmotions Integration
# This replaces your existing app.py

from flask import Flask, request, jsonify, render_template_string
import tensorflow as tf
import pickle
import numpy as np
import json
import os

app = Flask(__name__)

# ========================
# Load GoEmotions Model
# ========================

print("Loading Sofia.ai with GoEmotions...")

try:
    # Load the new GoEmotions model
    goemotions_model = tf.keras.models.load_model('sofia_goemotions/emotion_model.h5')
    
    # Load the custom tokenizer
    with open('sofia_goemotions/tokenizer.pkl', 'rb') as f:
        goemotions_tokenizer = pickle.load(f)
    
    # Load configuration
    with open('sofia_goemotions/config.json', 'r') as f:
        goemotions_config = json.load(f)
    
    print("✅ GoEmotions model loaded successfully!")
    print(f"   Emotions: {len(goemotions_config['emotion_names'])}")
    
    # Try to load your existing sentiment model (optional)
    try:
        sentiment_model = tf.keras.models.load_model('best_sentiment_model.h5')
        with open('sofia_tokenizer.pkl', 'rb') as f:
            sentiment_tokenizer = pickle.load(f)
        with open('sofia_model_config.json', 'r') as f:
            sentiment_config = json.load(f)
        print("✅ Original sentiment model also loaded!")
        HAVE_SENTIMENT = True
    except:
        print("⚠️  Original sentiment model not found - using emotion-based sentiment")
        HAVE_SENTIMENT = False
    
except Exception as e:
    print(f"❌ Error loading GoEmotions model: {e}")
    print("Make sure the model files are in the correct location!")
    exit(1)

# ========================
# Prediction Functions
# ========================

def predict_emotions_goemotions(text, threshold=0.3):
    """Predict emotions using GoEmotions model"""
    try:
        # Tokenize text using the custom tokenizer
        tokenized = goemotions_tokenizer.texts_to_sequences([text])
        
        # Get predictions
        predictions = goemotions_model.predict(tokenized, verbose=0)[0]
        
        # Get emotion names
        emotion_names = goemotions_config['emotion_names']
        
        # Find top emotions above threshold
        top_emotions = []
        for i, score in enumerate(predictions):
            if score > threshold:
                top_emotions.append({
                    'label': emotion_names[i],
                    'confidence': float(score)
                })
        
        # Sort by confidence
        top_emotions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # If no emotions above threshold, get top 3
        if not top_emotions:
            top_indices = np.argsort(predictions)[-3:][::-1]
            top_emotions = [
                {
                    'label': emotion_names[idx],
                    'confidence': float(predictions[idx])
                }
                for idx in top_indices
            ]
        
        # Primary emotion
        primary_emotion = top_emotions[0] if top_emotions else {
            'label': 'neutral',
            'confidence': 0.5
        }
        
        return {
            'primary': primary_emotion,
            'all_detected': top_emotions[:5],  # Top 5 max
            'total_emotions': len([e for e in top_emotions if e['confidence'] > threshold])
        }
        
    except Exception as e:
        print(f"Error in emotion prediction: {e}")
        return {
            'primary': {'label': 'neutral', 'confidence': 0.5},
            'all_detected': [{'label': 'neutral', 'confidence': 0.5}],
            'total_emotions': 1
        }

def emotion_to_sentiment(emotion_results):
    """Convert emotion predictions to sentiment using emotion mapping"""
    
    # Map emotions to sentiment categories
    positive_emotions = {
        'admiration', 'amusement', 'approval', 'caring', 'excitement', 
        'gratitude', 'joy', 'love', 'optimism', 'pride', 'relief'
    }
    
    negative_emotions = {
        'anger', 'annoyance', 'disappointment', 'disapproval', 'disgust', 
        'embarrassment', 'fear', 'grief', 'nervousness', 'remorse', 'sadness'
    }
    
    neutral_emotions = {
        'confusion', 'curiosity', 'desire', 'realization', 'surprise', 'neutral'
    }
    
    # Calculate sentiment scores based on detected emotions
    positive_score = 0.0
    negative_score = 0.0
    neutral_score = 0.0
    
    for emotion in emotion_results['all_detected']:
        label = emotion['label']
        confidence = emotion['confidence']
        
        if label in positive_emotions:
            positive_score += confidence
        elif label in negative_emotions:
            negative_score += confidence
        elif label in neutral_emotions:
            neutral_score += confidence
    
    # Normalize scores
    total_score = positive_score + negative_score + neutral_score
    if total_score > 0:
        positive_score /= total_score
        negative_score /= total_score
        neutral_score /= total_score
    else:
        # Fallback
        neutral_score = 1.0
    
    # Determine primary sentiment
    sentiment_scores = {
        'positive': positive_score,
        'negative': negative_score,
        'neutral': neutral_score
    }
    
    primary_sentiment = max(sentiment_scores.keys(), key=lambda k: sentiment_scores[k])
    
    return {
        'label': primary_sentiment,
        'confidence': sentiment_scores[primary_sentiment],
        'probabilities': sentiment_scores
    }

def predict_sentiment_legacy(text):
    """Use original sentiment model if available"""
    if not HAVE_SENTIMENT:
        return None
    
    try:
        # Use your original preprocessing
        clean_text = text.lower().strip()
        sequence = sentiment_tokenizer.texts_to_sequences([clean_text])
        
        # Pad sequence
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        padded = pad_sequences(sequence, maxlen=sentiment_config['max_sequence_length'], padding='post')
        
        # Predict
        prediction = sentiment_model.predict(padded, verbose=0)
        
        return {
            'label': sentiment_config['sentiment_classes'][np.argmax(prediction)],
            'confidence': float(np.max(prediction)),
            'probabilities': {
                cls: float(prob) for cls, prob in zip(sentiment_config['sentiment_classes'], prediction[0])
            }
        }
    except Exception as e:
        print(f"Error in legacy sentiment prediction: {e}")
        return None

# ========================
# Flask Routes
# ========================

@app.route('/')
def home():
    """Updated test page with GoEmotions"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sofia.ai - Enhanced with GoEmotions</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f0f8ff; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }
            h1 { color: #4a90e2; text-align: center; }
            .upgrade-notice { background: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #4caf50; }
            .test-box { background: #f9f9f9; padding: 20px; border-radius: 5px; margin: 20px 0; }
            button { background: #4a90e2; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background: #357abd; }
            #result { margin-top: 20px; padding: 15px; background: #e8f5e8; border-radius: 5px; }
            .emotion-chip { display: inline-block; background: #e3f2fd; padding: 4px 8px; margin: 2px; border-radius: 15px; font-size: 12px; }
            .primary-emotion { background: #4caf50; color: white; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎭 Sofia.ai - Now with GoEmotions!</h1>
            
            <div class="upgrade-notice">
                <h3>🚀 Major Upgrade Complete!</h3>
                <p><strong>New features:</strong></p>
                <ul>
                    <li>27 fine-grained emotions (vs basic categories)</li>
                    <li>Multi-label emotion detection</li>
                    <li>Trained on 58K human-annotated samples</li>
                    <li>Better real-world performance</li>
                </ul>
            </div>
            
            <div class="test-box">
                <h3>Test Your Enhanced API:</h3>
                <input type="text" id="testText" placeholder="Type a message to analyze..." style="width: 70%; padding: 8px;">
                <button onclick="testAPI()">Analyze</button>
                <div id="result"></div>
            </div>
            
            <div class="test-box">
                <h3>Sample Texts to Try:</h3>
                <p><button onclick="testSample('I\\'m so excited about this new opportunity!')">Excitement</button></p>
                <p><button onclick="testSample('Thank you so much for your help!')">Gratitude</button></p>
                <p><button onclick="testSample('This makes me really angry and frustrated.')">Anger</button></p>
                <p><button onclick="testSample('I feel confused and uncertain about what happened.')">Confusion</button></p>
                <p><button onclick="testSample('I love this so much, it brings me joy!')">Multiple emotions</button></p>
            </div>
            
            <div class="test-box">
                <h3>API Endpoint:</h3>
                <p><strong>POST</strong> /analyze</p>
                <p><strong>Body:</strong> {"text": "your message here"}</p>
                <p><strong>Response:</strong> Enhanced with 27 emotions + sentiment</p>
            </div>
        </div>
        
        <script>
        async function testAPI() {
            const text = document.getElementById('testText').value;
            if (!text) {
                alert('Please enter some text to analyze');
                return;
            }
            await analyzeText(text);
        }
        
        async function testSample(text) {
            document.getElementById('testText').value = text;
            await analyzeText(text);
        }
        
        async function analyzeText(text) {
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text: text})
                });
                
                const result = await response.json();
                
                let emotionChips = '';
                if (result.emotions && result.emotions.all_detected) {
                    emotionChips = result.emotions.all_detected.map((emotion, index) => {
                        const chipClass = index === 0 ? 'emotion-chip primary-emotion' : 'emotion-chip';
                        return `<span class="${chipClass}">${emotion.label}: ${(emotion.confidence * 100).toFixed(1)}%</span>`;
                    }).join(' ');
                }
                
                document.getElementById('result').innerHTML = `
                    <h4>🎭 Analysis Results:</h4>
                    <p><strong>Text:</strong> "${text}"</p>
                    
                    <h5>Emotions Detected:</h5>
                    <p><strong>Primary:</strong> ${result.emotions.primary.label} (${(result.emotions.primary.confidence * 100).toFixed(1)}%)</p>
                    <p><strong>All Detected:</strong><br>${emotionChips}</p>
                    
                    <h5>Sentiment:</h5>
                    <p><strong>Overall:</strong> ${result.sentiment.label} (${(result.sentiment.confidence * 100).toFixed(1)}%)</p>
                    
                    <p><em>Total emotions detected: ${result.emotions.total_emotions}</em></p>
                `;
            } catch (error) {
                document.getElementById('result').innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
            }
        }
        </script>
    </body>
    </html>
    """
    return html

@app.route('/analyze', methods=['POST'])
def analyze_text():
    """Enhanced analysis endpoint with GoEmotions"""
    try:
        # Get text from request
        data = request.json
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text field'}), 400
        
        text = data['text']
        if not text.strip():
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Get emotion predictions using GoEmotions
        emotion_results = predict_emotions_goemotions(text, threshold=0.3)
        
        # Get sentiment prediction
        if HAVE_SENTIMENT:
            # Try legacy sentiment model first
            sentiment_results = predict_sentiment_legacy(text)
            if sentiment_results is None:
                # Fallback to emotion-based sentiment
                sentiment_results = emotion_to_sentiment(emotion_results)
        else:
            # Use emotion-based sentiment
            sentiment_results = emotion_to_sentiment(emotion_results)
        
        # Format response
        response = {
            'text': text,
            'emotions': {
                'primary': emotion_results['primary'],
                'all_detected': emotion_results['all_detected'],
                'total_emotions': emotion_results['total_emotions']
            },
            'sentiment': sentiment_results,
            'model_info': {
                'emotion_model': 'GoEmotions (58k samples, 27 emotions)',
                'sentiment_model': 'Legacy Sofia.ai' if HAVE_SENTIMENT else 'Emotion-derived',
                'version': '2.0'
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'service': 'Sofia.ai v2.0 with GoEmotions',
        'emotions_available': len(goemotions_config['emotion_names']),
        'sentiment_model': 'legacy' if HAVE_SENTIMENT else 'emotion-derived'
    })

@app.route('/emotions')
def list_emotions():
    """List all available emotions"""
    return jsonify({
        'emotions': goemotions_config['emotion_names'],
        'total': len(goemotions_config['emotion_names']),
        'model': 'GoEmotions'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n🎭 Sofia.ai v2.0 is running on port {port}")
    print("✅ Enhanced with GoEmotions - 27 fine-grained emotions")
    print("🔗 Visit http://localhost:5000 to test the API")
    app.run(host='0.0.0.0', port=port, debug=False)
