from flask import Flask, request, jsonify, render_template_string
import tensorflow as tf
import pickle
import numpy as np
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

app = Flask(__name__)

# Load models and config once at startup
print("Loading Sofia.ai models...")

try:
    sentiment_model = tf.keras.models.load_model('best_sentiment_model.h5')
    emotion_model = tf.keras.models.load_model('best_emotion_model.h5')
    
    with open('sofia_tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    with open('sofia_model_config.json', 'r') as f:
        config = json.load(f)
    
    print("✅ Sofia.ai models loaded successfully!")
    
except Exception as e:
    print(f"❌ Error loading models: {e}")

@app.route('/')
def home():
    """Simple test page"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sofia.ai API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f0f8ff; }
            .container { max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }
            h1 { color: #4a90e2; text-align: center; }
            .test-box { background: #f9f9f9; padding: 20px; border-radius: 5px; margin: 20px 0; }
            button { background: #4a90e2; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background: #357abd; }
            #result { margin-top: 20px; padding: 10px; background: #e8f5e8; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 Sofia.ai API is Running!</h1>
            <p>Your sentiment and emotion analysis API is ready to use.</p>
            
            <div class="test-box">
                <h3>Test Your API:</h3>
                <input type="text" id="testText" placeholder="Type a message to analyze..." style="width: 70%; padding: 8px;">
                <button onclick="testAPI()">Analyze</button>
                <div id="result"></div>
            </div>
            
            <div class="test-box">
                <h3>API Endpoint:</h3>
                <p><strong>POST</strong> /analyze</p>
                <p><strong>Body:</strong> {"text": "your message here"}</p>
            </div>
        </div>
        
        <script>
        async function testAPI() {
            const text = document.getElementById('testText').value;
            if (!text) {
                alert('Please enter some text to analyze');
                return;
            }
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text: text})
                });
                
                const result = await response.json();
                
                document.getElementById('result').innerHTML = `
                    <h4>Analysis Results:</h4>
                    <p><strong>Sentiment:</strong> ${result.sentiment.label} (${(result.sentiment.confidence * 100).toFixed(1)}%)</p>
                    <p><strong>Emotion:</strong> ${result.emotion.label} (${(result.emotion.confidence * 100).toFixed(1)}%)</p>
                    <p><strong>Top 3 Emotions:</strong></p>
                    <ul>
                        ${result.emotion.top_3.map(e => `<li>${e.label}: ${(e.confidence * 100).toFixed(1)}%</li>`).join('')}
                    </ul>
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
    """Main API endpoint for text analysis"""
    try:
        # Get text from request
        data = request.json
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text field'}), 400
        
        text = data['text']
        if not text.strip():
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Preprocess text
        clean_text = text.lower().strip()
        sequence = tokenizer.texts_to_sequences([clean_text])
        padded = pad_sequences(sequence, maxlen=config['max_sequence_length'], padding='post')
        
        # Get predictions
        sent_pred = sentiment_model.predict(padded, verbose=0)
        emo_pred = emotion_model.predict(padded, verbose=0)
        
        # Format response
        response = {
            'text': text,
            'sentiment': {
                'label': config['sentiment_classes'][np.argmax(sent_pred)],
                'confidence': float(np.max(sent_pred)),
                'probabilities': {
                    cls: float(prob) for cls, prob in zip(config['sentiment_classes'], sent_pred[0])
                }
            },
            'emotion': {
                'label': config['emotion_classes'][np.argmax(emo_pred)],
                'confidence': float(np.max(emo_pred)),
                'top_3': [
                    {
                        'label': config['emotion_classes'][idx],
                        'confidence': float(emo_pred[0][idx])
                    }
                    for idx in np.argsort(emo_pred[0])[-3:][::-1]
                ]
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'Sofia.ai API'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
