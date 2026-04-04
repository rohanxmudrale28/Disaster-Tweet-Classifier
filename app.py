import os
import re
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

MAX_SEQUENCE_LENGTH = 100

# ── HTML for a simple browser UI ─────────────────────────────────────────────
HOME_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Disaster Tweet Classifier</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 700px; margin: 60px auto; padding: 20px; }
    h1   { color: #c0392b; }
    textarea { width: 100%; height: 100px; font-size: 15px; padding: 10px; }
    button   { margin-top: 10px; padding: 12px 30px; background: #c0392b;
               color: white; border: none; border-radius: 5px; font-size: 15px; cursor: pointer; }
    #result  { margin-top: 20px; padding: 15px; border-radius: 8px; font-size: 16px; }
    .disaster    { background: #fde8e8; color: #c0392b; }
    .not-disaster{ background: #e8fde8; color: #27ae60; }
  </style>
</head>
<body>
  <h1>🚨 Disaster Tweet Classifier</h1>
  <p>Enter a tweet below to classify it as a disaster or not.</p>
  <textarea id="tweet" placeholder="e.g. There is a wildfire near my house!"></textarea><br>
  <button onclick="classify()">Classify</button>
  <div id="result"></div>

  <script>
    async function classify() {
      const tweet = document.getElementById('tweet').value;
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tweet })
      });
      const data = await res.json();
      const div = document.getElementById('result');
      if (data.label === 1) {
        div.className = 'disaster';
        div.innerHTML = '🔴 <b>DISASTER</b> — Confidence: ' + (data.confidence * 100).toFixed(1) + '%';
      } else {
        div.className = 'not-disaster';
        div.innerHTML = '🟢 <b>NOT DISASTER</b> — Confidence: ' + ((1 - data.confidence) * 100).toFixed(1) + '%';
      }
    }
  </script>
</body>
</html>
"""

# ── Text cleaning (same as training) ─────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+', '', text)   # remove URLs
    text = re.sub(r'@\w+', '', text)            # remove mentions
    text = re.sub(r'[^a-z\s]', '', text)        # remove special chars
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── Load model and tokenizer at startup ──────────────────────────────────────
MODEL_PATH     = os.path.join('model', 'disaster_model.keras')
TOKENIZER_PATH = os.path.join('model', 'tokenizer.pkl')

model     = None
tokenizer = None

def load_artifacts():
    global model, tokenizer
    if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
        print("✅ Model and tokenizer loaded successfully.")
    else:
        print("⚠️  Model files not found. Run model/train.py first.")

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template_string(HOME_HTML)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    # ✅ Step 1: Validate request JSON
    if not request.is_json:
        return jsonify({'error': 'Invalid request'}), 400

    data = request.get_json()

    # ✅ Step 2: Check tweet key exists
    if 'tweet' not in data:
        return jsonify({'error': 'No tweet provided'}), 400

    tweet = data.get('tweet', '').strip()

    # ✅ Step 3: Check empty tweet (THIS FIXES YOUR TEST)
    if not tweet:
        return jsonify({'error': 'No tweet provided'}), 400

    # ✅ Step 4: Check model AFTER validation
    if model is None or tokenizer is None:
        return jsonify({'error': 'Model not loaded. Run model/train.py first.'}), 503

    # ✅ Step 5: Prediction logic (unchanged)
    cleaned = clean_text(tweet)
    seq     = tokenizer.texts_to_sequences([cleaned])
    padded  = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    prob    = float(model.predict(padded, verbose=0)[0][0])
    label   = int(prob > 0.5)

    return jsonify({
        'tweet':      tweet,
        'label':      label,
        'confidence': round(prob, 4),
        'result':     'DISASTER' if label == 1 else 'NOT DISASTER'
    })
# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    load_artifacts()
    
    app.run(host='0.0.0.0', port=8080, debug=False)
