"""
train.py  —  Train the Disaster Tweet Classifier and save model artifacts.
Run once before starting the Flask app:
    python model/train.py
"""

import os
import re
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from sklearn.model_selection import train_test_split

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_PATH        = os.path.join('data', 'train.csv')
MODEL_DIR         = 'model'
MODEL_PATH        = os.path.join(MODEL_DIR, 'disaster_model.h5')
TOKENIZER_PATH    = os.path.join(MODEL_DIR, 'tokenizer.pkl')

MAX_VOCAB_SIZE    = 10000
MAX_SEQ_LENGTH    = 100
EMBEDDING_DIM     = 100
EPOCHS            = 5
BATCH_SIZE        = 64
TEST_SIZE         = 0.2
RANDOM_STATE      = 42

# ── Text cleaning ─────────────────────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── Load data ─────────────────────────────────────────────────────────────────
print("📂 Loading data...")
train_df = pd.read_csv(TRAIN_PATH)
train_df['clean_text'] = train_df['text'].map(clean_text)

# ── Tokenize ──────────────────────────────────────────────────────────────────
print("🔤 Tokenizing...")
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(train_df['clean_text'])

sequences = tokenizer.texts_to_sequences(train_df['clean_text'])
padded    = pad_sequences(sequences, maxlen=MAX_SEQ_LENGTH, padding='post')
y         = train_df['target'].values

# ── Train / val split ─────────────────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    padded, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# ── Build model ───────────────────────────────────────────────────────────────
print("🏗️  Building model...")
model = Sequential([
    Embedding(input_dim=MAX_VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_SEQ_LENGTH),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(32)),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# ── Train ─────────────────────────────────────────────────────────────────────
print("🚀 Training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

val_acc = max(history.history['val_accuracy'])
print(f"\n✅ Best validation accuracy: {val_acc:.4f}")

# ── Save artifacts ────────────────────────────────────────────────────────────
os.makedirs(MODEL_DIR, exist_ok=True)

model.save(MODEL_PATH)
print(f"💾 Model saved → {MODEL_PATH}")

with open(TOKENIZER_PATH, 'wb') as f:
    pickle.dump(tokenizer, f)
print(f"💾 Tokenizer saved → {TOKENIZER_PATH}")

print("\n🎉 Training complete! You can now start app.py")
