"""
rebuild_model.py — Rebuilds model architecture fresh inside Docker,
loads saved weights, and saves a clean .keras file.
Run automatically during Docker build.
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional

MAX_VOCAB_SIZE   = 10000
MAX_SEQ_LENGTH   = 100
EMBEDDING_DIM    = 100

print("Building model architecture...")
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
model.build(input_shape=(None, MAX_SEQ_LENGTH))

print("Loading weights...")
model.load_weights('model/model_weights.weights.h5')

print("Saving clean model...")
model.save('model/disaster_model.keras')
print("Done — model ready for app.py")
