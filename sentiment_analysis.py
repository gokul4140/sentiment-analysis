import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import streamlit as st
import os

# Load Model Weights
max_words = 20000
max_len = 200

# Build Model
def build_cnn_model():
    inputs = Input(shape=(max_len,))
    embedding = Embedding(input_dim=max_words, output_dim=128, input_length=max_len)(inputs)
    conv = Conv1D(filters=128, kernel_size=5, activation='relu')(embedding)
    pool = GlobalMaxPooling1D()(conv)
    dropout = Dropout(0.5)(pool)
    outputs = Dense(1, activation='sigmoid')(dropout)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_cnn_model()

# Load Trained Weights
weights_file = "cnn_model_weights.weights.h5"
if os.path.exists(weights_file):
    model.load_weights(weights_file)
else:
    st.warning(f"Warning: {weights_file} not found. Model will use random weights.")

# Streamlit UI
st.title("Sentiment Analysis App")
user_input = st.text_area("Enter a review:")

def preprocess_text(text):
    from tensorflow.keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts([text])
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequences, maxlen=max_len)
    return padded_sequence

if st.button("Analyze Sentiment"):
    if user_input.strip():
        processed_text = preprocess_text(user_input)
        prediction = model.predict(processed_text)[0][0]
        sentiment = "Positive" if prediction > 0.5 else "Negative"
        st.write(f"Predicted Sentiment: {sentiment}")
    else:
        st.write("Please enter a review before analyzing.")
