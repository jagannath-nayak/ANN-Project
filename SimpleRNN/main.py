import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value : key for key, value in word_index.items()}

# load the model
model = load_model('SimpleRNN/simple_rnn_model.h5')

# Helper function
# Function to decode reviews
def decode_review(encoded_review):
    # Decode the review text
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Fuction to preprocess the input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    # Pad the sequence to a fixed length
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Streamlit app
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (positive or negative).")

# user input
user_input = st.text_area("Movie Review")

if st.button('Classify'):
    preprocessed_input = preprocess_text(user_input)

    # Make prediction
    prediction = model.predict(preprocessed_input)
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'

    # Display the result
    st.write(f"Sentiment:{sentiment}")
    st.write(f"Prediction Score:{prediction[0][0]}")  
else:
    st.write("Please enter a movie review")    