import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb

# Load the IMDB dataset
word_index = imdb.get_word_index()
reverse_word_index = {v: k for k, v in word_index.items()}

# Load the pre-trained model
model= load_model('simple_rnn_model.keras',compile=False)

##preprocess the input text
def preprocess_text(text):
    words=text.lower().split()
    encoded_review = [word_index.get(word, 2) for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

##prediction function
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment= 'positive' if prediction[0][0] > 0.5 else 'negative'
    return sentiment, prediction[0][0]

##streamlit app
import streamlit as st
st.title("IMDB movie review Sentiment Analysis with Simple RNN")
st.write("Enter a movie review to predict the movie experience:")

user_input = st.text_area('movie review')

if st.button('Classify'):

    preprocess_input = preprocess_text(user_input)

    ##make prediction
    prediction= model.predict(preprocess_input)
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'

    ## Display the result
    st.write(f"The sentiment of the review is: {sentiment}")
    st.write(f"Prediction score: {prediction[0][0]:.4f}")

else:
    st.write("Please enter a review and click 'Classify' to see the sentiment prediction.")