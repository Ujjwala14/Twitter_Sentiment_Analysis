import tensorflow as tf

# Custom initializer if it's not recognized by Keras by default
from tensorflow.keras.initializers import Orthogonal

custom_objects = {
    'Orthogonal': Orthogonal(gain=1.0, seed=None)
}

model = tf.keras.models.load_model('sentiment_model.h5', custom_objects=custom_objects)

import pickle



import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Assuming you have loaded the model and tokenizer as shown above

# Define the maximum sequence length (use the same value as during training)
max_length = 100

# Create a function to predict sentiment
def predict_sentiment(text):
    # Tokenize and pad the input text
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=max_length, truncating='post')

    # Predict the sentiment
    prediction = model.predict(padded)
    return prediction

# Set up Streamlit app
st.title("Sentiment Analysis App")
user_input = st.text_area("Enter your text:")

if st.button("Predict"):
    if user_input:
        # Call the prediction function
        prediction = predict_sentiment(user_input)
        # Determine the sentiment based on the prediction
        sentiment = ["Negative", "Neutral", "Positive"][prediction.argmax()]
        # Display the result
        st.write(f"Sentiment: {sentiment}")
    else:
        st.warning("Please enter some text for analysis.")
