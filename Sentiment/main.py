import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the pre-trained model
model = tf.keras.models.load_model('sentiment_analysis_model.h5')

# Set up the tokenizer
tokenizer = Tokenizer(num_words=1000)

# Set up the UI
st.title("Sentiment Analysis App")

input_text = st.text_input("Enter your text here:")
if st.button("Analyze"):
    # Preprocess the input text
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=100)

    # Make a prediction using the model
    prediction = model.predict(input_seq)[0]
    sentiment = 'Positive' if prediction[0] > prediction[1] else 'Negative'

    # Display the prediction
    st.write("Prediction:", sentiment)
