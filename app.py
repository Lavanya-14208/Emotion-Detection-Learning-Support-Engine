
import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load Model
model = load_model("models/bilstm_model.h5")

# Load Tokenizer
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load Label Encoder
with open("models/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

st.set_page_config(page_title="Emotion Detection", page_icon="😊")

st.title("😊 Emotion Detection Learning Support Engine")

st.write("Enter any sentence to predict the emotion.")

user_input = st.text_area("Enter Text")

if st.button("Predict Emotion"):

    if user_input.strip() != "":

        sequence = tokenizer.texts_to_sequences([user_input])

        padded = pad_sequences(sequence,maxlen=20)

        prediction = model.predict(padded)

        emotion = encoder.inverse_transform([np.argmax(prediction)])[0]

        confidence = np.max(prediction)*100

        st.success(f"Predicted Emotion : {emotion}")

        st.info(f"Confidence : {confidence:.2f}%")

    else:

        st.warning("Please enter some text.")
