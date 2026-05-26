import streamlit as st
import pickle
import sys
import os
import re
import pandas as pd
from scipy.sparse import hstack

sys.path.append(
    os.path.join(os.path.dirname(__file__), 'src')
)

from preprocess import preprocess_text

# Page configuration
st.set_page_config(
    page_title="SMS Spam Filter",
    page_icon="📩",
    layout="centered"
)

# Load model
@st.cache_resource
def load_model():
    with open('models/naive_bayes_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, vectorizer, scaler

model, vectorizer, scaler = load_model()

# Sidebar
st.sidebar.title("About")
st.sidebar.write(
    "Machine Learning SMS Spam Detection System "
    "using TF-IDF, Meta Features, and Naive Bayes."
)

# Main UI
st.title("📩 SMS Spam Filtering System")

st.write(
    "Type or paste an SMS message below "
    "to determine whether it is spam."
)

user_input = st.text_area(
    "Enter SMS Message",
    height=150
)

if st.button("Check Message"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Meta Features
        msg_length = len(user_input)
        digit_count = len(re.findall(r'\d', user_input))
        special_count = len(re.findall(r'[^a-zA-Z0-9\s]', user_input))

        # Preprocessing
        cleaned = preprocess_text(user_input)

        # Feature extraction
        vectorized = vectorizer.transform([cleaned])
        
        # Scale meta features
        meta_df = pd.DataFrame([[msg_length, digit_count, special_count]], columns=['msg_length', 'digit_count', 'special_count'])
        meta_scaled = scaler.transform(meta_df)

        # Combine features
        combined_features = hstack([vectorized, meta_scaled])

        # Predict using threshold
        THRESHOLD = 0.15
        probability = model.predict_proba(combined_features)[0]
        prediction = 1 if probability[1] >= THRESHOLD else 0

        if prediction == 1:
            st.error("⚠️ This is a Spam Message")
            st.write(f"Confidence (Model Probability): {probability[1] * 100:.2f}%")
        else:
            st.success("✅ This is Not Spam")
            st.write(f"Confidence (Model Probability): {probability[0] * 100:.2f}%")
