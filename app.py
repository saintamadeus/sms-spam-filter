import streamlit as st
import pickle
import sys
import os

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

    return model, vectorizer

model, vectorizer = load_model()

# Sidebar
st.sidebar.title("About")
st.sidebar.write(
    "Machine Learning SMS Spam Detection System "
    "using TF-IDF and Naive Bayes."
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

        cleaned = preprocess_text(user_input)

        vectorized = vectorizer.transform([cleaned])

        prediction = model.predict(vectorized)[0]

        probability = model.predict_proba(vectorized)[0]

        if prediction == 1:
            st.error("⚠️ This is a Spam Message")
            st.write(f"Confidence: {probability[1] * 100:.2f}%")
        else:
            st.success("✅ This is Not Spam")
            st.write(f"Confidence: {probability[0] * 100:.2f}%")
