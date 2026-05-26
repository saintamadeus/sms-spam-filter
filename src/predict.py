import pickle
import re
import pandas as pd
from scipy.sparse import hstack
from preprocess import preprocess_text

def load_model():
    with open('../models/naive_bayes_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('../models/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('../models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, vectorizer, scaler

def predict(message: str) -> str:
    model, vectorizer, scaler = load_model()

    # Meta Features
    msg_length = len(message)
    digit_count = len(re.findall(r'\d', message))
    special_count = len(re.findall(r'[^a-zA-Z0-9\s]', message))

    # Preprocessing
    cleaned = preprocess_text(message)

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
    is_spam = probability[1] >= THRESHOLD

    return "Spam" if is_spam else "Not Spam"
