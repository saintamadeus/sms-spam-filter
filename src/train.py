import pandas as pd
import pickle
import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy.sparse import hstack

from preprocess import preprocess_text

# Load dataset
df = pd.read_csv('../data/spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['target', 'text']

# Feature Engineering (Meta Features)
df['msg_length'] = df['text'].apply(len)
df['digit_count'] = df['text'].apply(lambda x: len(re.findall(r'\d', x)))
df['special_count'] = df['text'].apply(lambda x: len(re.findall(r'[^a-zA-Z0-9\s]', x)))

# Encode labels
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])

# Apply preprocessing
print("Preprocessing text...")
df['cleaned_text'] = df['text'].apply(preprocess_text)

# TF-IDF Vectorization with N-Grams
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X_tfidf = vectorizer.fit_transform(df['cleaned_text'])

# Scale Meta Features
scaler = MinMaxScaler()
X_meta = scaler.fit_transform(df[['msg_length', 'digit_count', 'special_count']])

# Combine Features
X = hstack([X_tfidf, X_meta])
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions with custom threshold
THRESHOLD = 0.15
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= THRESHOLD).astype(int)

# Evaluation
print(f"\n--- Model Evaluation (Threshold: {THRESHOLD}) ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model, vectorizer, and scaler
os.makedirs('../models', exist_ok=True)
with open('../models/naive_bayes_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('../models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('../models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\nModel, vectorizer, and scaler saved successfully.")
