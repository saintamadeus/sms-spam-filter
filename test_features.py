import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score, precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import sys
import os
import re

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from preprocess import preprocess_text

# Load dataset
df = pd.read_csv('data/spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['target', 'text']
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])

# Feature Engineering before preprocessing
df['msg_length'] = df['text'].apply(len)
df['digit_count'] = df['text'].apply(lambda x: len(re.findall(r'\d', x)))
df['special_count'] = df['text'].apply(lambda x: len(re.findall(r'[^a-zA-Z0-9\s]', x)))

print("Preprocessing text...")
df['cleaned_text'] = df['text'].apply(preprocess_text)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_tfidf = vectorizer.fit_transform(df['cleaned_text'])

# Scale numeric features
scaler = StandardScaler()
X_meta = scaler.fit_transform(df[['msg_length', 'digit_count', 'special_count']])

# Combine
X = hstack([X_tfidf, X_meta])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = LogisticRegression(class_weight='balanced', max_iter=2000)
model.fit(X_train, y_train)

y_proba = model.predict_proba(X_test)[:, 1]

print("\n--- Logistic Regression + Meta Features ---")
for t in np.arange(0.3, 0.9, 0.05):
    y_pred = (y_proba >= t).astype(int)
    r = recall_score(y_test, y_pred)
    f = f1_score(y_test, y_pred)
    p = precision_score(y_test, y_pred)
    if r >= 0.94 or f >= 0.94:
        print(f"Threshold: {t:.2f} | Recall: {r:.4f} | F1: {f:.4f} | Precision: {p:.4f}")
