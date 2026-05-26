import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score
from sklearn.preprocessing import LabelEncoder
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from preprocess import preprocess_text

# Load dataset
df = pd.read_csv('data/spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['target', 'text']
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])

# Use preprocessed text if we don't want to wait
print("Preprocessing text...")
df['cleaned_text'] = df['text'].apply(preprocess_text)

# Try different max_features and n-grams
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = MultinomialNB()
model.fit(X_train, y_train)

y_proba = model.predict_proba(X_test)[:, 1]

print("\n--- Threshold Tuning ---")
for t in np.arange(0.05, 0.5, 0.05):
    y_pred = (y_proba >= t).astype(int)
    r = recall_score(y_test, y_pred)
    f = f1_score(y_test, y_pred)
    p = precision_score(y_test, y_pred)
    a = accuracy_score(y_test, y_pred)
    print(f"Threshold: {t:.2f} | Recall: {r:.4f} | F1: {f:.4f} | Precision: {p:.4f} | Acc: {a:.4f}")
