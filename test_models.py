import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score
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

print("Preprocessing text...")
df['cleaned_text'] = df['text'].apply(preprocess_text)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "SVM (C=10, balanced)": SVC(C=10, class_weight='balanced', probability=True),
    "MLP (Neural Net)": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42)
}

print("\n--- Testing Models ---")
for name, model in models.items():
    if name == "HistGradientBoosting":
        model.fit(X_train.toarray(), y_train)
        y_pred = model.predict(X_test.toarray())
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    r = recall_score(y_test, y_pred)
    f = f1_score(y_test, y_pred)
    print(f"{name} -> Recall: {r:.4f} | F1: {f:.4f}")
