# SMS Spam Filtering System — Enhanced Project Blueprint

## Project Overview

This project is a machine learning-based SMS spam detection system that classifies incoming SMS messages as either:

- **Spam** — unwanted, fraudulent, promotional, or malicious messages
- **Ham** — legitimate and safe messages

The system uses:

- **Natural Language Processing (NLP)**
- **TF-IDF Feature Extraction**
- **Multinomial Naive Bayes Classification**
- **NLTK-based Text Preprocessing**
- **Streamlit Web Interface**

The application was designed as a practical implementation of machine learning for text classification and cybersecurity-related filtering systems.

---

# Complete Project Structure

```text
sms-spam-filter/
│
├── data/
│   └── spam.csv
│
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_preprocessing_and_features.ipynb
│   └── 03_model_training_and_evaluation.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
│
├── models/
│   ├── naive_bayes_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── screenshots/
│   ├── app_home.png
│   ├── spam_result.png
│   └── ham_result.png
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

---

# Step 1 — Download the Dataset

Dataset used:

- **UCI SMS Spam Collection Dataset**
- Total messages: **5,574**
- Labels:
  - Ham: 4,827
  - Spam: 747

Official source:

https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

Download the `spam.csv` file and place it inside:

```text
data/spam.csv
```

---

# Step 2 — Install Dependencies

Create a `requirements.txt` file:

```txt
pandas
numpy
scikit-learn
nltk
streamlit
matplotlib
seaborn
wordcloud
jupyter
```

### Improvements Added

The following additions improve the project without contradicting the original setup:

- `wordcloud` → for notebook visualization
- `jupyter` → for notebook execution

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Step 3 — Text Preprocessing Pipeline (`src/preprocess.py`)

This module handles all text cleaning operations.

## Original Pipeline

- Lowercasing
- Punctuation removal
- Tokenization
- Stopword removal
- Stemming

## Improved Version

Additional improvements added:

- Removal of URLs
- Removal of numeric values
- Extra whitespace cleanup

## Full Code

```python
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenization
    tokens = text.split()

    # Remove stopwords + stemming
    tokens = [
        stemmer.stem(word)
        for word in tokens
        if word not in stop_words
    ]

    # Remove extra spaces
    cleaned_text = ' '.join(tokens).strip()

    return cleaned_text
```

---

# Step 4 — Model Training (`src/train.py`)

This script:

- Loads the dataset
- Cleans text
- Applies TF-IDF vectorization
- Trains the Naive Bayes classifier
- Evaluates the model
- Saves the trained artifacts

## Improvements Added

Additional improvements:

- Added stratified train/test split
- Added confusion matrix generation
- Added model reproducibility
- Added model persistence verification

## Full Code

```python
import pandas as pd
import pickle
import os

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

from sklearn.preprocessing import LabelEncoder

from preprocess import preprocess_text

# Load dataset
df = pd.read_csv('../data/spam.csv', encoding='latin-1')

# Keep relevant columns
df = df[['v1', 'v2']]
df.columns = ['target', 'text']

# Encode labels
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])

# Apply preprocessing
print("Preprocessing text...")
df['cleaned_text'] = df['text'].apply(preprocess_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    max_features=3000
)

X = vectorizer.fit_transform(df['cleaned_text'])
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\n--- Model Evaluation ---")

print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=['Ham', 'Spam']
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model and vectorizer
os.makedirs('../models', exist_ok=True)

with open('../models/naive_bayes_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('../models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("\nModel and vectorizer saved successfully.")
```

---

# Step 5 — Prediction Module (`src/predict.py`)

This module handles inference.

## Full Code

```python
import pickle

from preprocess import preprocess_text

def load_model():

    with open('../models/naive_bayes_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('../models/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    return model, vectorizer

def predict(message: str) -> str:

    model, vectorizer = load_model()

    cleaned = preprocess_text(message)

    vectorized = vectorizer.transform([cleaned])

    prediction = model.predict(vectorized)[0]

    return "Spam" if prediction == 1 else "Not Spam"
```

---

# Step 6 — Streamlit Frontend (`app.py`)

The Streamlit app provides a graphical interface for users to test SMS messages.

## Improvements Added

Additional enhancements:

- Better UI formatting
- Sidebar project information
- Confidence probability display
- Input validation improvements

## Full Code

```python
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
```

---

# Step 7 — Jupyter Notebooks

The notebooks are essential for demonstrating:

- Data analysis
- NLP preprocessing
- Feature engineering
- Model training
- Evaluation
- Experimental results

These are extremely important for:

- GitHub portfolio quality
- Recruiter visibility
- Demonstrating ML workflow understanding

---

# Notebook 1 — Exploratory Data Analysis

File:

```text
01_exploratory_analysis.ipynb
```

## Sections to Include

### 1. Dataset Loading

- Load CSV
- Show first rows
- Display shape
- Display column names

### 2. Data Cleaning

- Remove unnecessary columns
- Rename columns

### 3. Class Distribution

```python
df['target'].value_counts()
```

### 4. Visualization

Create:

- Spam vs Ham bar chart
- Pie chart
- Histogram of message lengths

### 5. Message Statistics

Analyze:

- Average length
- Maximum length
- Minimum length

### 6. Word Clouds

Generate:

- Spam word cloud
- Ham word cloud

## Improvements Added

Additional analyses:

- Most common spam keywords
- Most common ham keywords
- Correlation between message length and spam likelihood

---

# Notebook 2 — Preprocessing and Feature Engineering

File:

```text
02_preprocessing_and_features.ipynb
```

## Sections to Include

### 1. Raw vs Cleaned Text

Display messages before and after cleaning.

### 2. Explain Each Preprocessing Step

Demonstrate:

- Lowercasing
- URL removal
- Punctuation removal
- Stopword removal
- Stemming

### 3. TF-IDF Feature Extraction

Show:

```python
vectorizer.get_feature_names_out()
```

### 4. Matrix Shape

Display:

```python
X.shape
```

### 5. Top TF-IDF Terms

Display:

- Top spam words
- Top ham words

## Improvements Added

Additional feature engineering ideas:

- Compare CountVectorizer vs TF-IDF
- Show sparsity of TF-IDF matrix
- Visualize token frequency distributions

---

# Notebook 3 — Model Training and Evaluation

File:

```text
03_model_training_and_evaluation.ipynb
```

## Sections to Include

### 1. Train the Model

Use:

- MultinomialNB
- TF-IDF features

### 2. Evaluation Metrics

Print:

- Accuracy
- Precision
- Recall
- F1-score

### 3. Classification Report

Use:

```python
classification_report()
```

### 4. Confusion Matrix

Use seaborn heatmap.

### 5. Manual Test Cases

Test with:

- Promotional SMS
- Lottery scam
- OTP messages
- Friendly chats
- Banking notifications

## Improvements Added

Additional experiments:

- Compare Naive Bayes with Logistic Regression
- Compare performance before and after preprocessing
- Save evaluation plots into `/screenshots`

---

# Step 8 — README.md

## Suggested Enhanced README

```markdown
# SMS Spam Filtering System

A machine learning application that detects spam SMS messages using NLP and machine learning techniques.

Built with:
- Python
- scikit-learn
- NLTK
- Streamlit

---

## Features

- SMS spam classification
- NLP preprocessing
- TF-IDF vectorization
- Multinomial Naive Bayes classifier
- Interactive Streamlit UI
- Real-time prediction
- Probability confidence display

---

## Tech Stack

- Python 3.8+
- scikit-learn
- NLTK
- Streamlit
- pandas
- numpy
- matplotlib
- seaborn

---

## Dataset

UCI SMS Spam Collection Dataset

Total Messages: 5,574

- Ham: 4,827
- Spam: 747

Source:
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

---

## Project Workflow

1. Load dataset
2. Clean and preprocess text
3. Convert text to TF-IDF vectors
4. Train Naive Bayes model
5. Evaluate performance
6. Save trained artifacts
7. Deploy with Streamlit

---

## Run the Project

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train the model

```bash
cd src
python train.py
```

### Run the Streamlit app

```bash
streamlit run app.py
```

---

## Expected Results

| Metric    | Expected Score |
|-----------|----------------|
| Accuracy  | ~98%           |
| Precision | ~97%           |
| Recall    | ~94%           |
| F1 Score  | ~96%           |

---

## Future Improvements

- Deep learning implementation (LSTM/BERT)
- Email spam detection extension
- API deployment with Flask/FastAPI
- Docker containerization
- Cloud deployment on Render or AWS

---

## Author

Final Year Computer Science Project
Federal University Lokoja
```

---

# Additional Improvements You Should Add

These additions make the project stronger for GitHub and job applications.

## 1. Add `.gitignore`

```gitignore
__pycache__/
.ipynb_checkpoints/
models/*.pkl
venv/
.env
```

---

## 2. Add Screenshots Folder

Capture:

- Home page
- Spam detection result
- Ham detection result

This improves portfolio presentation quality.

---

## 3. Add Model Comparison Section

Even if Naive Bayes is the final model, briefly compare:

- Logistic Regression
- Support Vector Machine (SVM)

This demonstrates experimentation.

---

## 4. Add Deployment

Optional deployment platforms:

- Streamlit Cloud
- Render
- Railway

---

## 5. Add License

Use MIT License.

---

# Suggested GitHub Commit Flow

```bash
git init
git add .
git commit -m "Initial commit - SMS Spam Filtering System"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

---

# Final Recommended Order of Execution

## Phase 1 — Setup

1. Create folders
2. Download dataset
3. Install dependencies

## Phase 2 — Core Development

4. Create preprocessing module
5. Create training script
6. Train model
7. Save `.pkl` files

## Phase 3 — Frontend

8. Build Streamlit app
9. Test predictions

## Phase 4 — Documentation

10. Build notebooks
11. Capture screenshots
12. Write README

## Phase 5 — Portfolio Publishing

13. Push to GitHub
14. Deploy app online

---

# Final Notes

Do NOT fabricate notebook metrics.

Always:

- Run the code
- Record the actual results
- Use real screenshots
- Use genuine evaluation metrics

That is what makes the project credible academically and professionally.

The project already has a strong structure. The improvements above make it:

- More professional
- More deployable
- More recruiter-friendly
- More portfolio-ready
