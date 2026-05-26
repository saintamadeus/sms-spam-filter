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

## Actual Results

| Metric    | Actual Score |
|-----------|--------------|
| Accuracy  | 97.58%       |
| Precision | 99.19%       |
| Recall    | 82.55%       |
| F1 Score  | 90.11%       |


