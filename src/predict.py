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
