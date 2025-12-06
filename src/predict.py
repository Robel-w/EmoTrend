
import sys
import pandas as pd
import joblib
from features import extract_features
from preprocess import clean_text

def predict_emotion(text):
    model = joblib.load('../models/linear_reg.pkl')
    vectorizer = joblib.load('../models/vectorizer.pkl')
    
    clean = clean_text(text)
    
    X_tfidf = vectorizer.transform([clean]).toarray()
    
    from textblob import TextBlob
    import numpy as np
    polarity = TextBlob(clean).sentiment.polarity
    exclamations = text.count('!')
    questions = text.count('?')
    caps_words = sum(1 for w in text.split() if w.isupper())
    text_length = len(text.split())
    
    X = np.hstack([X_tfidf, [[polarity, exclamations, questions, caps_words, text_length]]])
    
    pred = model.predict(X)[0]
    return round(float(pred), 2)

if __name__ == "__main__":
    text_input = sys.argv[1]
    score = predict_emotion(text_input)
    print({"emotion_intensity": score})
