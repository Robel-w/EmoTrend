import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import joblib


def extract_features(df, vectorizer_path='../models/vectorizer.pkl'):
    vectorizer = TfidfVectorizer(max_features=500)
    X_tfidf = vectorizer.fit_transform(df['clean_text']).toarray()

    # Resolve vectorizer path relative to repo root (two levels up from this file)
    base_dir = Path(__file__).resolve().parent.parent
    vec_path = Path(vectorizer_path)
    if not vec_path.is_absolute():
        vec_path = base_dir / vec_path
    vec_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, str(vec_path))

    df['polarity'] = df['clean_text'].apply(lambda x: TextBlob(x).sentiment.polarity)

    df['exclamations'] = df['text'].apply(lambda x: x.count('!'))
    df['questions'] = df['text'].apply(lambda x: x.count('?'))
    df['caps_words'] = df['text'].apply(lambda x: sum(1 for w in x.split() if w.isupper()))
    df['text_length'] = df['text'].apply(lambda x: len(x.split()))

    X = np.hstack([X_tfidf,
                   df[['polarity','exclamations','questions','caps_words','text_length']].values])
    return X, df['intensity'].values
