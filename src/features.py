import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import joblib

def extract_features(df, vectorizer_path='../models/vectorizer.pkl'):
    vectorizer = TfidfVectorizer(max_features=500)
    X_tfidf = vectorizer.fit_transform(df['clean_text']).toarray()
    
    joblib.dump(vectorizer, vectorizer_path)
        df['polarity'] = df['clean_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    
    df['exclamations'] = df['text'].apply(lambda x: x.count('!'))
    df['questions'] = df['text'].apply(lambda x: x.count('?'))
    df['caps_words'] = df['text'].apply(lambda x: sum(1 for w in x.split() if w.isupper()))
    df['text_length'] = df['text'].apply(lambda x: len(x.split()))
    
    import numpy as np
    X = np.hstack([X_tfidf,
                   df[['polarity','exclamations','questions','caps_words','text_length']].values])
    return X, df['intensity'].values

