import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return ' '.join(tokens)

def preprocess_dataset(input_path, output_path):
    df = pd.read_csv(input_path)
    df['clean_text'] = df['text'].apply(clean_text)
    df.to_csv(output_path, index=False)
    print(f"Preprocessing done! Saved to {output_path}")

if __name__ == "__main__":
    preprocess_dataset('../data/raw.csv', '../data/processed.csv')

