# 📘 EmoTrend – Emotional Intensity Prediction Engine

A lightweight NLP + Linear Regression system that quantifies the **strength of human emotions** in text.

---

## 🔍 Overview

**EmoTrend** predicts the **intensity of emotion** expressed in any sentence.  
Unlike conventional sentiment classifiers that output simple labels (positive/negative), EmoTrend produces a **continuous score** (0–1 or 0–100) representing emotional strength.  

This project combines **classical ML (Linear Regression)**, **NLP preprocessing**, **feature engineering**, and **evaluation metrics** to emulate how humans perceive emotion intensity in text.

---

## 🎯 Objectives

- Build an ML model using **Linear Regression** to estimate emotional intensity.
- Apply classical **NLP preprocessing techniques**.
- Engineer **linguistic and sentiment-based features**.
- Compare multiple regression models:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - ElasticNet
- Deploy the final model through a **clean prediction pipeline** (CLI or API).

---

## 🧠 How It Works

### 1. Preprocessing

- Lowercasing  
- Stopword removal  
- Lemmatization  
- Tokenization  
- TF-IDF vectorization  

### 2. Feature Engineering

In addition to TF-IDF, EmoTrend uses:

- **Sentiment polarity** (VADER or TextBlob)  
- **Punctuation markers** (e.g., number of exclamation marks)  
- **Emotion lexicon counts** (NRC or custom lists)  
- **Sentence length metrics**  

These features capture **emotional nuances** in a simple and explainable way.

---

## 🧩 Modeling Approach

Trained & evaluated using **scikit-learn**:

| Model             | Purpose                        |
|------------------|--------------------------------|
| Linear Regression | Baseline model                 |
| Ridge Regression  | Handles multicollinearity       |
| Lasso Regression  | Feature selection              |
| ElasticNet        | Balanced regularization         |

### Evaluation Metrics

- Mean Absolute Error (MAE)  
- Mean Squared Error (MSE)  
- R² Score  

The **best-performing model** is selected for deployment.

---

## 📊 Dataset

Two options for experimentation:

**Option A:** SemEval 2018 – Emotion Intensity Dataset  
Labels emotions (anger, joy, fear, sadness) from 0–1.

**Option B:** Self-built dataset of 200–300 manually labeled sentences.  

Data files are stored under:

/data/raw.csv
/data/processed.csv


---

## 🛠️ Project Structure

EmoTrend/
│
├── data/
│ ├── raw.csv
│ └── processed.csv
│
├── models/
│ ├── best_model.pkl
│ ├── vectorizer.pkl
│ └── scaler.pkl
│
├── notebooks/
│ ├── 01_preprocessing.ipynb
│ ├── 02_feature_engineering.ipynb
│ └── 03_model_training.ipynb
│
├── src/
│ ├── preprocess.py
│ ├── features.py
│ ├── model.py
│ └── predict.py
│
├── api/
│ ├── main.py
│ └── routes.py
│
├── tests/
│
├── README.md
└── requirements.txt


---

## 🚀 Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt

2. Run model training
python src/model.py

3. Predict emotion intensity
python src/predict.py "I am extremely excited today!"


Output:

{
  "emotion_intensity": 0.87
}

🌐 API (Optional Enhancement)

Using FastAPI:

POST /predict

{
  "text": "I am extremely happy!"
}


Response:

{
  "intensity": 0.82
}

💡 Notes & Next Steps

Feature expansion: Use n-grams, word embeddings (Word2Vec, GloVe) for richer features.

Model experimentation: Explore tree-based regressors (RandomForest, XGBoost) for comparison.

Deployment: Containerize via Docker or deploy on Heroku/Streamlit/FastAPI for live prediction.


---

If you want, I can also **add badges and a visually modern GitHub-style header** to make it pop for recruiters when they see your repo. This is often what makes an AI project *look professional at first glance*. Do you want me to do that?
