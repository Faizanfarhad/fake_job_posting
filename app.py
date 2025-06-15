import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from model.model import Model
import os 
import sys
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from eda import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Load model and vectorizer
model = joblib.load('fake_job_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Title
st.title("🛡️ Spot the Scam: Fake Job Detector")

# Upload file
uploaded_file = st.file_uploader("📤 Upload a job listings CSV", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("✅ Uploaded Data Preview:", df.head())

    # Show available columns
    st.write("📌 Available columns:", df.columns.tolist())

    # Combine text fields safely
    expected_columns = ['title', 'description', 'company_profile', 'requirements', 'benefits']
    existing_columns = [col for col in expected_columns if col in df.columns]

    df['combined_text'] = df[existing_columns].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)

    # Vectorize
    X = vectorizer.transform(df['combined_text'])

    # Predict
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)

    df['fraud_probability'] = probs
    df['prediction'] = preds
    df['label'] = np.where(df['fraud_probability'] < 0.6, 'Real', 'Fake')

    # Results
    st.write("📋 Prediction Results:", df[['title', 'location', 'fraud_probability', 'prediction', 'label']])

    # Pie chart
    st.subheader("🔍 Real vs Fake Distribution")
    fig1, ax1 = plt.subplots()
    value_counts = df['prediction'].value_counts()
    label_map = {0: 'Real', 1: 'Fake'}
    labels = [label_map[i] for i in value_counts.index]
    colors = ['green' if i == 0 else 'red' for i in value_counts.index]
    ax1.pie(value_counts, labels=labels, autopct='%1.1f%%', colors=colors)
    st.pyplot(fig1)
    # Histogram of probabilities
    st.subheader("📊 Fraud Probability Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['fraud_probability'], bins=20, kde=True, ax=ax, color='orange')
    st.pyplot(fig)
    
    
    # Get X_test and Y_test once
    X_test = joblib.load('X_test.pkl')
    Y_test = joblib.load('Y_test.pkl')


    # Only transform X_test ONCE
    # Suppose X_test contains text data (not vectorized)
    if isinstance(X_test, np.ndarray):
        X_test_clean = [str(x) for x in X_test.tolist()]
    else:
        X_test_clean = [str(x) for x in X_test]

    y_pred = joblib.load('prediction.pkl')
    # Now transform
    X_test_vectorizer = vectorizer.transform(X_test_clean)

    # Predict ONCE

    # Compute all scores directly

    accuracy = accuracy_score(Y_test, y_pred)
    precision = precision_score(Y_test, y_pred, zero_division=0)
    recall = recall_score(Y_test, y_pred, zero_division=0)
    f1 = f1_score(Y_test, y_pred, zero_division=0)
    micro = f1_score(Y_test, y_pred, average='micro', zero_division=0)
    binary = f1_score(Y_test, y_pred, average='binary', zero_division=0)
    weighted = f1_score(Y_test, y_pred, average='weighted', zero_division=0)

    # Show
    st.write(f"🎯 **Model Accuracy:** {accuracy:.4f}")
    st.write(f"🎯✅ **Precision:** {precision:.4f}")
    st.write(f"🔍 **Recall:** {recall:.4f}")
    st.write(f"⚡ **F1 Score:** {f1:.4f}")
    st.write(f"⚡ **F1 micro score:** {micro:.4f}")
    st.write(f"⚡ **F1 binary score:** {binary:.4f}")
    st.write(f"⚡ **F1 weighted score:** {weighted:.4f}")
