import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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