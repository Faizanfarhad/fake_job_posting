import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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

    # Combine text fields
    df['combined_text'] = df[['title', 'description', 'company_profile', 'requirements', 'benefits']].fillna('').agg(' '.join, axis=1)

    # Vectorize
    X = vectorizer.transform(df['combined_text'])

    # Predict
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)

    df['fraud_probability'] = probs
    df['prediction'] = preds

    st.write("📋 Prediction Results:", df[['title', 'location', 'fraud_probability', 'prediction']])

    # Pie chart
    st.subheader("🔍 Real vs Fake Distribution")
    st.pyplot(plt.pie(df['prediction'].value_counts(), 
                      labels=['Real', 'Fake'], 
                      autopct='%1.1f%%', colors=['green', 'red'])[0].figure)

    # Histogram of probabilities
    st.subheader("📊 Fraud Probability Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['fraud_probability'], bins=20, kde=True, ax=ax, color='orange')
    st.pyplot(fig)