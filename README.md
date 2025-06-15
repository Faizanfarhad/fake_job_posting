# 🛡️ Spot the Scam - Fake Job Detection using Machine Learning

## 🔍 Overview

Online job platforms are increasingly targeted by scammers. These fake job listings not only waste applicants’ time but also put their personal data and finances at risk.

**Spot the Scam** is a machine learning-powered project that detects fraudulent job postings **before** users apply.

### 🎯 Features:
- **Trained binary classifier** (Real vs Fake jobs)
- **Preprocessing & feature extraction pipeline**
- **Interactive dashboard** built with **Streamlit**
- **Insightful visualizations** to explore scam patterns

---

## 🚨 Problem Statement

Manual detection of fake job listings is slow and unreliable. This project aims to **automate scam detection** using machine learning trained on real-world job listing data.

---

## ✨ Key Features

- 📂 **CSV Upload**: Accepts job listings with `title`, `description`, etc.
- ✅ **Prediction**: Classifies each job as Real (0) or Fake (1) with a confidence score
- 📊 **Visual Insights**:
  - Histogram of fraud probabilities
  - Pie chart of real vs fake jobs
  - Table of predictions
  - Top 10 most suspicious listings

> **Model**: Logistic Regression trained on cleaned dataset

---

## 🧰 Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Joblib
- Matplotlib, Seaborn

---

## 📁 Project Structure

```
spot_the_scam_dashboard/
├── dashboard.py           # Main Streamlit app
├── eda.py                 # EDA and preprocessing
├── processing.py          # Data cleaning and text feature generation
├── model_training.py      # Model training and saving
├── requirements.txt
├── models/
│   ├── fake_job_model.pkl
│   └── tfidf_vectorizer.pkl
├── datasets/
│   ├── train.csv
│   └── cleaned_data.csv
```

---

## 🛠️ How to Run Locally

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/spot-the-scam.git
   cd spot-the-scam
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App**
   ```bash
   streamlit run dashboard.py
   ```

---

## 📈 Model Performance

- **Model**: Logistic Regression
- **F1 Score**: `0.7674` (Binary)
- **Precision**: `0.6758`
- **Recall**: `0.8879`
- **TF-IDF Vectorizer**: 7000 features
- **Class Balancing**: Stratified split, F1 evaluation

> **Overall Accuracy**: 0.9732

---

## 🌐 Deployed App

🔗 **Try Live**: [Fake Job Detection App](https://faizanfarhad-fake-job-posting-app-fl81iw.streamlit.app/)

📺 **Demo Video**: [Watch on YouTube](https://youtu.be/hCbrecWw39w?feature=shared)

---

## 👥 Team Members & Roles

| Name            | Role                                |
|-----------------|-------------------------------------|
| Ayaan Shaikh    | Data Cleaning, EDA                  |
| Faizan Farhad   | Feature Extraction, Model Training  |
| Dheevesh Pujari | Evaluation, Dashboard, Documentation|

---

## 🙏 Acknowledgements

Thanks to the **Streamlit** team for making dashboarding so simple and powerful.

---

## 🔗 Additional Resources

- 📁 **Project GitHub Repository**: [github.com/Faizanfarhad/fake_job_posting](https://github.com/Faizanfarhad/fake_job_posting)
- 📎 **Project Drive File**: [Google Drive](https://drive.google.com/file/d/1uBzpfzecB-fHRfHbG2eCwXBTOkkxWFyw/view?usp=drivesdk)

---
