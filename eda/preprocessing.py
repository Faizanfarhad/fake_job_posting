import pandas as pd
import re
import string
import os

# Load the CSV (update path if needed)
df = pd.read_csv('datasets/fake_job_postings.csv')

# Drop irrelevant or low-value columns
df.drop(['job_id', 'salary_range', 'telecommuting', 'has_company_logo', 'has_questions'], axis=1, inplace=True)

# Fill NaNs in text columns with empty string
text_columns = ['title', 'location', 'department', 'company_profile', 'description',
                'requirements', 'benefits', 'required_experience', 'required_education',
                'industry', 'function']

for col in text_columns:
    df[col] = df[col].fillna('')

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'.*?', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# Combine important columns into one
df['combined_text'] = (
    df['title'] + ' ' + df['company_profile'] + ' ' +
    df['description'] + ' ' + df['requirements'] + ' ' +
    df['benefits'] + ' ' + df['industry'] + ' ' + df['function']
).apply(clean_text)

# Keep only necessary columns
df = df[['combined_text', 'fraudulent']]

# Ensure output directory exists
os.makedirs('datasets', exist_ok=True)

# Save cleaned data
df.to_csv('datasets/cleaned_fake_job_postings.csv', index=False)

print("✅ Preprocessing completed. Cleaned data saved to 'datasets/cleaned_fake_job_postings.csv'")