import pandas as pd
import nltk 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re 
from sklearn.feature_extraction.text import TfidfVectorizer

link = '/home/faizan/Fake_Job_Posting/datasets/fake_job_postings.csv'
class Dataset:
    def __init__(self,csv_link:str):
        super().__init__()
        self.df = pd.read_csv(csv_link)
        ''' if any error shows by nlkt first uncomment this '''
        # nltk.download('punkt_tab')
        # nltk.download('stopwords')
        # nltk.download('wordnet')
        self.df['text'] = (
        self.df['title'].fillna('') + ' ' +
        self.df['company_profile'].fillna('') + ' ' +
        self.df['description'].fillna('') + ' ' +
        self.df['requirements'].fillna('') + ' ' +
        self.df['benefits'].fillna(''))

    def preprocess_text(self,text):
        text = str(text).lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
        # Remove HTML tags if any
        text = re.sub(r'<[^>]+>', '', text)
        # Remove punctuation but keep letters and spaces (CONSISTENT)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = nltk.word_tokenize(text, language="english")
        stopword = set(stopwords.words("english"))
        tokens = [word for word in tokens  if word not in stopword ]
        lemitizer = WordNetLemmatizer()
        token_lemitize = [lemitizer.lemmatize(words) for words in tokens]
        cleaned_text = ' '.join(token_lemitize)
        return cleaned_text
    
    def vectorizer(self):
        self.df['preprocessed_text'] = self.df['text'].apply(self.preprocess_text)
        self.df['text'] = self.df['preprocessed_text']
        tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        max_df=0.85,
        ngram_range=(1,2),
        norm='l2',
        sublinear_tf=True)
        tfidf_matrix = tfidf_vectorizer.fit_transform(self.df['preprocessed_text'])
        tfidf_features = tfidf_vectorizer.get_feature_names_out()
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=tfidf_features
        )
        return tfidf_df
    def target(self):
        return self.df['fraudulent']
if __name__ == '__main__':
    data = Dataset(link)
    