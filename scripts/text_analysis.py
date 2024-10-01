
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pandas as pd

# Download stopwords and lemmatizer
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
df = pd.read_csv('../data/impression_300_llm.csv')

# Initialize tools
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def process_text(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    stemmed = [ps.stem(word) for word in filtered_words]
    lemmatized = [lemmatizer.lemmatize(word) for word in stemmed]
    return ' '.join(lemmatized)

# Apply to the 'Observation' column
df['Processed_Observation'] = df['Observation'].apply(process_text)
df.to_csv('../results/processed_data.csv', index=False)
