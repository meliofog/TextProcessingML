import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re


dataset_name = "sample.csv"

try:
    data = pd.read_csv(dataset_name)
    print("Data Loaded successfully!")
except pd.errors.ParserError as e:
    print("Error occurred while reading CSV file:", e)
    exit()

# Function pour le nettoyage des données
"""
Mise en miniscules
Suppression des ponctuations, mots vides, mots fréquents, mots rares, émojis, URL et des balises HTML
Simplification de texte (stemming)
Lemmatisation
"""

def nettoyage_data(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    clean_text = ' '.join(tokens)
    return clean_text

# Nettoyage de data
data['text_cleaned'] = data['text'].apply(nettoyage_data)
print("Data cleaning!")

# Nouveau dataframe
processed_data = data[['tweet_id', 'text', 'text_cleaned']]
print("Writing cleaned_data DataFrame to a new CSV file!")
processed_data.to_csv(f"cleaned_{dataset_name}", index=False)
