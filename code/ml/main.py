import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.wordnet import WordNetLemmatizer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer # https://en.wikipedia.org/wiki/Tf%E2%80%93idf

nltk.download('stopwords')
nltk.download('punkt')
nlp = spacy.load('en_core_web_sm')
lmtzr = WordNetLemmatizer()

def text_preprocess(text):
    lm = []
    text = nlp(text)
    for word in text:
        lm.append(word.lemma_) # change verbs to infintive form

    text = " ".join(lm)
    text = text.translate(str.maketrans("", "", string.punctuation)) # removing punctuation
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')] # removing stopwords

    return " ".join(text)

if __name__ == "__main__":
    df = pd.read_csv('../data/tweeter_3.csv')
    # print(df.describe())
    # print(df.info())
    df.columns = df.columns.str.replace(" ", "_")
    df = df.head(100)

    df['processed'] = df['message_to_examine'].apply(text_preprocess) # preprocess text
    df['processed1'] = df.processed.str.replace(r"[0-9]", "") # remove numbers

    vectorizer = TfidfVectorizer()
    processed = vectorizer.fit_transform(df['processed1'])
    print(1)

