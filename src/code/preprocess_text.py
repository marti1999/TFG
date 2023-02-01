import re

import nltk
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
import spacy
import pandas as pd

import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('stopwords')
nltk.download('punkt')
nlp = spacy.load('en_core_web_sm')
lmtzr = WordNetLemmatizer()


def remove_stopword_punctuation(line):
    list_words = []
    line = nlp(line)
    for w in line:
        list_words.append(w.lemma_)  # change verbs to infintive form

    line = " ".join(list_words)
    line = line.translate(str.maketrans("", "", string.punctuation))  # removing punctuation
    line = [word for word in line.split() if word.lower() not in stopwords.words('english')]  # removing stopwords

    return " ".join(line)

def remove_usernames(line, pattern):
    r = re.findall(pattern, line)

    for i in r:
        line = re.sub(i, "", line)

    return line

def preprocess_text(df):
    df['processed'] = np.vectorize(remove_usernames)(df['message'], "@[\w]*")
    df['processed'] = df['processed'].apply(remove_stopword_punctuation)  # preprocess text
    df['processed'] = df.processed.str.replace(r"[0-9]", "")  # remove numbers
    df['processed'] = df.processed.str.replace("[^a-zA-Z#]", " ")  # remove numbers
    return df


def create_tfidf(df, max_features=None):
    vectorizer = TfidfVectorizer(max_features=max_features)
    processed = vectorizer.fit_transform(df['processed'].values.astype('U'))
    return processed


def create_bow(df, max_features=None):
    bow_vectorizer = CountVectorizer(max_features=max_features)
    bow = bow_vectorizer.fit_transform(df['processed'].values.astype('U'))
    df_bow = pd.DataFrame(bow.todense())
    return df_bow

def save_df_to_csv(df, file_name):
    df.to_csv(file_name)
