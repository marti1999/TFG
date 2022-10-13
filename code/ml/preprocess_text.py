import nltk
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


def preprocess_text(df):
    df['processed'] = df['message_to_examine'].apply(remove_stopword_punctuation)  # preprocess text
    df['processed'] = df.processed.str.replace(r"[0-9]", "")  # remove numbers
    return df


def create_tfidf(df):
    vectorizer = TfidfVectorizer()
    processed = vectorizer.fit_transform(df['processed'])
    return processed


def create_bow(df):
    bow_vectorizer = CountVectorizer()
    bow = bow_vectorizer.fit_transform(df['processed'])
    df_bow = pd.DataFrame(bow.todense())
    return df_bow
