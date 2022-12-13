import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tensorflow import keras
from sklearn.model_selection import train_test_split
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

nltk.download("all")

from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.preprocessing.text import one_hot
from keras.layers import Embedding
from keras.utils import pad_sequences
from keras.models import Sequential


def main():
    df = pd.read_csv("../../data/reddit_cleaned.csv")
    print(df.head())
    w = WordNetLemmatizer()
    for i in range(len(df)):
        review = re.sub('[^a-zA-Z]', ' ', df["clean_text"][i])
        review = review.lower()
        review = review.split()
        review = [w.lemmatize(word) for word in review if not word in set(stopwords.words("english"))]
        review = " ".join(review)
        df["clean_text"][i] = review
    print(df.head())

    s = set()
    for i in range(len(df)):
        k = df["clean_text"][i].split()
        for j in range(len(k)):
            s.add(k[j])
    voc_size = len(s)
    onehot_repr1 = [one_hot(words, voc_size) for words in df["clean_text"]]
    max = 0
    for i in onehot_repr1:
        if len(i) > max:
            max = len(i)

    sent_length = max
    embedded_docs1 = pad_sequences(onehot_repr1, padding='pre', maxlen=sent_length)
    embedding_vector_features = sent_length * 2
    model = Sequential()
    model.add(Embedding(voc_size, embedding_vector_features, input_length=sent_length))
    model.add((LSTM(100)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    Y = df["is_depression"]
    X_train, X_test, Y_train, Y_test = train_test_split(embedded_docs1, Y, test_size=0.2, random_state=10, stratify=Y)

    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=5, batch_size=16)

    Y_pred = model.predict(X_test)
    Y_pred = (Y_pred >= 0.5).astype("int")

    print(classification_report(Y_test, Y_pred))
    print(confusion_matrix(Y_test, Y_pred))


if __name__ == "__main__":
    main()
