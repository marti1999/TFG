import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
import re
import string

from matplotlib import pyplot
from nltk.corpus import stopwords
nltk.download("stopwords")
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.layers import TextVectorization
from keras import layers
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def main():
    # data = pd.read_csv("../data/clean_reddit_cleaned.csv")
    # data = pd.read_csv('../data/clean_tweeter_3.csv')
    data = pd.read_csv('../data/clean_twitter_scale.csv')
    stemmer = nltk.SnowballStemmer("english")
    stopword=set(stopwords.words('english'))

    def clean(text):
        text = str(text).lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        text = [word for word in text.split(' ') if word not in stopword]
        text = " ".join(text)
        text = [stemmer.stem(word) for word in text.split(' ')]
        text = " ".join(text)
        return text

    # data["clean_text"] = data["clean_text"].apply(clean)
    data["message"] = data["message"].apply(clean)
    # x = data["clean_text"]
    x = data["message"]
    # y = data["is_depression"]
    y = data["label"]

    fig_title = "twitter_scale_clean"

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)

    max_vocab_length = 10000
    max_length = 34

    text_vectorizer = TextVectorization(max_tokens=max_vocab_length,
                                        output_mode="int",
                                        output_sequence_length=max_length)
    text_vectorizer.adapt(X_train)
    words_in_vocab = text_vectorizer.get_vocabulary()
    top_5_words = words_in_vocab[:5]
    bottom_5_words = words_in_vocab[-5:]
    print(f"Vocablary size: {len(words_in_vocab)}")
    print(f"Top 5 most common words: {top_5_words}")
    print(f"Bottom 5 least common words:: {bottom_5_words}")


    embedding = layers.Embedding(input_dim=max_vocab_length,
                                 output_dim=128,
                                 embeddings_initializer="uniform",
                                 input_length=max_length
                                 )
    inputs = layers.Input(shape=(1,), dtype=tf.string)
    x = text_vectorizer(inputs)
    x = embedding(x)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model_1 = tf.keras.Model(inputs, outputs, name="model_1_dense")
    model_1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_1.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=5)


    inputs = layers.Input(shape=(1,), dtype=tf.string)
    x = text_vectorizer(inputs)
    x = embedding(x)
    x = layers.LSTM(64, activation="tanh")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model_2 = tf.keras.Model(inputs, outputs, name="model_2_lstm")
    model_2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['Recall'])
    hist = model_2.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=5)

    Y_pred = model_2.predict(X_test)
    Y_pred = (Y_pred >= 0.5).astype("int")

    pyplot.figure(figsize=(15, 5))
    pyplot.subplot(1, 2, 1)
    pyplot.plot(hist.history['loss'], 'r', label='Training loss')
    pyplot.plot(hist.history['val_loss'], 'g', label='Validation loss')
    pyplot.legend()
    pyplot.subplot(1, 2, 2)
    pyplot.plot(hist.history['recall'], 'r', label='Training recall')
    pyplot.plot(hist.history['val_recall'], 'g', label='Validation recall')
    pyplot.legend()
    pyplot.savefig("../results/figures/" + fig_title + ".png")
    pyplot.show()

    print(classification_report(Y_test, Y_pred))


if __name__ == "__main__":
    main()