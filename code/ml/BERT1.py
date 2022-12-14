import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import Sequential, layers
from matplotlib import pyplot

from ml.preprocess_text import preprocess_text, save_df_to_csv


def main():
    # filename = "../data/tweeter_3.csv"
    # filename = "../data/clean_tweeter_3.csv"
    filename = "../data/clean_twitter_scale.csv"
    # filename = "../data/clean_reddit_cleaned.csv"

    df = pd.read_csv(filename)
    df.drop(columns=['processed'])

    df = preprocess_text(df)
    # save_df_to_csv(df, "../data/clean2_tweeter_3.csv")

    # tweets = df.values[:, 1]
    # tweets = df["message"]
    tweets = df["processed"]
    # labels = df.values[:, 2].astype(float)
    labels = df["label"].astype(float)


    bert_model = SentenceTransformer('distilbert-base-nli-mean-tokens')

    embeddings = bert_model.encode(tweets, show_progress_bar=True)
    print(embeddings.shape)

    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels,
                                                        test_size=0.2, random_state=42)
    print("Training set shapes:", X_train.shape, y_train.shape)
    print("Test set shapes:", X_test.shape, y_test.shape)

    classifier = Sequential()
    classifier.add(layers.Dense(256, activation='relu', input_shape=(768,)))
    classifier.add(layers.Dense(1, activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['Recall'])

    hist = classifier.fit(X_train, y_train, epochs=10, batch_size=16,
                          validation_data=(X_test, y_test))

    pyplot.figure(figsize=(15, 5))
    pyplot.subplot(1, 2, 1)
    pyplot.plot(hist.history['loss'], 'r', label='Training loss')
    pyplot.plot(hist.history['val_loss'], 'g', label='Validation loss')
    pyplot.legend()
    pyplot.subplot(1, 2, 2)
    pyplot.plot(hist.history['recall'], 'r', label='Training recall')
    pyplot.plot(hist.history['val_recall'], 'g', label='Validation recall')
    pyplot.legend()
    pyplot.show()

if __name__ == "__main__":
    main()