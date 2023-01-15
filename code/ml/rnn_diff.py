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
from keras.models import Sequential
from keras.layers import InputLayer, Dense, SimpleRNN, Activation, Dropout, Conv1D
from keras.layers import Embedding, Flatten, LSTM, GRU
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.callbacks import EarlyStopping
from keras.layers import TextVectorization
from keras import layers
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import spacy
from sklearn.model_selection import train_test_split



def load_glove_model(glove_file):
    print("[INFO]Loading GloVe Model...")
    model = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embeddings = [float(val) for val in split_line[1:]]
            model[word] = embeddings
    print("[INFO] Done...{} words loaded!".format(len(model)))
    return model
# adopted from utils.py
nlp =  spacy.load("en_core_web_sm")

# def remove_stopwords(sentence):
#     '''
#     function to remove stopwords
#         input: sentence - string of sentence
#     '''
#     new = []
#     # tokenize sentence
#     sentence = nlp(sentence)
#     for tk in sentence:
#         if (tk.is_stop == False) & (tk.pos_ !="PUNCT"):
#             new.append(tk.string.strip())
#     # convert back to sentence string
#     c = " ".join(str(x) for x in new)
#     return c


def lemmatize(sentence):
    '''
    function to do lemmatization
        input: sentence - string of sentence
    '''
    sentence = nlp(sentence)
    s = ""
    for w in sentence:
        s +=" "+w.lemma_
    return nlp(s)

def sent_vectorizer(sent, model):
    '''
    sentence vectorizer using the pretrained glove model
    '''
    sent_vector = np.zeros(300)
    num_w = 0
    for w in sent.split():
        try:
            # add up all token vectors to a sent_vector
            sent_vector = np.add(sent_vector, model[str(w)])
            num_w += 1
        except:
            pass
    return sent_vector

def main():
    # data = pd.read_csv("../data/clean_reddit_cleaned.csv")
    # data = pd.read_csv('../data/clean_tweeter_3.csv')
    data = pd.read_csv('../data/clean_twitter_scale.csv')
    file_title = "rnn_diff_twitter_scale"
    fig_title = "twitter_scale dataset"

    # data_X  = data["clean_text"].to_numpy()
    data_X = data["message"].to_numpy()
    # data_y  = data["is_depression"].to_numpy()
    data_y  = data["label"].to_numpy()
    data_y = pd.get_dummies(data_y).to_numpy()

    glove_model = load_glove_model("./embedding/glove.6B.300d.txt")
    # number of vocab to keep
    max_vocab = 18000
    # length of sequence that will generate
    max_len = 10
    tokenizer = Tokenizer(num_words=max_vocab)
    tokenizer.fit_on_texts(data_X)
    sequences = tokenizer.texts_to_sequences(data_X)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    data_keras = pad_sequences(sequences, maxlen=max_len, padding="post")

    train_X, valid_X, train_y, valid_y = train_test_split(data_keras, data_y, test_size=0.3, random_state=42)

    # calcultaete number of words
    nb_words = len(tokenizer.word_index) + 1

    # obtain the word embedding matrix
    embedding_matrix = np.zeros((nb_words, 300))
    for word, i in word_index.items():
        embedding_vector = glove_model.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

    # adopted from sent_tran_eval.py
    def build_model(nb_words, rnn_model="SimpleRNN", embedding_matrix=None):
        '''
        build_model function:
        inputs: 
            rnn_model - which type of RNN layer to use, choose in (SimpleRNN, LSTM, GRU)
            embedding_matrix - whether to use pretrained embeddings or not
        '''
        model = Sequential()
        # add an embedding layer
        if embedding_matrix is not None:
            model.add(Embedding(nb_words,
                                300,
                                weights=[embedding_matrix],
                                input_length=max_len,
                                trainable=False))
        else:
            model.add(Embedding(nb_words,
                                300,
                                input_length=max_len,
                                trainable=False))

        # add an RNN layer according to rnn_model
        if rnn_model == "SimpleRNN":
            model.add(SimpleRNN(300))
        elif rnn_model == "LSTM":
            model.add(LSTM(300))
        else:
            model.add(GRU(300))
        # model.add(Dense(500,activation='relu'))
        # model.add(Dense(500, activation='relu'))
        model.add(Dense(2, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['Recall'])
        return model

    model_rnn = build_model(nb_words, "SimpleRNN", embedding_matrix)
    hist1 = model_rnn.fit(train_X, train_y, epochs=200, batch_size=120,
                  validation_data=(valid_X, valid_y)
                          #,callbacks=EarlyStopping(monitor='val_recall', mode='max', patience=5)
                          )
    predictions = model_rnn.predict(valid_X)
    predictions = predictions.argmax(axis=1)
    print(classification_report(valid_y.argmax(axis=1), predictions))


    model_lstm = build_model(nb_words, "LSTM", embedding_matrix)
    hist2 = model_lstm.fit(train_X, train_y, epochs=200, batch_size=120,
                   validation_data=(valid_X, valid_y)
                          #,callbacks=EarlyStopping(monitor='val_recall', mode='max', patience=5)
                          )
    predictions = model_lstm.predict(valid_X)
    predictions = predictions.argmax(axis=1)
    print(classification_report(valid_y.argmax(axis=1), predictions))


    model_gru = build_model(nb_words, "GRU", embedding_matrix)
    hist3 = model_gru.fit(train_X, train_y, epochs=200, batch_size=120,
                  validation_data=(valid_X, valid_y)
                          #,callbacks=EarlyStopping(monitor='val_recall', mode='max', patience=5)
                          )
    predictions = model_gru.predict(valid_X)
    predictions = predictions.argmax(axis=1)
    print(classification_report(valid_y.argmax(axis=1), predictions))

    pyplot.plot(hist1.history['val_recall'], 'r', label='SimpleRNN Recall')
    pyplot.plot(hist2.history['val_recall'], 'g', label='LSTM Recall')
    pyplot.plot(hist3.history['val_recall'], 'b', label='GRU Recall')
    pyplot.legend()
    pyplot.ylabel("Recall")
    pyplot.xlabel("Epochs")
    pyplot.title(fig_title)
    # pyplot.savefig("../results/figures/" + file_title + ".png")
    pyplot.show()


if __name__ == "__main__":
    main()