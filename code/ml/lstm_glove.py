import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from ml.old.customimagedataset import CustomImageDataset
from ml.rnn import RNN


def tokens_to_string(tokens, idx_to_word):  # Convert tokens back into their sting value
    words = [idx_to_word[token] for token in tokens]
    text = " ".join(words)
    return text


def main():
    # df_depression = pd.read_csv('../data/clean_reddit_cleaned.csv')
    # df_depression = pd.read_csv('../data/clean_tweeter_3.csv')
    df_depression = pd.read_csv('../data/clean_twitter_scale.csv')
    print(df_depression.head())

    # data = [x for x in df_depression['clean_text']]
    # labels = [x for x in df_depression['is_depression']]
    data = [x for x in df_depression['message']]
    labels = [x for x in df_depression['label']]
    data, labels = shuffle(data, labels)

    # Load our embeddings
    embeddings_text = open('./embedding/glove.6B.300d.txt', 'r', encoding='utf-8')

    # diccionri que té totes les paraules úniques del dataset
    words = {}
    for example in data:
        for word in example.split():
            if word not in words:
                words[word] = 0
            else:
                words[word] += 1

    # mapping de les paraules del diccionari amb el seu corresponent embedding
    embs = {}
    for line in embeddings_text:
        split = line.split()
        word = split[0]
        if word in words:
            try:
                embedding = np.array([float(value) for value in split[1:]])
                embs[word] = embedding
            except:
                print('error loading embedding')

    # mirem quantes no ha pogut mapejar
    missing_words = 0
    for word in words:
        if word not in embs:
            missing_words += 1
    print(missing_words)

    # ara creem la matriu embedding, de mides (vocab size * embedding dimensions)
    embedding_matrix = []
    idx_to_word = []
    word_to_idx = {}
    embedding_matrix.append(np.zeros(300))  # this will be our zero padding for the network
    idx_to_word.append('')
    word_to_idx[''] = 0
    for i, (word, emb) in enumerate(embs.items()):
        embedding_matrix.append(emb)
        idx_to_word.append(word)
        word_to_idx[word] = i + 1
    embedding_matrix = np.asarray(embedding_matrix)
    # fent alguens comprovacions
    index = word_to_idx['depression']
    print(index)
    print(idx_to_word[index])
    print(np.array_equal(embs['depression'], embedding_matrix[index]))
    print(embedding_matrix.shape)

    # passem el array of strings a array of integer tokens. S'eliminen les paraules les quals no es té embedding.
    x_train = []
    for example in data:
        temp = []
        for word in example.split():
            if word in word_to_idx:
                temp.append(word_to_idx[word])
        x_train.append(temp)
    x_train = np.asarray(x_train)
    # mirem quina és la mida màxima del dataset i la mitjana.
    max_length = 0
    for example in x_train:
        if len(example) > max_length:
            max_length = len(example)
    total_length = 0
    for i in range(len(x_train)):
        total_length += len(x_train[i])
    print("MITJANA: ", total_length / len(x_train))
    # de moment està hardcodejat, però max és >4k i mitjana 74. Ens quedem amb les 200 primeres que ja hi hauria
    # d'haver suficient i estalviem recursos a l'ordinador
    for i in range(len(x_train)):
        x_train[i] = x_train[i][:200]

    # mirem com ha quedat fins ara amb un exemple
    print(x_train[0])
    print(tokens_to_string(x_train[0], idx_to_word))

    # Ara cal fer un padding als missatges que són més curts que les 200 paraules que s'ha posat de límit.
    # Es fa padding amb 0 pel principi i no pel final per la naturales de les RNN.
    # Així comença amb els "activation values"  sent 0  i els anem passant. Com que està revent 0 fins a la primera paraula,
    # els valors d'activació seguirant sent 0 fins que trobi la primera paraula. Evita perdre informació.
    for i in range(len(x_train)):
        x_train[i] = np.pad(x_train[i], (200 - len(x_train[i]), 0), 'constant')
    x_train_data = []
    for x in x_train:
        x_train_data.append([k for k in x])
    x_train_data = np.array(x_train_data)
    print(x_train_data.shape)

    # creant el model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = RNN(torch.tensor(embedding_matrix), 128, 1, False).to(device)
    batch_size = 128
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # separem train i test, X i y
    dataset = CustomImageDataset(x_train_data, labels)
    train_length = int(len(dataset) * 0.8)  # 80% training data, 20% test data
    test_length = len(dataset) - train_length
    x_dataset, y_dataset = random_split(dataset, [train_length, test_length])
    x_train_dataloader = DataLoader(x_dataset, batch_size=batch_size, shuffle=False)
    y_test_dataloader = DataLoader(y_dataset, batch_size=batch_size, shuffle=False)
    print(len(x_train_dataloader) * batch_size)
    print(len(y_test_dataloader) * batch_size)

    # característiques generals del model
    print(model)

    # Training
    def train(epochs):
        for epoch in range(epochs):
            for i, (batch, labels) in enumerate(x_train_dataloader):
                batch, labels = batch.to(device), labels.to(device)
                labels = labels.reshape((len(labels), 1))
                labels = labels.float()

                model.zero_grad()

                output = model(batch)
                loss = criterion(output, labels)

                loss.backward()
                optimizer.step()
                if i == 0:
                    print(f'Epoch: {epoch + 1}/{epochs} Loss: {loss}')

    train(100)

    def print_metrics(dataloader):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for messages, labels in dataloader:
                messages = torch.tensor(messages).to(device)
                outputs = model(messages)
                outputs = outputs.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                outputs = outputs >= 0.5
                labels = labels == 1.0
                total += len(labels)
                for i in range(len(labels)):
                    true_i = labels[i]
                    pred_i = outputs[i][0]
                    if true_i == 1 and pred_i == 1: tp += 1
                    if true_i == 0 and pred_i == 0: tn += 1
                    if true_i == 0 and pred_i == 1: fp += 1
                    if true_i == 1 and pred_i == 0: fn += 1
                    if labels[i] == outputs[i][0]:
                        correct += 1
        print("accuracy: ", correct / total)
        print("precission: ", tp/(tp+fp))
        print("recall: ", tp/(tp+fn))
        print("f1: ", (2*tp)/(2*tp + fp + fn))

    print_metrics(x_train_dataloader)
    print_metrics(y_test_dataloader)


if __name__ == "__main__":
    main()
