import numpy as np
import pandas as pd
from string import punctuation
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml.sentimentLSTM import SentimentLSTM

train_on_gpu = False

def read_dataset(name):
    df = pd.read_csv('../data/' + name + ".csv")
    df = df.sample(frac=1)
    df.columns = df.columns.str.replace(" ", "_")

    if name == "tweeter_3":
        df = df.rename({'message_to_examine': 'message', 'label_(depression_result)': 'label'}, axis=1)
        return df
    if name == "reddit_cleaned":
        df = df.rename({'clean_text': 'message', 'is_depression': 'label'}, axis=1)
        return df
    if name == "twitter_13":
        df = df.rename({'post_text': 'message'}, axis=1)
        return df
    if name == "twitter_scale":
        df = df.rename({'Text': 'message', 'Sentiment': 'label'}, axis=1)
        df.loc[df["label"] == 2, "label"] = 1
        df.loc[df["label"] == 3, "label"] = 1
        return df

    return df


def pad_features(reviews_int, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's or truncated to the input seq_length.
    '''
    features = np.zeros((len(reviews_int), seq_length), dtype=int)

    for i, review in enumerate(reviews_int):
        review_len = len(review)

        if review_len <= seq_length:
            zeroes = list(np.zeros(seq_length - review_len))
            new = zeroes + review
        elif review_len > seq_length:
            new = review[0:seq_length]

        features[i, :] = np.array(new)

    return features

def main():
    # file_name = "reddit_cleaned"
    file_name = "tweeter_3"
    # file_name = "clean_twitter_scale"

    df = read_dataset(file_name)
    # df = df[:][:100]

    message = df['message'].tolist()
    label = df['label'].tolist()

    # print(message[:5])
    for i, m in enumerate(message):
        aux = ''.join([c for c in m if c not in punctuation])
        aux = aux.lower()
        message[i] = aux
    # print()
    # print(message[:5])

    all_text2 = ' '.join(message)
    # create a list of words
    words = all_text2.split()
    # count all the words using counter
    count_words = Counter(words)
    total_words = len(words)
    sorted_words = count_words.most_common(total_words)
    # print(count_words)

    # index mapping dictionary
    vocab_to_int = {w: i + 1 for i, (w, c) in enumerate(sorted_words)}
    a = 1

    # tokenize - encode the words and labels
    reviews_int = []
    for review in message:
        r = [vocab_to_int[w] for w in review.split()]
        reviews_int.append(r)
    encoded_labels = label
    encoded_labels = np.array(encoded_labels)

    # stats
    reviews_len = [len(x) for x in reviews_int]
    pd.Series(reviews_len).hist()
    #plt.show()

    features = pad_features(reviews_int, 200)

    split_frac = 0.8
    train_x = features[0:int(split_frac * len(features))]
    train_y = encoded_labels[0:int(split_frac * len(features))]
    remaining_x = features[int(split_frac * len(features)):]
    remaining_y = encoded_labels[int(split_frac * len(features)):]
    valid_x = remaining_x[0:int(len(remaining_x) * 0.5)]
    valid_y = remaining_y[0:int(len(remaining_y) * 0.5)]
    test_x = remaining_x[int(len(remaining_x) * 0.5):]
    test_y = remaining_y[int(len(remaining_y) * 0.5):]

    # create Tensor datasets
    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    valid_data = TensorDataset(torch.from_numpy(valid_x), torch.from_numpy(valid_y))
    test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
    # dataloaders
    batch_size = 50
    # make sure to SHUFFLE your data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    # obtain one batch of training data
    dataiter = iter(train_loader)
    sample_x, sample_y = next(dataiter)
    print('Sample input size: ', sample_x.size())  # batch_size, seq_length
    print('Sample input: \n', sample_x)
    print()
    print('Sample label size: ', sample_y.size())  # batch_size
    print('Sample label: \n', sample_y)

    # Instantiate the model w/ hyperparams
    vocab_size = len(vocab_to_int) + 1  # +1 for the 0 padding
    output_size = 1
    embedding_dim = 400
    hidden_dim = 256
    n_layers = 2
    net = SentimentLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
    print(net)




    # loss and optimization functions
    lr = 0.001

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # training params

    epochs = 4  # 3-4 is approx where I noticed the validation loss stop decreasing

    counter = 0
    print_every = 100
    clip = 5  # gradient clipping

    # move model to GPU, if available
    if (train_on_gpu):
        net.cuda()

    net.train()
    # train for some number of epochs
    for e in range(epochs):
        # initialize hidden state
        h = net.init_hidden(batch_size)

        # batch loop
        for inputs, labels in train_loader:
            counter += 1

            if (train_on_gpu):
                inputs, labels = inputs.cuda(), labels.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            inputs = inputs.type(torch.LongTensor)
            output, h = net(inputs, h)

            # calculate the loss and perform backprop
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for inputs, labels in valid_loader:

                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    if (train_on_gpu):
                        inputs, labels = inputs.cuda(), labels.cuda()

                    inputs = inputs.type(torch.LongTensor)
                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output.squeeze(), labels.float())

                    val_losses.append(val_loss.item())

                net.train()
                print("Epoch: {}/{}...".format(e + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)))



if __name__ == "__main__":
    main()




