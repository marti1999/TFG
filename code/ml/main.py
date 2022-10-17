import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

from ml.decision_tree import execute_dtc
from ml.naive_bayes import execute_nb
from ml.preprocess_text import preprocess_text, create_tfidf, create_bow

def show_score(y_test, y_pred, title="", avg='binary'):
    print("\n", title)
    print("accuracy: ", accuracy_score(y_test, y_pred))
    print("precission: ", precision_score(y_test, y_pred, average=avg))
    print("recall: ", recall_score(y_test, y_pred, average=avg))
    print("f1_score: ", f1_score(y_test, y_pred, average=avg))

def read_dataset(name):
    df = pd.read_csv('../data/'+name)
    df = df.sample(frac=1)
    df.columns = df.columns.str.replace(" ", "_")


    if name == "tweeter_3.csv":
        df = df.rename({'message_to_examine': 'message', 'label_(depression_result)': 'label'}, axis=1)
        return df
    if name == "reddit_cleaned.csv":
        df = df.rename({'clean_text': 'message', 'is_depression': 'label'}, axis=1)
        return df
    if name == "twitter_13.csv":
        df = df.rename({'post_text': 'message'}, axis=1)
        return df
    if name == "twitter_scale.csv":
        df = df.rename({'Text': 'message', 'Sentiment': 'label'}, axis=1)
        return df



if __name__ == "__main__":


    df = read_dataset("tweeter_3.csv")
    # df = read_dataset("reddit_cleaned.csv")
    # df = read_dataset("twitter_13.csv")
    # df = read_dataset("twitter_scale.csv")
    df = df[:][:1000]

    df = preprocess_text(df)

    tfidf = create_tfidf(df)
    bow = create_bow(df)

    X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(tfidf, df['label'],
                                                                                test_size=0.2, random_state=10)
    X_train_bow, X_test_bow, y_train_bow, y_test_bow = train_test_split(bow, df['label'],
                                                                        test_size=0.2, random_state=10)


    y_pred = execute_nb(X_train_tfidf, y_train_tfidf, X_test_tfidf)
    show_score(y_test_tfidf, y_pred, "NB tfidf")
    y_pred = execute_nb(X_train_bow, y_train_bow, X_test_bow)
    show_score(y_test_bow, y_pred, "NB bow")

    y_pred = execute_dtc(X_train_tfidf, y_train_tfidf, X_test_tfidf)
    show_score(y_test_tfidf, y_pred, "DT tfidf")
    y_pred = execute_dtc(X_train_bow, y_train_bow, X_test_bow)
    show_score(y_test_bow, y_pred, "DT bow")


