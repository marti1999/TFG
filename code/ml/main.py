import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

from ml.decision_tree import execute_dtc
from ml.naive_bayes import execute_nb
from ml.preprocess_text import preprocess_text, create_tfidf, create_bow

def show_score(y_test, y_pred, title=""):
    print("\n", title)
    print("accuracy: ", accuracy_score(y_test, y_pred))
    print("precission: ", precision_score(y_test, y_pred))
    print("recall: ", recall_score(y_test, y_pred))
    print("f1_score: ", f1_score(y_test, y_pred))

if __name__ == "__main__":
    df = pd.read_csv('../data/tweeter_3.csv')
    df = df.sample(frac=1)
    # print(df.describe())
    # print(df.info())
    df.columns = df.columns.str.replace(" ", "_")
    # df = df.head(100)
    df = df[:][:1000]

    df = preprocess_text(df)

    tfidf = create_tfidf(df)
    bow = create_bow(df)

    X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(tfidf, df['label_(depression_result)'],
                                                                                test_size=0.2, random_state=10)
    X_train_bow, X_test_bow, y_train_bow, y_test_bow = train_test_split(bow, df['label_(depression_result)'],
                                                                        test_size=0.2, random_state=10)


    y_pred = execute_nb(X_train_tfidf, y_train_tfidf, X_test_tfidf)
    show_score(y_test_tfidf, y_pred, "NB tfidf")
    y_pred = execute_nb(X_train_bow, y_train_bow, X_test_bow)
    show_score(y_test_bow, y_pred, "NB bow")

    y_pred = execute_dtc(X_train_tfidf, y_train_tfidf, X_test_tfidf)
    show_score(y_test_tfidf, y_pred, "DT tfidf")
    y_pred = execute_dtc(X_train_bow, y_train_bow, X_test_bow)
    show_score(y_test_bow, y_pred, "DT bow")


