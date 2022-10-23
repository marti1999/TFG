# from line_profile_pycharm import profile

import pandas as pd
from pandas_profiling import ProfileReport


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, \
    mean_squared_error
from ml.ml_models import execute_nb, execute_dtc, execute_rf, execute_svm, execute_knn
from ml.plots import bar_plot_multiple_column, save_results_to_csv
from ml.preprocess_text import preprocess_text, create_tfidf, create_bow, save_df_to_csv


class Metrics:
    def __init__(self):
        self.name = ''
        self.acc = []
        self.prec = []
        self.recall = []
        self.f1 = []


def show_score(y_test, y_pred, metric_list, title="", avg='binary', model_name='', params='', dataset_name='', type='',
               seconds=0):
    print(f"\n{title}: {seconds:.03f} seconds")
    if avg == 'mse':
        print("mean error (no squared): ", mean_squared_error(y_test, y_pred, squared=False))
    else:
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        metric_list.acc.append(acc)
        metric_list.prec.append(prec)
        metric_list.recall.append(recall)
        metric_list.f1.append(f1)
        print("accuracy: ", acc)
        print("precission: ", prec)
        print("recall: ", recall)
        print("f1_score: ", f1)
        save_results_to_csv(model_name, params, dataset_name, type, avg, acc, prec, recall, f1, seconds)


def read_dataset(name):
    df = pd.read_csv('../data/' + name + ".csv")
    df = df.sample(frac=1)
    df.columns = df.columns.str.replace(" ", "_")

    if name == "tweeter_3":
        df = df.rename({'message_to_examine': 'processed', 'label_(depression_result)': 'label'}, axis=1)
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


if __name__ == "__main__":
    # file_name = "clean_tweeter_3"
    # file_name = "clean_reddit_cleaned"
    # file_name = "clean_twitter_13"
    file_name = "clean_twitter_scale"

    df = read_dataset(file_name)
    # df = read_dataset(file_name)
    # df = read_dataset(file_name)
    # df = read_dataset(file_name)
    # df = df[:][:300]

    rp = ProfileReport(df)
    rp.to_file(output_file="../results/eda/" + file_name+".html")

    # df = preprocess_text(df)
    # save_df_to_csv(df, '../data/clean_twitter_scale.csv')
    max_features = None
    tfidf = create_tfidf(df, max_features=max_features)
    bow = create_bow(df, max_features=max_features)

    X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(tfidf, df['label'],
                                                                                test_size=0.2, random_state=10)
    X_train_bow, X_test_bow, y_train_bow, y_test_bow = train_test_split(bow, df['label'],
                                                                        test_size=0.2, random_state=10)

    models = ['NB', 'DT', 'RF', 'SVM', 'KNN']
    bow_score = Metrics()
    tfidf_score = Metrics()

    y_pred, params, elapsed = execute_nb(X_train_tfidf, y_train_tfidf, X_test_tfidf)
    show_score(y_test_tfidf, y_pred, title="NB tfidf", metric_list=tfidf_score, model_name='NB', params=params,
               dataset_name=file_name, type='tfidf max_features=' + str(max_features), seconds=elapsed)
    y_pred, params, elapsed = execute_nb(X_train_bow, y_train_bow, X_test_bow)
    show_score(y_test_bow, y_pred, title="NB bow", metric_list=bow_score, model_name='NB', params=params,
               dataset_name=file_name, type='bow max_features=' + str(max_features), seconds=elapsed)

    y_pred, params, elapsed = execute_dtc(X_train_tfidf, y_train_tfidf, X_test_tfidf)
    show_score(y_test_tfidf, y_pred, title="DT tfidf", metric_list=tfidf_score, model_name='DT', params=params,
               dataset_name=file_name, type='tfidf max_features=' + str(max_features), seconds=elapsed)
    y_pred, params, elapsed = execute_dtc(X_train_bow, y_train_bow, X_test_bow)
    show_score(y_test_bow, y_pred, title="DT bow", metric_list=bow_score, model_name='DT', params=params,
               dataset_name=file_name, type='bow max_features=' + str(max_features), seconds=elapsed)

    y_pred, params, elapsed = execute_rf(X_train_tfidf, y_train_tfidf, X_test_tfidf)
    show_score(y_test_tfidf, y_pred, title="RF tfidf", metric_list=tfidf_score, model_name='RF', params=params,
               dataset_name=file_name, type='tfidf max_features=' + str(max_features), seconds=elapsed)
    y_pred, params, elapsed = execute_rf(X_train_bow, y_train_bow, X_test_bow)
    show_score(y_test_bow, y_pred, title="RF bow", metric_list=bow_score, model_name='RF', params=params,
               dataset_name=file_name, type='bow max_features=' + str(max_features), seconds=elapsed)

    y_pred, params, elapsed = execute_svm(X_train_tfidf, y_train_tfidf, X_test_tfidf)
    show_score(y_test_tfidf, y_pred, title="SVM tfidf", metric_list=tfidf_score, model_name='SVM', params=params,
               dataset_name=file_name, type='tfidf max_features=' + str(max_features), seconds=elapsed)
    y_pred, params, elapsed = execute_svm(X_train_bow, y_train_bow, X_test_bow)
    show_score(y_test_bow, y_pred, title="SVM bow", metric_list=bow_score, model_name='SVM', params=params,
               dataset_name=file_name, type='bow max_features=' + str(max_features), seconds=elapsed)

    y_pred, params, elapsed = execute_knn(X_train_tfidf, y_train_tfidf, X_test_tfidf)
    show_score(y_test_tfidf, y_pred, title="KNN tfidf", metric_list=tfidf_score, model_name='KNN', params=params,
               dataset_name=file_name, type='tfidf max_features=' + str(max_features), seconds=elapsed)
    y_pred, params, elapsed = execute_knn(X_train_bow, y_train_bow, X_test_bow)
    show_score(y_test_bow, y_pred, title="KNN bow", metric_list=bow_score, model_name='KNN', params=params,
               dataset_name=file_name, type='bow max_features=' + str(max_features), seconds=elapsed)

    bar_plot_multiple_column(models, bow_score, tfidf_score, "bow max_features=" + str(max_features),
                             'tfidf max_features=' + str(max_features),
                             file_name + "_max_features=" + str(max_features))
