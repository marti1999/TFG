import pandas as pd


def main():
    filename = "../data/clean_tweeter_3.csv"
    filename = "../data/clean_twitter_scale.csv"
    # filename = "../data/clean_reddit_cleaned.csv"

    df = pd.read_csv(filename)
    df_false = df.loc[(df["label"] == 0) & (df['message'].str.contains("sad"))]

    a=12

if __name__ == "__main__":
    main()