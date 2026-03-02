import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess():
    df = pd.read_csv("ingested/Heart Attack Data Set.csv")
    categorical_cols = ['cp', 'restecg', 'slope', 'ca', 'thal']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    train.to_csv("train.csv", index=False)
    test.to_csv("test.csv", index=False)

if __name__ == "__main__":
    preprocess()