import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess():

    os.makedirs("artifacts", exist_ok=True)

    df = pd.read_csv("ingested/Heart Attack Data Set.csv")

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )

    joblib.dump(scaler, "artifacts/preprocessor.pkl")

    train_data = pd.concat(
        [X_train_scaled, y_train.reset_index(drop=True)],
        axis=1
    )

    test_data = pd.concat(
        [X_test_scaled, y_test.reset_index(drop=True)],
        axis=1
    )

    train_data.to_csv("artifacts/train.csv", index=False)
    test_data.to_csv("artifacts/test.csv", index=False)

    print("✅ Preprocessing selesai dan data disimpan di artifacts/")

if __name__ == "__main__":
    preprocess()