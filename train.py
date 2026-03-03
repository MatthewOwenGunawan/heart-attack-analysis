import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train():

    train_data = pd.read_csv("artifacts/train.csv")

    X_train = train_data.drop("target", axis=1)
    y_train = train_data["target"]

    model = RandomForestClassifier(random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_train)
    acc = accuracy_score(y_train, y_pred)

    print(f"Training Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_train, y_pred))

    # Save model
    joblib.dump(model, "artifacts/model.pkl")

    print("✅ Model berhasil disimpan di artifacts/model.pkl")

if __name__ == "__main__":
    train()