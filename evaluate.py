import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate():

    test_data = pd.read_csv("artifacts/test.csv")

    X_test = test_data.drop("target", axis=1)
    y_test = test_data["target"]

    model = joblib.load("artifacts/model.pkl")

    y_pred = model.predict(X_test)

    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    evaluate()