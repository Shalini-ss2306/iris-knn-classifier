import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    ConfusionMatrixDisplay,
)
RANDOM_STATE = 42 
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = iris.target
    df["species_name"] = df["species"].map(dict(enumerate(iris.target_names)))
    print("=" * 60)
    print("STEP 1: DATASET OVERVIEW")
    print("=" * 60)
    print(f"Samples: {df.shape[0]}  |  Features: {iris.data.shape[1]}  |  Classes: {len(iris.target_names)}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nClass balance:")
    print(df["species_name"].value_counts())
    print("\nStatistical summary:")
    print(df.describe())

    return iris, df
def prepare_data(iris):
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y, shuffle=True
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n" + "=" * 60)
    print("STEP 2: TRAIN/TEST SPLIT & SCALING")
    print("=" * 60)
    print(f"Training samples: {X_train.shape[0]}  |  Testing samples: {X_test.shape[0]}")
    print("Feature scaling applied: StandardScaler (mean=0, variance=1)")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def find_best_k(X_train, y_train, k_range=range(1, 21), cv=5):
    
    error_rates = []
    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        error_rates.append(1 - scores.mean())

    k_list = list(k_range)
    candidates = [(k, e) for k, e in zip(k_list, error_rates) if k > 1]
    best_k = min(candidates, key=lambda pair: pair[1])[0]

    plt.figure(figsize=(8, 5))
    plt.plot(k_list, error_rates, marker="o", linestyle="--", color="steelblue")
    plt.axvline(best_k, color="orangered", linestyle=":", label=f"Chosen K = {best_k}")
    plt.title("Choosing K: Cross-Validated Error Rate vs. K Value")
    plt.xlabel("K Value")
    plt.ylabel("Cross-Validation Error Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig("elbow_plot.png", dpi=150)
    plt.close()

    print("\n" + "=" * 60)
    print("STEP 3: TUNING K (CROSS-VALIDATED ELBOW METHOD)")
    print("=" * 60)
    for k, e in zip(k_list, error_rates):
        marker = "  <-- chosen" if k == best_k else ""
        print(f"  K={k:2d}  CV error={e:.4f}{marker}")
    print(f"\nBest K selected: {best_k}")

    return best_k

def train_model(X_train, y_train, k):
    model = KNeighborsClassifier(n_neighbors=k)   
    model.fit(X_train, y_train)                   
    return model

def evaluate_model(model, X_test, y_test, target_names):
    predictions = model.predict(X_test)             # PREDICT

    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="macro")
    cm = confusion_matrix(y_test, predictions)

    print("\n" + "=" * 60)
    print("STEP 5: OUTPUT VALIDATION")
    print("=" * 60)
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Score (macro): {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nFull Classification Report:")
    print(classification_report(y_test, predictions, target_names=target_names))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix - Iris KNN Classifier")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()

    return predictions, acc, f1, cm

def main():
    iris, df = load_data()
    X_train, X_test, y_train, y_test, scaler = prepare_data(iris)
    best_k = find_best_k(X_train, y_train)
    model = train_model(X_train, y_train, best_k)
    evaluate_model(model, X_test, y_test, iris.target_names)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE - Project 2 requirements satisfied:")
    print(" [x] Loaded and understood a dataset")
    print(" [x] Split data into training and testing sets")
    print(" [x] Applied a simple classification algorithm (KNN)")
    print("=" * 60)


if __name__ == "__main__":
    main()
