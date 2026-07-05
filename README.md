# Iris KNN Classifier

A K-Nearest Neighbors (KNN) classifier trained on the classic Iris flower dataset, with cross-validated hyperparameter tuning, model evaluation, and visualizations.

## 📋 Overview

This project implements a complete machine learning pipeline using the Iris dataset — a well-known benchmark dataset containing 150 samples across three species: **Setosa**, **Versicolor**, and **Virginica**.

The pipeline covers:
- Data loading and exploration
- Train/test split with feature scaling
- Cross-validated hyperparameter tuning to find the optimal K
- Model training and evaluation
- Visualization of results

## 📊 Dataset

The Iris dataset includes 150 samples with 4 features per sample:
- Sepal length
- Sepal width
- Petal length
- Petal width

Each sample is labeled with one of three Iris species.

## ⚙️ Methodology

1. **Data Exploration** — Load the dataset and check class balance and summary statistics.
2. **Train/Test Split** — Split data into 80% training and 20% testing sets.
3. **Feature Scaling** — Apply `StandardScaler` to normalize features (mean = 0, variance = 1).
4. **Hyperparameter Tuning** — Use cross-validation to test K values from 1–20, selecting the K with the lowest cross-validation error rate.
5. **Model Training** — Train the final KNN classifier using the optimal K.
6. **Evaluation** — Assess performance using accuracy, F1-score (macro-averaged), and a full classification report.
7. **Visualization** — Generate an elbow plot (for K selection) and a confusion matrix heatmap.

## 📈 Results

- **Elbow Plot** (`elbow_plot.png`) — Shows cross-validation error rate across different K values, with the chosen K highlighted.
- **Confusion Matrix** (`confusion_matrix.png`) — Visualizes prediction accuracy across the three Iris species.

## 🛠️ Tech Stack

- Python
- scikit-learn
- pandas
- matplotlib

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/Shalini-ss2306/iris-knn-classifier.git
   cd iris-knn-classifier
   ```

2. Install dependencies:
   ```bash
   pip install scikit-learn pandas matplotlib
   ```

3. Run the script:
   ```bash
   python project2.py
   ```

4. Output plots (`elbow_plot.png`, `confusion_matrix.png`) will be saved in the project folder.

## 📁 Project Structure

```
iris-knn-classifier/
├── project2.py           # Main script: data loading, training, evaluation
├── elbow_plot.png         # Cross-validation error vs. K value
├── confusion_matrix.png   # Confusion matrix heatmap
└── README.md              # Project documentation
```

## 📄 License

This project is open source and available for educational purposes.
