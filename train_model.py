
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# --------------------------------
# 1. Project Paths
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "dataset" / "diabetes.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)


# --------------------------------
# 2. Load Dataset
# --------------------------------

print("Loading dataset...")

df = pd.read_csv(DATASET_PATH)

print("Dataset Shape:", df.shape)


# --------------------------------
# 3. Data Preprocessing
# --------------------------------

print("\nChecking missing values...")

print("Total Missing Values:", df.isnull().sum().sum())

# Separate features and target
X = df.drop("Diabetes_binary", axis=1)

y = df["Diabetes_binary"].astype(int)

print("\nFeatures:", X.shape)
print("Target:", y.shape)


# --------------------------------
# 4. Train-Test Split
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", X_train.shape[0])
print("Testing Records:", X_test.shape[0])


# --------------------------------
# 5. Define ML Models
# --------------------------------

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=12,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
}


# --------------------------------
# 6. Train and Evaluate Models
# --------------------------------

results = []

best_model = None
best_model_name = ""
best_f1 = -1

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    roc_auc = roc_auc_score(y_test, probabilities)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": roc_auc
    })

    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC AUC:", round(roc_auc, 4))

    # Select model using F1 score
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name


# --------------------------------
# 7. Compare Model Performance
# --------------------------------

results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")

print(results_df.to_string(index=False))


# --------------------------------
# 8. Save Best Model
# --------------------------------

model_path = MODEL_DIR / "best_model.pkl"

joblib.dump(best_model, model_path)

print("\nBest Model:", best_model_name)

print("Best F1 Score:", round(best_f1, 4))

print("\nModel saved successfully at:")
print(model_path)

