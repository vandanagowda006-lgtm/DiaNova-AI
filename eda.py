
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "diabetes.csv"
OUTPUT_DIR = BASE_DIR / "visualizations"

# Create visualization folder if not exists
OUTPUT_DIR.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(DATASET_PATH)

# Visualization style
sns.set_theme(style="whitegrid")

print("Dataset loaded successfully!")
print("Dataset Shape:", df.shape)


# --------------------------------
# 1. Diabetes Distribution
# --------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Diabetes_binary"
)

plt.title("Diabetes Class Distribution")
plt.xlabel("Diabetes Status (0 = No, 1 = Yes)")
plt.ylabel("Number of Records")

plt.savefig(OUTPUT_DIR / "diabetes_distribution.png")
plt.close()


# --------------------------------
# 2. Age Distribution
# --------------------------------

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="Age",
    hue="Diabetes_binary",
    multiple="stack",
    bins=13
)

plt.title("Age Distribution")
plt.xlabel("Age Category")
plt.ylabel("Number of Records")

plt.savefig(OUTPUT_DIR / "age_distribution.png")
plt.close()


# --------------------------------
# 3. BMI Distribution
# --------------------------------

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="BMI",
    hue="Diabetes_binary",
    kde=True,
    bins=30
)

plt.title("BMI Distribution")
plt.xlabel("Body Mass Index")
plt.ylabel("Frequency")

plt.savefig(OUTPUT_DIR / "bmi_distribution.png")
plt.close()


# --------------------------------
# 4. Blood Pressure Analysis
# --------------------------------

plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    x="HighBP",
    hue="Diabetes_binary"
)

plt.title("Blood Pressure vs Diabetes")
plt.xlabel("High Blood Pressure (0 = No, 1 = Yes)")
plt.ylabel("Number of Records")

plt.savefig(OUTPUT_DIR / "blood_pressure.png")
plt.close()


# --------------------------------
# 5. Cholesterol Analysis
# --------------------------------

plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    x="HighChol",
    hue="Diabetes_binary"
)

plt.title("High Cholesterol vs Diabetes")
plt.xlabel("High Cholesterol (0 = No, 1 = Yes)")
plt.ylabel("Number of Records")

plt.savefig(OUTPUT_DIR / "cholesterol.png")
plt.close()


# --------------------------------
# 6. Correlation Heatmap
# --------------------------------

plt.figure(figsize=(16, 12))

correlation = df.corr(numeric_only=True)

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Health Indicators Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "correlation_heatmap.png",
    dpi=300
)

plt.close()


# --------------------------------
# Final Output
# --------------------------------

print("\nAll visualizations generated successfully!")

print("\nSaved Visualizations:")

for file in OUTPUT_DIR.iterdir():
    print(file.name)

