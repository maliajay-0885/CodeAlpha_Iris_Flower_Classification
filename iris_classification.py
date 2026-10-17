
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "Iris.csv"

df = pd.read_csv(CSV_FILE)

print("First five rows of the dataset:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))
print(f"\nModel accuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

plt.figure()
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    xticks_rotation=45,
    cmap="Blues"
)
plt.title("Iris Flower Classification - Confusion Matrix")
plt.tight_layout()

confusion_file = BASE_DIR / "confusion_matrix.png"
plt.savefig(confusion_file, dpi=300, bbox_inches="tight")
print("\nConfusion matrix saved to:", confusion_file)
plt.show()
plt.close("all")


importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values()

fig, ax = plt.subplots(figsize=(9, 5))
importance.plot(kind="barh", ax=ax)

ax.set_title("Iris Flower Classification - Feature Importance")
ax.set_xlabel("Importance")
ax.set_ylabel("Flower Measurements")

fig.tight_layout()

importance_file = BASE_DIR / "feature_importance.png"
fig.savefig(importance_file, dpi=300, bbox_inches="tight")

print("\nFeature importance saved to:", importance_file)
plt.show()
plt.close(fig)


new_flower = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=X.columns
)

prediction = model.predict(new_flower)

print("\nNew flower measurements:")
print(new_flower)
print("Predicted species:", prediction[0])
