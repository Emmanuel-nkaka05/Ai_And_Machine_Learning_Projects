import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

try:
    ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    ROOT = os.getcwd()

def out(filename):
    return os.path.join(ROOT, filename)

iris = load_iris()
X = iris.data
y = iris.target

df = pd.DataFrame(X, columns=iris.feature_names)
df["species"] = y
df["species_name"] = df["species"].map(dict(enumerate(iris.target_names)))
print(df.head())
print("Shape:", df.shape)
print(df["species"].value_counts())
df.to_csv(out("iris_dataset.csv"), index=False)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

k_values = range(1, 21)
error_rates = []
for k in k_values:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train_scaled, y_train)
    pred_temp = knn_temp.predict(X_test_scaled)
    error_rates.append(1 - accuracy_score(y_test, pred_temp))

plt.figure(figsize=(7, 4))
plt.plot(k_values, error_rates, marker="o")
plt.xlabel("K Value")
plt.ylabel("Error Rate")
plt.title("Choosing K (Elbow Method)")
plt.tight_layout()
plt.savefig(out("elbow_plot.png"), dpi=150)
plt.show()

best_k = k_values[error_rates.index(min(error_rates))]
print("Best K:", best_k)

model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)

acc = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="weighted")
cm = confusion_matrix(y_test, predictions)
report = classification_report(y_test, predictions, target_names=iris.target_names)

print("Accuracy:", round(acc, 3))
print("F1 Score:", round(f1, 3))
print("Confusion Matrix:\n", cm)
print(report)

pd.DataFrame(cm, index=iris.target_names, columns=iris.target_names).to_csv(out("confusion_matrix.csv"))

pd.DataFrame({
    "actual": [iris.target_names[i] for i in y_test],
    "predicted": [iris.target_names[i] for i in predictions]
}).to_csv(out("predictions.csv"), index=False)

with open(out("classification_report.txt"), "w") as f:
    f.write(report)

with open(out("metrics.txt"), "w") as f:
    f.write(f"Best K: {best_k}\n")
    f.write(f"Accuracy: {acc:.3f}\n")
    f.write(f"F1 Score (weighted): {f1:.3f}\n")

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(out("confusion_matrix.png"), dpi=150)
plt.show()

new_flower = [[5.1, 3.5, 1.4, 0.2]]
new_scaled = scaler.transform(new_flower)
result = model.predict(new_scaled)
print("Predicted species:", iris.target_names[result[0]])

print("Files saved in:", ROOT)