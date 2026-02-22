import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# -----------------------------
# Step 1: Load dataset
# -----------------------------
data = pd.read_csv(r"C:\Users\alina\OneDrive\Desktop\MINI\safety_dataset.csv")  # adjust path

# -----------------------------
# Step 2: Prepare features and binary target
# -----------------------------
X = data.drop("safety_score", axis=1)  # all features
y_binary = (data["safety_score"] >= 50).astype(int)  # 1 = SAFE, 0 = UNSAFE

# -----------------------------
# Step 3: Split data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42
)

# -----------------------------
# Step 4: Train Logistic Regression
# -----------------------------
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred_prob = lr_model.predict_proba(X_test)[:, 1]
lr_pred_label = lr_model.predict(X_test)

print("----- Logistic Regression -----")
print("Accuracy:", round(accuracy_score(y_test, lr_pred_label), 3))
print("ROC AUC:", round(roc_auc_score(y_test, lr_pred_prob), 3))
print()

# -----------------------------
# Step 5: Train Gradient Boosting Classifier
# -----------------------------
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)
gb_model.fit(X_train, y_train)
gb_pred_prob = gb_model.predict_proba(X_test)[:, 1]
gb_pred_label = gb_model.predict(X_test)

print("----- Gradient Boosting Classifier -----")
print("Accuracy:", round(accuracy_score(y_test, gb_pred_label), 3))
print("ROC AUC:", round(roc_auc_score(y_test, gb_pred_prob), 3))

# -----------------------------
# Step 6: Save the trained Gradient Boosting Classifier
# -----------------------------
joblib.dump(gb_model, "safety_model.pkl")
print("Gradient Boosting Classifier saved as safety_model.pkl")