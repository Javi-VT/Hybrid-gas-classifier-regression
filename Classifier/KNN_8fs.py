import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ------------------- Load and Prepare Data -------------------
df = pd.read_csv('processed_gas_sensor_data.csv')

# Fill NaNs
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)

# Select features with value 1
selected_features = [
    'TFSI_norm', 'NCN_norm', 'Fe_norm', 'sum_norm',
    'ratio_TFSI', 'ratio_NCN', 'ratio_Ni', 'ratio_Fe',
    'deriv_TFSI', 'deriv_NCN', 'deriv_Ni', 'deriv_Fe',
    'roll_mean_TFSI', 'roll_std_TFSI',
    'roll_mean_NCN', 'roll_std_NCN',
    'roll_mean_Ni', 'roll_std_Ni',
    'roll_mean_Fe', 'roll_std_Fe',
    'cross_corr_TFSI_NCN', 'cross_corr_TFSI_Ni', 'cross_corr_TFSI_Fe',
    'cross_corr_NCN_Ni', 'cross_corr_NCN_Fe', 'cross_corr_Ni_Fe'
]

X_selected = df[selected_features].values
X_full = df.drop(columns=['Predictor']).values  # full feature set
full_feature_names = df.drop(columns=['Predictor']).columns.tolist()
y = df['Predictor'].values

# ------------------- Train/Test Split -------------------
X_train_sel, X_test_sel, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42, shuffle=True)
X_train_full, X_test_full, _, _ = train_test_split(X_full, y, test_size=0.2, random_state=42, shuffle=True)

# ------------------- Define and Train Classifier -------------------
knn = KNeighborsClassifier(
    n_neighbors=8,
    metric='euclidean',
    weights='uniform'
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_val_pred = cross_val_predict(knn, X_train_sel, y_train, cv=cv)

knn.fit(X_train_sel, y_train)
y_test_pred = knn.predict(X_test_sel)

# ------------------- Evaluation Functions -------------------
def compute_tpr_fnr(cm):
    tpr = np.diag(cm) / np.sum(cm, axis=1)
    fnr = 1 - tpr
    return tpr * 100, fnr * 100

# ------------------- Validation Results -------------------
print("\n[Validation - 5-Fold Cross-Validation]")
val_cm = confusion_matrix(y_train, y_val_pred)
val_tpr, val_fnr = compute_tpr_fnr(val_cm)

print("Validation Confusion Matrix:\n", val_cm)
print("\nValidation TPR (%):", np.round(val_tpr, 2))
print("Validation FNR (%):", np.round(val_fnr, 2))

plt.figure(figsize=(8, 6))
sns.heatmap(val_cm, annot=True, fmt='d', cmap='Purples')
plt.title('Validation Confusion Matrix (5-Fold CV)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('validation_confusion_matrix.png')
plt.close()

# ------------------- Test Results -------------------
print("\n[Test Set Evaluation]")
test_cm = confusion_matrix(y_test, y_test_pred)
test_tpr, test_fnr = compute_tpr_fnr(test_cm)

print("Test Set Classification Report:")
print(classification_report(y_test, y_test_pred))

print("Test Confusion Matrix:\n", test_cm)
print("\nTest TPR (%):", np.round(test_tpr, 2))
print("Test FNR (%):", np.round(test_fnr, 2))

plt.figure(figsize=(8, 6))
sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Test Set Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('test_confusion_matrix.png')
plt.close()

# Train a new KNN model on the full feature set
new_knn = KNeighborsClassifier(
    n_neighbors=8,
    metric='euclidean',
    weights='uniform'
)
new_knn.fit(X_train_full, y_train)