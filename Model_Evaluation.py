# =====================================================================
# STEP 3: MODEL EVALUATION AND CONFUSION MATRIX GENERATION
# =====================================================================

import os
# Hide standard TensorFlow informational logs for a cleaner terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# 1. Rapid Data Preparation (Recreating the exact test set)
print("[-] Preparing test data...")
df = pd.read_csv('creditcard_2023.csv')
if 'id' in df.columns:
    df = df.drop('id', axis=1)

scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))

X = df.drop('Class', axis=1)
y = df['Class']

# We use the exact same random_state (42) to ensure the test set matches perfectly
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Load the Trained Model
print("[-] Loading 'fraud_detection_model.h5'...")
model = tf.keras.models.load_model('fraud_detection_model.h5')

# 3. Generate Predictions
print("[-] Generating predictions on unseen test data...")
y_pred_probabilities = model.predict(X_test, verbose=0)
# Convert probabilities (0.0 to 1.0) into binary class labels (0 or 1) using a 0.5 threshold
y_pred = (y_pred_probabilities > 0.5).astype(int)

# 4. Print Classification Report
print("\n======================================================")
print("             DETAILED CLASSIFICATION REPORT             ")
print("======================================================")
print(classification_report(y_test, y_pred, target_names=['Legitimate (0)', 'Fraudulent (1)']))

# 5. Generate and Save Confusion Matrix
print("\n[-] Generating Confusion Matrix visualization...")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=0.5, annot_kws={"size": 14},
            xticklabels=['Predicted Legitimate', 'Predicted Fraud'],
            yticklabels=['Actual Legitimate', 'Actual Fraud'])

plt.title('Neural Network Confusion Matrix', fontsize=15, fontweight='bold', pad=15)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)

# Save the visualization for the report
output_img = 'confusion_matrix.png'
plt.savefig(output_img, dpi=300, bbox_inches='tight')
plt.show()

print(f"[+] Confusion matrix successfully saved as '{output_img}'.")