# =====================================================================
# STEP 2: PREPROCESSING AND NEURAL NETWORK TRAINING
# =====================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import warnings
warnings.filterwarnings('ignore') # Clean up terminal output

# 1. Load Dataset
print("[-] Loading dataset into memory...")
df = pd.read_csv('creditcard_2023.csv')

# Drop the 'id' column as it has no predictive value for behavior
if 'id' in df.columns:
    df = df.drop('id', axis=1)

# 2. Preprocessing: Scaling the 'Amount' feature
print("[-] Scaling transaction amounts...")
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))

# 3. Splitting the Data (Features vs. Target)
print("[-] Splitting data into Training and Testing sets (80/20)...")
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"[+] Training Set: {X_train.shape[0]:,} samples")
print(f"[+] Testing Set: {X_test.shape[0]:,} samples")

# 4. Neural Network Architecture Definition
print("\n[-] Compiling Neural Network Architecture...")
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3), # Prevents overfitting by randomly turning off 30% of neurons
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid') # Sigmoid outputs a probability between 0 and 1 (Fraud vs Legitimate)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("[+] Model compiled successfully.")

# 5. Training the Model
print("\n[-] Initiating Model Training (This may take a few minutes)...")
# We use a small number of epochs (5) and a larger batch size (512) for speed on local hardware
history = model.fit(
    X_train, y_train, 
    epochs=5, 
    batch_size=512, 
    validation_split=0.1, 
    verbose=1
)

# 6. Evaluation on Unseen Test Data
print("\n[-] Evaluating model on test data...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print("\n========================================")
print(f" FINAL TEST ACCURACY: {test_accuracy * 100:.2f}%")
print(f" FINAL TEST LOSS: {test_loss:.4f}")
print("========================================")

# Save the trained model for our live demo later
model.save('fraud_detection_model.h5')
print("\n[+] Model saved successfully as 'fraud_detection_model.h5'")