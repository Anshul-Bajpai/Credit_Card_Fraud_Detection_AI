# =====================================================================
# STEP 1: LOCAL ENVIRONMENT SETUP, DATA LOADING & EXPLORATORY ANALYSIS
# =====================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Define Dataset Path (Assuming it is in the same folder as this script)
dataset_path = 'creditcard_2023.csv' 

if not os.path.exists(dataset_path):
    print(f"[!] Error: File not found at '{dataset_path}'.")
    print("Please ensure the CSV file is in the exact same folder as this Python script.")
else:
    print("[+] Dataset found. Loading into memory...")
    df = pd.read_csv(dataset_path)
    print(f"[+] Dataset loaded successfully! Shape: {df.shape[0]:,} rows, {df.shape[1]} columns.\n")

    # 2. Structural Integrity Audit
    print("--- DATASET ARCHITECTURE AUDIT ---")
    print(df.info())
    
    missing_values = df.isnull().sum().sum()
    print(f"\n[+] Total Missing/Null Values across entire dataset: {missing_values}")

    # Display first 5 rows to verify feature types
    print("\n--- FIRST 5 RECORDS ---")
    print(df.head())

    # 3. Class Distribution Visual Analysis
    print("\n[-] Generating class distribution diagnostics...")
    class_counts = df['Class'].value_counts()
    class_percentages = df['Class'].value_counts(normalize=True) * 100
    
    print("\n--- CLASS DISTRIBUTION METRICS ---")
    for cls, count in class_counts.items():
        label = "Fraudulent (Class 1)" if cls == 1 else "Legitimate (Class 0)"
        print(f"  * {label}: {count:,} transactions ({class_percentages[cls]:.2f}%)")

    # 4. Plotting the target variable distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Class', data=df, palette='Set2')
    plt.title('Distribution of Financial Transactions (Class Imbalance Analysis)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Transaction Class (0 = Legitimate, 1 = Fraudulent)', fontsize=12, labelpad=10)
    plt.ylabel('Transaction Count', fontsize=12, labelpad=10)
    plt.xticks(ticks=[0, 1], labels=['Legitimate (0)', 'Fraudulent (1)'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save chart to local directory for project report insertion
    chart_output_path = 'class_distribution_plot.png'
    plt.savefig(chart_output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n[+] Diagnostic visualization saved successfully as '{chart_output_path}' in your current folder.")