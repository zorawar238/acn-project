import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import SGDClassifier, LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

# 1. Load Data
print("Loading data...")
path_prefix = 'MachineLearningCVE/' if os.path.exists('MachineLearningCVE/') else ''

try:
    df1 = pd.read_csv(f'{path_prefix}Tuesday-WorkingHours.pcap_ISCX.csv', low_memory=True)
    df2 = pd.read_csv(f'{path_prefix}Wednesday-workingHours.pcap_ISCX.csv', low_memory=True)
    df3 = pd.read_csv(f'{path_prefix}Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv', low_memory=True)
    dataset = pd.concat([df1, df2, df3], ignore_index=True)
    dataset.columns = dataset.columns.str.strip()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# 2. Data Cleaning
print("Cleaning data...")
X = X.apply(pd.to_numeric, errors='coerce')
X.replace([np.inf, -np.inf], np.nan, inplace=True)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imputer.fit_transform(X)

# 3. Data Scaling
print("Scaling features...")
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 4. Encode Labels
print("Encoding labels...")
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# 5. Define Models
models = {
    "SGD": SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, max_iter=1000, tol=1e-3, random_state=0),
    "Logistic Reg": LogisticRegression(max_iter=1000, random_state=0),
    "Ridge": RidgeClassifier(alpha=1.0),
    "Decision Tree": DecisionTreeClassifier(random_state=0),
    "Naive Bayes": GaussianNB(),
    "k-NN": KNeighborsClassifier(n_neighbors=5),
    "LinearSVC": LinearSVC(random_state=0, max_iter=1000, dual=False)
}

results = []

# Prepare datasets for both splits
print("Preparing 60/40 Split...")
X_train_60, X_test_60, y_train_60, y_test_60 = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
print("Preparing 20/80 Split...")
X_train_20, X_test_20, y_train_20, y_test_20 = train_test_split(X, y, test_size=0.8, random_state=0, stratify=y)

print("Applying PCA...")
pca60 = PCA(n_components=10)
X_train_60 = pca60.fit_transform(X_train_60)
X_test_60 = pca60.transform(X_test_60)

pca20 = PCA(n_components=10)
X_train_20 = pca20.fit_transform(X_train_20)
X_test_20 = pca20.transform(X_test_20)

print("\nStarting Model Training and Evaluation for BOTH splits...\n")

def evaluate_model(model, X_tr, y_tr, X_te, y_te, model_name, split_name):
    print(f"--- Training {model_name} on {split_name} ---")
    start_time = time.time()
    try:
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        prec = precision_score(y_te, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_te, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_te, y_pred, average='weighted', zero_division=0)
        time_taken = time.time() - start_time
        print(f"Finished {model_name} ({split_name}) in {time_taken:.2f} seconds.")
        return {"Model": model_name, "Split": split_name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1, "Time (sec)": time_taken}
    except Exception as e:
        print(f"Error with {model_name}: {e}")
        return None

for name, model in models.items():
    # 60/40 Split
    res_60 = evaluate_model(model, X_train_60, y_train_60, X_test_60, y_test_60, name, "60/40")
    if res_60: results.append(res_60)
    
    # 20/80 Split
    res_20 = evaluate_model(model, X_train_20, y_train_20, X_test_20, y_test_20, name, "20/80")
    if res_20: results.append(res_20)

results_df = pd.DataFrame(results)
results_df.to_csv("all_models_both_splits.csv", index=False)

# Create Wide Format for comparison (Side by Side)
wide_df = results_df.pivot(index="Model", columns="Split", values=["Accuracy", "Time (sec)"])
wide_df.columns = [f"{col[0]} ({col[1]})" for col in wide_df.columns]
wide_df.reset_index(inplace=True)

wide_df.to_csv("comparison_table_wide.csv", index=False)
print("\nResults successfully saved to 'comparison_table_wide.csv'")

# Generate PDF with side-by-side comparison
try:
    fig, ax = plt.subplots(figsize=(16, max(4, len(wide_df)*0.8)))
    ax.axis('tight')
    ax.axis('off')
    display_df = wide_df.copy()
    for col in display_df.columns:
        if col != "Model":
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) and isinstance(x, (int, float)) else x)
            
    table = ax.table(cellText=display_df.values, colLabels=display_df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    plt.title("Algorithm Comparison: 60/40 vs 20/80 Split", y=1.05, fontsize=16, weight='bold')
    plt.savefig('comparison_table_wide.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print("Exported side-by-side comparison to 'comparison_table_wide.pdf'")
except Exception as e:
    print(f"Failed PDF: {e}")
