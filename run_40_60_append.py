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

print("Loading data...")
path_prefix = 'MachineLearningCVE/' if os.path.exists('MachineLearningCVE/') else ''
df1 = pd.read_csv(f'{path_prefix}Tuesday-WorkingHours.pcap_ISCX.csv', low_memory=True)
df2 = pd.read_csv(f'{path_prefix}Wednesday-workingHours.pcap_ISCX.csv', low_memory=True)
df3 = pd.read_csv(f'{path_prefix}Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv', low_memory=True)
dataset = pd.concat([df1, df2, df3], ignore_index=True)
dataset.columns = dataset.columns.str.strip()

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

print("Cleaning data...")
X = X.apply(pd.to_numeric, errors='coerce')
X.replace([np.inf, -np.inf], np.nan, inplace=True)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imputer.fit_transform(X)

print("Scaling features...")
scaler = StandardScaler()
X = scaler.fit_transform(X)

print("Encoding labels...")
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# 40/60 Split (40% Train, 60% Test)
print("Splitting dataset (40% Train, 60% Test)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=0, stratify=y)

print("Applying PCA...")
pca = PCA(n_components=10)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

models = {
    "SGD Classifier (40/60 Split)": SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, max_iter=1000, tol=1e-3, random_state=0),
    "Logistic Regression (40/60 Split)": LogisticRegression(max_iter=1000, random_state=0),
    "Ridge Classifier (40/60 Split)": RidgeClassifier(alpha=1.0),
    "Decision Tree (40/60 Split)": DecisionTreeClassifier(random_state=0),
    "Naive Bayes (40/60 Split)": GaussianNB(),
    "k-Nearest Neighbors (40/60 Split)": KNeighborsClassifier(n_neighbors=5),
    "Support Vector Machine (40/60 Split)": LinearSVC(random_state=0, max_iter=1000, dual=False)
}

print("\nStarting Model Training and Evaluation...\n")
file_name = 'comparison_results.csv'

for name, model in models.items():
    print(f"--- Training {name} ---")
    start_time = time.time()
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        time_taken = time.time() - start_time
        
        # Append immediately so we don't lose data if it crashes
        res = pd.DataFrame([{
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "Time (sec)": time_taken
        }])
        
        if not os.path.exists(file_name):
            res.to_csv(file_name, index=False)
        else:
            res.to_csv(file_name, mode='a', header=False, index=False)
            
        print(f"Finished {name} in {time_taken:.2f} seconds. Result appended to CSV.")
    except Exception as e:
        print(f"Error training {name}: {e}")

# Generate PDF from the updated CSV
print("Generating final PDF...")
full_df = pd.read_csv(file_name)
fig, ax = plt.subplots(figsize=(14, max(4, len(full_df)*0.5)))
ax.axis('tight')
ax.axis('off')

display_df = full_df.copy()
for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
    if col in display_df.columns:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) and isinstance(x, (int, float)) else x)
if 'Time (sec)' in display_df.columns:
    display_df['Time (sec)'] = display_df['Time (sec)'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) and isinstance(x, (int, float)) else x)

table = ax.table(cellText=display_df.values, colLabels=display_df.columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)
plt.savefig('comparison_results.pdf', format='pdf', bbox_inches='tight')
plt.close()
print("Result successfully exported to 'comparison_results.pdf'")
