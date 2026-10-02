import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import os

# 1. Load Data
print("Loading data...")
path_prefix = 'MachineLearningCVE/' if os.path.exists('MachineLearningCVE/') else ''

try:
    df1 = pd.read_csv(f'{path_prefix}Tuesday-WorkingHours.pcap_ISCX.csv', low_memory=True)
    df2 = pd.read_csv(f'{path_prefix}Wednesday-workingHours.pcap_ISCX.csv', low_memory=True)
    df3 = pd.read_csv(f'{path_prefix}Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv', low_memory=True)
except FileNotFoundError:
    print("Error: CSV files not found. Make sure you extract the 'MachineLearningCSV.zip' file first!")
    exit()

dataset = pd.concat([df1, df2, df3], ignore_index=True)
dataset.columns = dataset.columns.str.strip()

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# 2. Data Cleaning
print("Cleaning data...")
X = X.apply(pd.to_numeric, errors='coerce')
X.replace([np.inf, -np.inf], np.nan, inplace=True)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imputer.fit_transform(X)

# 3. Data Scaling (Crucial before PCA and Models!)
print("Scaling features...")
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 4. Encode Labels
print("Encoding labels...")
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# 5. Train-Test Split (60% Train, 40% Test)
print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=0, stratify=y
)

# 6. Apply PCA (Dimensionality Reduction from i.pdf)
print("Applying PCA to reduce dimensionality...")
# Keeping 10 components speeds up training while retaining important variance
pca = PCA(n_components=10) 
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
print(f"Features reduced to {X_train.shape[1]} components.")

# 7. Define Algorithms to Compare (From ML_Algorithms_Unique_1.pdf)
from sklearn.linear_model import SGDClassifier, LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

# Note: Ensemble methods are excluded to avoid the Application Control DLL block
models = {
    "1. SGD Classifier": SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, max_iter=1000, tol=1e-3, random_state=0),
    "2. Logistic Regression": LogisticRegression(max_iter=1000, random_state=0),
    "3. Ridge Classifier": RidgeClassifier(alpha=1.0),
    "4. Decision Tree": DecisionTreeClassifier(random_state=0),
    "5. Naive Bayes": GaussianNB(),
    "6. k-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "7. Support Vector Machine": LinearSVC(random_state=0, max_iter=1000, dual=False), # LinearSVC is used as SVC(kernel='rbf') is too slow for 1M rows
    "8. LDA": LinearDiscriminantAnalysis(),
    "9. QDA": QuadraticDiscriminantAnalysis(reg_param=0.1),
    "10. Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
}

# 8. Train and Compare Models
results = []

print("\nStarting Model Training and Evaluation...\n")
for name, model in models.items():
    print(f"--- Training {name} ---")
    start_time = time.time()
    
    try:
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        time_taken = time.time() - start_time
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "Time (sec)": time_taken
        })
        
        print(f"Finished {name} in {time_taken:.2f} seconds.")
    except Exception as e:
        print(f"Error training {name}: {e}")

# 9. Print and Save Final Comparison Table
print("\n================== COMPARISON RESULTS ==================")
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
print("========================================================")

# Save to CSV
results_df.to_csv('comparison_results.csv', index=False)
print("Results successfully saved to 'comparison_results.csv'")
