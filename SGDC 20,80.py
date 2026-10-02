import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

path_prefix = 'MachineLearningCVE/' if os.path.exists('MachineLearningCVE/') else ''

df1 = pd.read_csv(f'{path_prefix}Tuesday-WorkingHours.pcap_ISCX.csv', low_memory=True)
df2 = pd.read_csv(f'{path_prefix}Wednesday-workingHours.pcap_ISCX.csv', low_memory=True)
df3 = pd.read_csv(f'{path_prefix}Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv', low_memory=True)

dataset = pd.concat([df1, df2, df3], ignore_index=True)
dataset.columns = dataset.columns.str.strip()

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

X = X.apply(pd.to_numeric, errors='coerce')
X.replace([np.inf, -np.inf], np.nan, inplace=True)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imputer.fit_transform(X)

print("Scaling data...")
scaler = StandardScaler()
X = scaler.fit_transform(X)

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting dataset (20% Train, 80% Test)
print("Splitting data (20% Train, 80% Test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.8,
    random_state=0,
    stratify=y
)

classifier = LinearSVC(random_state=0, max_iter=1000, dual=False)

start_time = time.time()
print("Training Support Vector Machine (LinearSVC)...")
classifier.fit(X_train, y_train)

print("Predicting with Support Vector Machine...")
y_pred = classifier.predict(X_test)

time_taken = time.time() - start_time

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n")
print(cm)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print("\nAccuracy : {:.4f}".format(acc))
print("\nPrecision : {:.4f}".format(prec))
print("\nRecall : {:.4f}".format(rec))
print("\nF1 Score : {:.4f}".format(f1))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))

# Save results to CSV
results = [{
    "Model": "Support Vector Machine (20/80 Split)",
    "Accuracy": acc,
    "Precision": prec,
    "Recall": rec,
    "F1-Score": f1,
    "Time (sec)": time_taken
}]

results_df = pd.DataFrame(results)
file_name = 'comparison_results.csv'
if not os.path.exists(file_name):
    results_df.to_csv(file_name, index=False)
else:
    results_df.to_csv(file_name, mode='a', header=False, index=False)

print(f"\nResult saved to '{file_name}'")

# Generate PDF from the updated CSV
full_df = pd.read_csv(file_name)
fig, ax = plt.subplots(figsize=(14, max(4, len(full_df)*0.5))) # scale height by number of rows
ax.axis('tight')
ax.axis('off')

# Format columns for display
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
print("Result also exported to 'comparison_results.pdf'")
