import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt

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

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imputer.fit_transform(X)

print("Scaling data...")
scaler = StandardScaler()
X = scaler.fit_transform(X)

from sklearn.preprocessing import LabelEncoder

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=0,
    stratify=y
)

from sklearn.svm import LinearSVC

classifier = LinearSVC(random_state=0, max_iter=1000, dual=False)

start_time = time.time()
print("Training Support Vector Machine (LinearSVC)...")
classifier.fit(X_train, y_train)

print("Predicting with Support Vector Machine...")
y_pred = classifier.predict(X_test)

time_taken = time.time() - start_time

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:\n")
print(cm)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\nAccuracy : {:.4f}".format(acc))
print("\nPrecision : {:.4f}".format(prec))
print("\nRecall : {:.4f}".format(rec))
print("\nF1 Score : {:.4f}".format(f1))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save results to CSV
results = [{
    "Model": "Support Vector Machine (60/40 Split)",
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
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=full_df.values, colLabels=full_df.columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
plt.savefig('comparison_results.pdf', format='pdf', bbox_inches='tight')
plt.close()
print("Result also exported to 'comparison_results.pdf'")