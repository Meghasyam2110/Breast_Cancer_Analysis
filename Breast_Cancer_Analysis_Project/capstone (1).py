# -*- coding: utf-8 -*-
"""capstone.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1E18YVD-e8Rww6MK7ojzaGPc1V_h82Ye8

**DATA EXPLORATION**
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
column_names = [
    "ID", "Diagnosis", "Radius_Mean", "Texture_Mean", "Perimeter_Mean", "Area_Mean", "Smoothness_Mean",
    "Compactness_Mean", "Concavity_Mean", "Concave_Points_Mean", "Symmetry_Mean", "Fractal_Dimension_Mean",
    "Radius_Se", "Texture_Se", "Perimeter_Se", "Area_Se", "Smoothness_Se", "Compactness_Se", "Concavity_Se",
    "Concave_Points_Se", "Symmetry_Se", "Fractal_Dimension_Se", "Radius_Worst", "Texture_Worst", "Perimeter_Worst",
    "Area_Worst", "Smoothness_Worst", "Compactness_Worst", "Concavity_Worst", "Concave_Points_Worst",
    "Symmetry_Worst", "Fractal_Dimension_Worst"
]

# Load into Pandas DataFrame
data = pd.read_csv(url, header=None, names=column_names)

# Step 2: Understand the data
print("Dataset Shape:", data.shape)
print("First 5 rows of the dataset:")
print(data.head())
print("\nDataset Info:")
print(data.info())

# Drop the "ID" column as it is not relevant
data.drop(columns=["ID"], inplace=True)

# Step 3: Check for missing values
print("\nMissing values in dataset:")
print(data.isnull().sum())

# Step 4: Explore the target variable
print("\nTarget variable distribution:")
print(data["Diagnosis"].value_counts())

sns.countplot(x="Diagnosis", data=data, palette="pastel")
plt.title("Diagnosis Distribution (Benign vs Malignant)")
plt.show()

# Step 5: Statistical summary of features
print("\nStatistical summary of features:")
print(data.describe())

# Step 6: Correlation analysis
correlation_matrix = data.iloc[:, 1:].corr()  # Exclude Diagnosis for correlation
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()

# Step 7: Pairplot for selected features
selected_features = ["Radius_Mean", "Texture_Mean", "Perimeter_Mean", "Area_Mean", "Smoothness_Mean"]
sns.pairplot(data=data, vars=selected_features, hue="Diagnosis", palette="husl")
plt.show()

# Step 8: Boxplot for selected features
plt.figure(figsize=(12, 6))
sns.boxplot(x="Diagnosis", y="Radius_Mean", data=data, palette="pastel")
plt.title("Radius Mean by Diagnosis")
plt.show()

# Step 9: Distribution of numeric features
plt.figure(figsize=(12, 6))
for feature in selected_features:
    sns.kdeplot(data[data["Diagnosis"] == "B"][feature], label=f"Benign - {feature}", shade=True)
    sns.kdeplot(data[data["Diagnosis"] == "M"][feature], label=f"Malignant - {feature}", shade=True)
plt.title("Feature Distributions for Benign and Malignant Tumors")
plt.legend()
plt.show()

"""**DATA PREPROCESSING**"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load dataset (Wisconsin Breast Cancer Dataset)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
columns = ['ID', 'Diagnosis'] + [f'Feature_{i}' for i in range(1, 31)]
data = pd.read_csv(url, header=None, names=columns)

# Convert Diagnosis to binary (M = 1, B = 0)
data['Diagnosis'] = data['Diagnosis'].map({'M': 1, 'B': 0})

# Separate features and labels
X = data.iloc[:, 2:]  # Features
y = data['Diagnosis']  # Labels

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Data preprocessing complete.")

"""**ELM MODEL**"""

import numpy as np

class ExtremeLearningMachine:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_weights = np.random.rand(self.hidden_size, self.input_size) * 2 - 1  # Random weights [-1, 1]
        self.biases = np.random.rand(self.hidden_size)  # Random biases
        self.output_weights = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, X, y):
        # Compute hidden layer output
        H = self.sigmoid(np.dot(X, self.input_weights.T) + self.biases)
        # Compute output weights (Moore-Penrose pseudo-inverse)
        self.output_weights = np.dot(np.linalg.pinv(H), y)

    def predict(self, X):
        # Compute hidden layer output
        H = self.sigmoid(np.dot(X, self.input_weights.T) + self.biases)
        # Compute predictions
        return np.dot(H, self.output_weights)

# Initialize and train the ELM
elm = ExtremeLearningMachine(input_size=X_train.shape[1], hidden_size=50)
elm.train(X_train, y_train.to_numpy().reshape(-1, 1))  # Reshape y_train for matrix multiplication

# Test the model
y_pred = elm.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int).flatten()  # Convert probabilities to binary

# Evaluate the model
from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(y_test, y_pred_binary)
print("Model Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred_binary))

"""**DATA** **VISUALISATION**"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
columns = ["ID", "Diagnosis"] + [f"Feature_{i}" for i in range(1, 31)]
data = pd.read_csv(url, header=None, names=columns)

# Drop the ID column (not useful for analysis)
data.drop("ID", axis=1, inplace=True)

# Convert Diagnosis to numeric (Malignant=1, Benign=0)
data["Diagnosis"] = data["Diagnosis"].map({"M": 1, "B": 0})

# 1. Diagnosis Distribution
sns.countplot(x="Diagnosis", data=data, palette=["skyblue", "salmon"])
plt.title("Diagnosis Distribution")
plt.xticks(ticks=[0, 1], labels=["Benign", "Malignant"])
plt.xlabel("Diagnosis")
plt.ylabel("Count")
plt.show()

# 2. Pairwise Scatterplot for First Two Features
sns.scatterplot(data=data, x="Feature_1", y="Feature_2", hue="Diagnosis", palette=["skyblue", "salmon"])
plt.title("Feature Scatterplot: Feature_1 vs Feature_2")
plt.xlabel("Feature_1")
plt.ylabel("Feature_2")
plt.legend(labels=["Benign", "Malignant"])
plt.show()

# 3. Boxplot for Feature_1 grouped by Diagnosis
sns.boxplot(x="Diagnosis", y="Feature_1", data=data, palette=["skyblue", "salmon"])
plt.title("Boxplot of Feature_1 by Diagnosis")
plt.xticks(ticks=[0, 1], labels=["Benign", "Malignant"])
plt.xlabel("Diagnosis")
plt.ylabel("Feature_1")
plt.show()

# 4. Correlation Heatmap
correlation_matrix = data.iloc[:, 1:].corr()  # Exclude Diagnosis column for feature correlation
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, cmap="coolwarm", annot=True)
plt.title("Feature Correlation Heatmap")
plt.show()

# Histograms for Feature Distributions
features_to_plot = ["Feature_1", "Feature_2", "Feature_3"]

for feature in features_to_plot:
    plt.figure(figsize=(8, 4))
    sns.histplot(data, x=feature, hue="Diagnosis", kde=True, palette=["skyblue", "salmon"], bins=30)
    plt.title(f"Distribution of {feature} by Diagnosis")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.legend(["Benign", "Malignant"])
    plt.show()

# Violin Plot for Feature_4
sns.violinplot(x="Diagnosis", y="Feature_4", data=data, palette=["skyblue", "salmon"])
plt.title("Violin Plot of Feature_4 by Diagnosis")
plt.xticks(ticks=[0, 1], labels=["Benign", "Malignant"])
plt.xlabel("Diagnosis")
plt.ylabel("Feature_4")
plt.show()

# Pair Plot for First 5 Features
sns.pairplot(data, vars=["Feature_1", "Feature_2", "Feature_3", "Feature_4", "Feature_5"], hue="Diagnosis", palette=["skyblue", "salmon"])
plt.suptitle("Pair Plot of Selected Features", y=1.02)
plt.show()


# Boxen Plot for Feature_6
sns.boxenplot(x="Diagnosis", y="Feature_6", data=data, palette=["skyblue", "salmon"])
plt.title("Boxen Plot of Feature_6 by Diagnosis")
plt.xticks(ticks=[0, 1], labels=["Benign", "Malignant"])
plt.xlabel("Diagnosis")
plt.ylabel("Feature_6")
plt.show()


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.iloc[:, 1:])

# Apply PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

# Create a DataFrame for PCA results
pca_df = pd.DataFrame(data=pca_result, columns=["PCA1", "PCA2"])
pca_df["Diagnosis"] = data["Diagnosis"]

# Scatter plot of PCA results
sns.scatterplot(x="PCA1", y="PCA2", hue="Diagnosis", data=pca_df, palette=["skyblue", "salmon"])
plt.title("PCA Scatter Plot")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(labels=["Benign", "Malignant"])
plt.show()


# Cluster Map of Correlations
plt.figure(figsize=(10, 8))
sns.clustermap(data.iloc[:, 1:].corr(), cmap="coolwarm", method="ward", figsize=(12, 10))
plt.title("Cluster Map of Feature Correlations")
plt.show()

from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.2f}')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()