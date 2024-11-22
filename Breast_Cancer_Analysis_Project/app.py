import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Custom Extreme Learning Machine Class
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

# Streamlit App
st.title("Breast Cancer Analysis using Extreme Learning Machine")

# File upload section
uploaded_file = st.file_uploader("Upload CSV File (with 30 features)", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.dataframe(data.head())

    # Assume the last column is the target (Diagnosis: 0 = Benign, 1 = Malignant)
    X = data.iloc[:, :-1]  # Features
    y = data.iloc[:, -1]  # Target

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train the ELM model
    elm = ExtremeLearningMachine(input_size=X_train.shape[1], hidden_size=50)
    elm.train(X_train, y_train.to_numpy().reshape(-1, 1))  # Reshape target for training

    # Predict on the test set
    y_pred = elm.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int).flatten()

    # Display results
    from sklearn.metrics import accuracy_score, classification_report
    accuracy = accuracy_score(y_test, y_pred_binary)
    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred_binary))

    # Allow prediction on new data
    st.write("Predict on New Data:")
    new_data = st.file_uploader("Upload New Data for Prediction (without target column)", type=["csv"])
    if new_data is not None:
        new_features = pd.read_csv(new_data)
        new_features_scaled = scaler.transform(new_features)
        new_predictions = elm.predict(new_features_scaled)
        new_predictions_binary = (new_predictions > 0.5).astype(int).flatten()
        st.write("Predictions (0 = Benign, 1 = Malignant):")
        new_features['Prediction'] = new_predictions_binary
        st.dataframe(new_features)


