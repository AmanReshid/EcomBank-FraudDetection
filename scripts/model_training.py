# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Functions
def split_data(X, y, test_size=0.2, random_state=42):
    """Splits the dataset into training and testing sets."""
    logger.info("Splitting the dataset into training and testing sets")
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def preprocess_data(X):
    """Encodes categorical features in the dataset."""
    logger.info("Encoding categorical features")
    X_encoded = X.copy()
    for col in X_encoded.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col])
        logger.info(f"Encoded column: {col}")
    return X_encoded


def preprocess_and_scale_data(X):
    """Encodes categorical features and scales the data."""
    logger.info("Preprocessing and scaling data")
    
    # Copy the dataset to avoid modifying the original
    X_processed = X.copy()
    
    # Encode categorical features
    for col in X_processed.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X_processed[col] = le.fit_transform(X_processed[col])
        logger.info(f"Encoded column: {col}")
    
    # Scale numeric data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_processed)
    
    return X_scaled



def balance_data(X_train, y_train):
    """Balances the dataset using SMOTE after encoding categorical features."""
    logger.info("Balancing the dataset using SMOTE")
    
    # Preprocess the data to ensure it is numeric
    X_train_processed = preprocess_data(X_train)
    
    # Ensure y_train is a 1D numeric array
    if isinstance(y_train, pd.DataFrame) or isinstance(y_train, pd.Series):
        y_train = y_train.values  # Convert to NumPy array
    if len(y_train.shape) > 1:
        y_train = y_train.flatten()  # Flatten to 1D array
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
    
    logger.info(f"Before Resampling: {np.bincount(y_train)}")
    logger.info(f"After Resampling: {np.bincount(y_train_resampled)}")
    
    return X_train_resampled, y_train_resampled



def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    """Trains and evaluates a model."""
    logger.info(f"Training and evaluating model: {model.__class__.__name__}")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    logger.info(f"Model: {model.__class__.__name__} | Accuracy: {accuracy:.2f} | AUC: {auc:.2f}")
    logger.info("Classification Report:\n" + classification_report(y_test, predictions))
    return model, accuracy, auc

def evaluate_model_with_cv(model, X_train, y_train, cv=5):
    """Performs cross-validation on a model."""
    logger.info(f"Performing cross-validation for model: {model.__class__.__name__}")
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    logger.info(f"Cross-Validation AUC: {scores.mean():.2f} ± {scores.std():.2f}")
    return scores

def compare_models(X_train, X_test, y_train, y_test):
    """Compares multiple models and evaluates their performance."""
    logger.info("Comparing multiple models")
    models = [
        LogisticRegression(max_iter=500, random_state=42, class_weight="balanced"),
        DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
        GradientBoostingClassifier(random_state=42),
        MLPClassifier(max_iter=500, random_state=42)
    ]
    results = []
    for model in models:
        evaluate_model_with_cv(model, X_train, y_train)
        trained_model, accuracy, auc = train_and_evaluate_model(model, X_train, X_test, y_train, y_test)
        results.append((model.__class__.__name__, accuracy, auc))
    return results

def plot_model_results(results, dataset_name):
    """Visualizes model results."""
    logger.info(f"Visualizing model results for {dataset_name}")
    model_names = [result[0] for result in results]
    accuracies = [result[1] for result in results]
    aucs = [result[2] for result in results]

    # Accuracy plot
    plt.figure(figsize=(10, 5))
    plt.barh(model_names, accuracies, color='skyblue')
    plt.title(f'{dataset_name}: Model Accuracies')
    plt.xlabel('Accuracy')
    plt.show()

    # AUC plot
    plt.figure(figsize=(10, 5))
    plt.barh(model_names, aucs, color='lightgreen')
    plt.title(f'{dataset_name}: Model AUC Scores')
    plt.xlabel('AUC')
    plt.show()
