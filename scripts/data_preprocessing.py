# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Necessary libraries imported successfully.")

# Function to handle missing values
def handle_missing_values(dataframes):
    logging.info("Handling missing values for dataframes.")
    for i, df in enumerate(dataframes):
        missing_before = df.isnull().sum().sum()
        df.fillna(method='ffill', inplace=True)
        missing_after = df.isnull().sum().sum()
        logging.info(f"Dataframe {i + 1}: Missing values before: {missing_before}, after: {missing_after}")
    return dataframes

# Function to clean data
def clean_data(fraud_data):
    logging.info("Cleaning fraud data.")
    duplicates_before = fraud_data.duplicated().sum()
    fraud_data.drop_duplicates(inplace=True)
    duplicates_after = fraud_data.duplicated().sum()
    logging.info(f"Duplicates removed: {duplicates_before - duplicates_after}")
    
    fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'])
    fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])
    logging.info("Datetime columns converted successfully.")
    return fraud_data

# Function for EDA
def perform_eda(fraud_data, creditcard_data):
    logging.info("Performing exploratory data analysis (EDA).")
    
    # Fraud Data: Purchase value distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(fraud_data['purchase_value'], kde=True, bins=30)
    plt.title('Distribution of Purchase Value')
    plt.xlabel('Purchase Value')
    plt.ylabel('Frequency')
    plt.show()
    logging.info("Plotted distribution of purchase value.")
    
    # Fraud Data: Correlation heatmap (numeric columns only)
    numeric_columns = fraud_data.select_dtypes(include=['number'])
    plt.figure(figsize=(10, 6))
    sns.heatmap(numeric_columns.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap (Numeric Columns Only)')
    plt.show()
    logging.info("Plotted correlation heatmap for numeric columns.")
    
    # Creditcard Data: Target class distribution
    sns.countplot(data=creditcard_data, x='Class')
    plt.title('Fraudulent vs Non-Fraudulent Transactions')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.show()
    logging.info("Plotted class distribution for credit card data.")

# Function to merge fraud data with IP data
def merge_fraud_with_ip(fraud_data, ip_data):
    logging.info("Merging fraud data with IP data.")
    fraud_data['ip_address_int'] = fraud_data['ip_address'].astype(int)
    merged_data = fraud_data.merge(
        ip_data, 
        left_on='ip_address_int', 
        right_on='lower_bound_ip_address', 
        how='left'
    )
    logging.info("Fraud data successfully merged with IP data.")
    return merged_data

# Function for feature engineering
def feature_engineering(fraud_data):
    logging.info("Performing feature engineering on fraud data.")
    fraud_data['transaction_count'] = fraud_data.groupby('user_id')['purchase_time'].transform('count')
    fraud_data['hour_of_day'] = fraud_data['purchase_time'].dt.hour
    fraud_data['day_of_week'] = fraud_data['purchase_time'].dt.dayofweek
    logging.info("Feature engineering completed successfully.")
    return fraud_data

# Function to encode categorical features
def encode_features(fraud_data):
    logging.info("Encoding categorical features in fraud data.")
    label_encoder = LabelEncoder()
    for column in ['sex', 'source', 'browser', 'country']:
        fraud_data[column] = label_encoder.fit_transform(fraud_data[column])
        logging.info(f"Encoded column: {column}")
    return fraud_data

# Function to scale numerical features
def scale_features(fraud_data, creditcard_data):
    logging.info("Scaling numerical features.")
    scaler = StandardScaler()
    fraud_data[['purchase_value', 'transaction_count']] = scaler.fit_transform(fraud_data[['purchase_value', 'transaction_count']])
    
    features_to_scale = [col for col in creditcard_data.columns if col not in ['Class', 'Time']]
    creditcard_data[features_to_scale] = scaler.fit_transform(creditcard_data[features_to_scale])
    logging.info("Scaling completed successfully.")
    return fraud_data, creditcard_data

# Function to prepare final datasets
def prepare_datasets(fraud_data, creditcard_data):
    logging.info("Preparing final datasets for model training.")
    X_fraud = fraud_data.drop(columns=['class', 'user_id', 'signup_time', 'purchase_time', 'ip_address', 
                                       'ip_address_int', 'lower_bound_ip_address', 'upper_bound_ip_address'])
    y_fraud = fraud_data['class']
    
    X_creditcard = creditcard_data.drop(columns=['Class'])
    y_creditcard = creditcard_data['Class']
    logging.info("Final datasets prepared successfully.")
    return X_fraud, y_fraud, X_creditcard, y_creditcard
