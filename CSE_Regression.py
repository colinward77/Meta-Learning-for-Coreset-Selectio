import pandas as pd
import numpy as np
import os

# For data splitting and model training
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # Updated import

# For model evaluation
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

# For data scaling and encoding
from sklearn.preprocessing import StandardScaler

# For plotting
import matplotlib.pyplot as plt
import seaborn as sns

# For clustering-based sampling
from sklearn.cluster import MiniBatchKMeans, KMeans

# For handling warnings
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42

# Flag to control whether plots are displayed
SHOW_PLOTS = True  # Set to True to display plots, False for data collection

# Task type
TASK_TYPE = 'Regression'  # Adjusted task type

# Dataset selection

# 1
TARGET_COLUMN = 'Salary'
DATASET_NAME = 'Salary_dataset.csv'
data = pd.read_csv('Regression_Datasets/Salary_dataset.csv')


def preprocess_data(data, target_column):
    """
    Preprocesses the data:
    - Handles missing values separately for numerical and categorical columns
    - Encodes categorical variables (if any)
    - Scales numerical features
    """
    # Identify numerical and categorical columns
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

    # Exclude the target column from features
    if target_column in numerical_cols:
        numerical_cols.remove(target_column)
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)

    # 1. Handle Missing Values
    if numerical_cols:
        data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].mean())
    if categorical_cols:
        data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])

    # 2. Handle the Target Variable
    # Ensure target variable is numeric
    data[target_column] = pd.to_numeric(data[target_column], errors='coerce')
    data = data.dropna(subset=[target_column])  # Drop rows where target is NaN

    # 3. Encode Categorical Variables
    if categorical_cols:
        data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
        if SHOW_PLOTS:
            print(f"One-Hot Encoded categorical columns: {categorical_cols}")
    else:
        if SHOW_PLOTS:
            print("No categorical columns to encode.")

    # 4. Scale Numerical Features
    if numerical_cols:
        scaler = StandardScaler()
        data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    else:
        if SHOW_PLOTS:
            print("No numerical columns to scale.")

    return data

def extract_dataset_features(data, target_column):
    """
    Extracts features of the dataset that might influence coreset selection methods.
    """
    features = {}

    # Number of Instances and Features
    features['dataset_name'] = DATASET_NAME
    features['task_type'] = TASK_TYPE
    features['num_instances'] = data.shape[0]
    features['num_features'] = data.shape[1] - 1  # Exclude target column

    # Data Types
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numerical_cols:
        numerical_cols.remove(target_column)
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    features['num_numerical_features'] = len(numerical_cols)
    features['num_categorical_features'] = len(categorical_cols)

    # Feature Type Indicator
    if features['num_numerical_features'] > 0 and features['num_categorical_features'] > 0:
        features['feature_type'] = 'Mixed'
    elif features['num_numerical_features'] > 0:
        features['feature_type'] = 'Numerical'
    else:
        features['feature_type'] = 'Categorical'

    # For regression, 'num_classes', 'class_balance', and 'imbalance_ratio' are not applicable
    features['num_classes'] = None
    features['class_balance'] = None
    features['imbalance_ratio'] = None

    # Missing Values
    features['missing_values'] = data.isnull().sum().sum()
    features['missing_value_percentage'] = features['missing_values'] / (features['num_instances'] * features['num_features'])

    # Dimensionality
    features['dimensionality'] = features['num_features'] / features['num_instances']

    # Correlation Matrix (only for numerical features)
    if len(numerical_cols) > 1:
        corr_matrix = data[numerical_cols].corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        # Mean Correlation
        features['mean_correlation'] = upper_triangle.stack().mean()
        # Max Correlation
        features['max_correlation'] = upper_triangle.stack().max()
        # Feature Redundancy
        high_corr_pairs = upper_triangle.stack().apply(lambda x: x > 0.8).sum()
        features['feature_redundancy'] = high_corr_pairs
    else:
        features['mean_correlation'] = 0.0
        features['max_correlation'] = 0.0
        features['feature_redundancy'] = 0

    # Statistical Properties of Features
    if numerical_cols:
        feature_means = data[numerical_cols].mean()
        feature_variances = data[numerical_cols].var()
        # Mean of Means
        features['mean_of_means'] = feature_means.mean()
        # Variance of Means
        features['variance_of_means'] = feature_means.var()
        # Mean of Variances
        features['mean_of_variances'] = feature_variances.mean()
        # Variance of Variances
        features['variance_of_variances'] = feature_variances.var()
        # Skewness and Kurtosis
        features['mean_skewness'] = data[numerical_cols].skew().mean()
        features['mean_kurtosis'] = data[numerical_cols].kurtosis().mean()
    else:
        features['mean_of_means'] = 0.0
        features['variance_of_means'] = 0.0
        features['mean_of_variances'] = 0.0
        features['variance_of_variances'] = 0.0
        features['mean_skewness'] = 0.0
        features['mean_kurtosis'] = 0.0

    # Presence of Outliers (using IQR method)
    if numerical_cols:
        Q1 = data[numerical_cols].quantile(0.25)
        Q3 = data[numerical_cols].quantile(0.75)
        IQR = Q3 - Q1
        outlier_condition = ((data[numerical_cols] < (Q1 - 1.5 * IQR)) | (data[numerical_cols] > (Q3 + 1.5 * IQR)))
        outliers = outlier_condition.sum().sum()
        total_values = data[numerical_cols].shape[0] * data[numerical_cols].shape[1]
        features['outlier_percentage'] = outliers / total_values
    else:
        features['outlier_percentage'] = 0.0

    # Data Sparsity
    total_elements = data.shape[0] * data.shape[1]
    zero_elements = (data == 0).sum().sum()
    features['data_sparsity'] = zero_elements / total_elements

    return features

# Coreset Selection Techniques

def no_coreset_selection(X_train, y_train):
    return X_train, y_train

def random_sampling_coreset(X_train, y_train):
    fraction = 0.2  # Use 20% for coreset
    coreset_size = int(len(X_train) * fraction)
    coreset_size = max(1, coreset_size)
    indices = np.random.choice(len(X_train), size=coreset_size, replace=False)
    return X_train.iloc[indices], y_train.iloc[indices]

def kmeans_clustering_coreset(X_train, y_train):
    fraction = 0.05  # Use 5% for coreset
    coreset_size = int(len(X_train) * fraction)
    coreset_size = max(1, coreset_size)
    if SHOW_PLOTS:
        print(f"Using coreset size {coreset_size} for KMeans clustering.")

    kmeans = KMeans(n_clusters=coreset_size, random_state=RANDOM_STATE)
    kmeans.fit(X_train)
    labels = kmeans.labels_

    # Select one random sample from each cluster
    X_coreset = []
    y_coreset = []
    for i in range(coreset_size):
        indices_in_cluster = np.where(labels == i)[0]
        if len(indices_in_cluster) == 0:
            continue
        idx = np.random.choice(indices_in_cluster)
        X_coreset.append(X_train.iloc[idx])
        y_coreset.append(y_train.iloc[idx])
    X_coreset = pd.DataFrame(X_coreset)
    y_coreset = pd.Series(y_coreset)
    return X_coreset.reset_index(drop=True), y_coreset.reset_index(drop=True)

def uncertainty_sampling_coreset(X_train, y_train):
    fraction = 0.1  # Use 10% for coreset
    coreset_size = int(len(X_train) * fraction)
    coreset_size = max(1, coreset_size)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    residuals = np.abs(y_train - y_pred)
    indices = np.argsort(residuals)[-coreset_size:]  # Select samples with largest residuals
    return X_train.iloc[indices], y_train.iloc[indices]

def importance_sampling_coreset(X_train, y_train):
    fraction = 0.1  # Use 10% for coreset
    coreset_size = int(len(X_train) * fraction)
    coreset_size = max(1, coreset_size)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    residuals = np.abs(y_train - y_pred)
    importance = residuals + 1e-6  # To avoid zero probabilities
    importance /= importance.sum()
    indices = np.random.choice(len(X_train), size=coreset_size, replace=False, p=importance)
    return X_train.iloc[indices], y_train.iloc[indices]

def reservoir_sampling_coreset(X_train, y_train):
    fraction = 0.2  # Use 20% for coreset
    coreset_size = int(len(X_train) * fraction)
    coreset_size = max(1, coreset_size)
    n = len(X_train)
    indices = np.arange(n)
    np.random.shuffle(indices)
    selected_indices = indices[:coreset_size]
    return X_train.iloc[selected_indices], y_train.iloc[selected_indices]


def gradient_based_coreset(X_train, y_train):
    # Reset index to ensure default integer index
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    fraction = 0.1  # Adjust as needed
    total_coreset_size = int(len(X_train) * fraction)
    total_coreset_size = max(2, total_coreset_size)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_train)
    residuals = y_train - y_pred

    # Compute per-sample loss (squared errors)
    loss_per_sample = residuals ** 2

    # Now loss_per_sample has a default index starting from 0
    # Sort samples by loss (ascending)
    sorted_indices = loss_per_sample.sort_values(ascending=True).index

    # Select samples with lowest loss
    selected_indices = sorted_indices[:total_coreset_size]

    X_coreset = X_train.iloc[selected_indices].reset_index(drop=True)
    y_coreset = y_train.iloc[selected_indices].reset_index(drop=True)

    return X_coreset, y_coreset


def clustering_based_coreset(X_train, y_train):
    fraction = 0.05  # Use 5% for coreset
    coreset_size = int(len(X_train) * fraction)
    coreset_size = max(1, coreset_size)
    if SHOW_PLOTS:
        print(f"Using coreset size {coreset_size} for MiniBatchKMeans clustering.")

    kmeans = MiniBatchKMeans(n_clusters=coreset_size, random_state=RANDOM_STATE, batch_size=10000)
    kmeans.fit(X_train)
    labels = kmeans.labels_

    # Select one random sample from each cluster
    X_coreset = []
    y_coreset = []
    for i in range(coreset_size):
        indices_in_cluster = np.where(labels == i)[0]
        if len(indices_in_cluster) == 0:
            continue
        idx = np.random.choice(indices_in_cluster)
        X_coreset.append(X_train.iloc[idx])
        y_coreset.append(y_train.iloc[idx])
    X_coreset = pd.DataFrame(X_coreset)
    y_coreset = pd.Series(y_coreset)
    return X_coreset.reset_index(drop=True), y_coreset.reset_index(drop=True)

# Function to train and evaluate the model
def train_and_evaluate(X_train, y_train, X_test, y_test, coreset_method):
    """
    Applies coreset selection, trains the Linear Regression model, and evaluates it.
    """
    coreset_methods = {
        'none': no_coreset_selection,
        'random': random_sampling_coreset,
        'kmeans': kmeans_clustering_coreset,
        'uncertainty': uncertainty_sampling_coreset,
        'importance': importance_sampling_coreset,
        'reservoir': reservoir_sampling_coreset,
        'gradient': gradient_based_coreset,
        'clustering': clustering_based_coreset
    }

    if coreset_method not in coreset_methods:
        raise ValueError("Invalid coreset selection method.")

    X_coreset, y_coreset = coreset_methods[coreset_method](X_train, y_train)

    # Initialize the Linear Regression model
    lr = LinearRegression()

    # Train the model on the coreset
    lr.fit(X_coreset, y_coreset)

    # Predict on the test set
    y_pred = lr.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    if SHOW_PLOTS:
        # Plot predicted vs actual values
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.7)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Actual vs Predicted ({coreset_method.capitalize()})')
        plt.grid(True)
        plt.show()

    # Return metrics for comparison
    return {
        'coreset_method': coreset_method.capitalize(),
        'coreset_size': len(X_coreset),
        'mse': mse,
        'mae': mae,
        'r2_score': r2
    }

def main():
    # Extract dataset features
    dataset_features = extract_dataset_features(data, TARGET_COLUMN)

    # Preprocess the data
    data_preprocessed = preprocess_data(data, TARGET_COLUMN)

    X = data_preprocessed.drop(TARGET_COLUMN, axis=1)
    y = data_preprocessed[TARGET_COLUMN]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Define coreset methods to test
    coreset_methods = [
        'none', 'random', 'kmeans', 'uncertainty',
        'importance', 'reservoir', 'gradient', 'clustering'
    ]

    # Store results for comparison
    results = []

    # Loop over coreset methods
    for method in coreset_methods:
        result = train_and_evaluate(X_train, y_train, X_test, y_test, method)
        results.append(result)

    # Create a DataFrame to compare results
    results_df = pd.DataFrame(results)

    # Identify the best coreset method based on MSE (excluding 'None')
    results_df_non_none = results_df[results_df['coreset_method'] != 'None']
    best_method_idx = results_df_non_none['mse'].idxmin()
    best_method = results_df_non_none.loc[best_method_idx]

    # Get baseline MSE
    baseline_mse = results_df[results_df['coreset_method'] == 'None']['mse'].values[0]
    performance_gain = baseline_mse - best_method['mse']

    # Prepare evaluation metrics DataFrame
    evaluation_metrics = results_df.copy()
    evaluation_metrics.insert(0, 'dataset_name', DATASET_NAME)  # Move dataset_name to first column
    evaluation_metrics.insert(1, 'task_type', TASK_TYPE)  # Insert task_type as second column
    evaluation_metrics['baseline_mse'] = baseline_mse
    evaluation_metrics['performance_gain'] = baseline_mse - evaluation_metrics['mse']

    # Define the columns to be used
    columns_order = [
        'dataset_name', 'task_type', 'coreset_method', 'coreset_size',
        'mse', 'mae', 'r2_score',
        'baseline_mse', 'performance_gain'
    ]

    # Ensure all columns are present
    for col in columns_order:
        if col not in evaluation_metrics.columns:
            evaluation_metrics[col] = None  # or np.nan

    # Reorder the DataFrame columns
    evaluation_metrics = evaluation_metrics[columns_order]

    # Optionally, print the results
    print("\nEvaluation Metrics for All Coreset Methods:")
    print(evaluation_metrics[columns_order])

    if SHOW_PLOTS:
        # Plotting results
        plt.figure(figsize=(12, 6))
        bar_colors = sns.color_palette('hls', len(results_df))
        bars = plt.bar(results_df['coreset_method'], results_df['mse'], color=bar_colors)
        plt.xlabel('Coreset Method')
        plt.ylabel('Mean Squared Error')
        plt.title(f'Model Performance by Coreset Method on {DATASET_NAME}')
        plt.ylim(0, results_df['mse'].max() * 1.1)
        plt.grid(axis='y')

        # Annotate bars with MSE scores
        for bar, score in zip(bars, results_df['mse']):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.005 * yval, f'{score:.4f}', ha='center', va='bottom')

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        # Only write to CSV files if SHOW_PLOTS is False

        # Write evaluation metrics to CSV
        evaluation_metrics_file = 'evaluation_metrics_regression.csv'  # Updated file name

        # Check if the file exists and if it is empty
        if not os.path.exists(evaluation_metrics_file) or os.stat(evaluation_metrics_file).st_size == 0:
            header_eval = True
        else:
            header_eval = False

        # Write to CSV with proper quoting to handle any special characters
        evaluation_metrics.to_csv(
            evaluation_metrics_file,
            mode='a',
            index=False,
            header=header_eval
        )

        # Write dataset characteristics to meta_dataset_regression.csv
        meta_dataset_file = 'meta_dataset_regression.csv'  # Updated file name
        meta_dataset_columns_order = [
            'dataset_name', 'task_type', 'num_instances', 'num_features',
            'num_numerical_features', 'num_categorical_features', 'feature_type',
            'missing_values', 'missing_value_percentage', 'dimensionality', 'mean_correlation',
            'max_correlation', 'feature_redundancy', 'mean_of_means',
            'variance_of_means', 'mean_of_variances', 'variance_of_variances',
            'mean_skewness', 'mean_kurtosis', 'outlier_percentage', 'data_sparsity',
            'best_coreset_method'
        ]

        # Add best coreset method to dataset_features
        dataset_features['best_coreset_method'] = best_method['coreset_method']

        # Convert dataset_features to DataFrame
        meta_dataset_df = pd.DataFrame([dataset_features])

        # Ensure all columns are present
        for col in meta_dataset_columns_order:
            if col not in meta_dataset_df.columns:
                meta_dataset_df[col] = None  # or np.nan

        # Reorder columns
        meta_dataset_df = meta_dataset_df[meta_dataset_columns_order]

        # Check if the file exists and if it is empty
        if not os.path.exists(meta_dataset_file) or os.stat(meta_dataset_file).st_size == 0:
            header_meta = True
        else:
            header_meta = False

        # Write to meta_dataset_regression.csv
        meta_dataset_df.to_csv(
            meta_dataset_file,
            mode='a',
            index=False,
            header=header_meta
        )

if __name__ == "__main__":
    main()

