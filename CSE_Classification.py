import pandas as pd
import numpy as np
import os

# For data splitting and model training
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# For model evaluation
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
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
TASK_TYPE = 'Binary Classification'  # Adjust as needed

# Dataset selection
# Uncomment the dataset you want to use and set the appropriate TARGET_COLUMN, POS_LABEL, and NEG_LABEL

# 1
#TARGET_COLUMN = 'Class'
#DATASET_NAME = 'creditcard.csv'
#POS_LABEL = 1
#NEG_LABEL = 0
#data = pd.read_csv('Classification_Datasets/creditcard.csv')

# 2
#TARGET_COLUMN = 'Bankrupt?'
#DATASET_NAME = 'bankrupcy.csv'
#POS_LABEL = 1
#NEG_LABEL = 0
#data = pd.read_csv('Classification_Datasets/bankrupcy.csv')

# 3
#TARGET_COLUMN = 'loan'
#DATASET_NAME = 'bank.csv'
#POS_LABEL = 'yes'
#NEG_LABEL = 'no'
#data = pd.read_csv('Classification_Datasets/bank.csv')

# 4
#TARGET_COLUMN = ' loan_status'
#DATASET_NAME = 'loan_approval_dataset.csv'
#POS_LABEL = ' Approved'
#NEG_LABEL = ' Rejected'
#data = pd.read_csv('Classification_Datasets/loan_approval_dataset.csv')

# 5
#TARGET_COLUMN = 'Exited'
#DATASET_NAME = 'Churn_Modelling.csv'
#POS_LABEL = 1
#NEG_LABEL = 0
#data = pd.read_csv('Classification_Datasets/Churn_Modelling.csv')

# 6
#TARGET_COLUMN = 'Attrition'
#DATASET_NAME = 'employee_atrr.csv'
#POS_LABEL = 'Stayed'
#NEG_LABEL = 'Left'
#data = pd.read_csv('Classification_Datasets/employee_atrr.csv')

# 7
#TARGET_COLUMN = 'Class'
#DATASET_NAME = 'Raisin_Dataset.csv'
#POS_LABEL = 'Kecimen'
#NEG_LABEL = 'Besni'
#data = pd.read_csv('Classification_Datasets/Raisin_Dataset.csv')

# 8
#TARGET_COLUMN = 'class'
#DATASET_NAME = 'mushroom_cleaned.csv'
#POS_LABEL = 1
#NEG_LABEL = 0
#data = pd.read_csv('Classification_Datasets/mushroom_cleaned.csv')

# 9
TARGET_COLUMN = 'diagnosis'
DATASET_NAME = 'breast-cancer.csv'
POS_LABEL = 'M'
NEG_LABEL = 'B'
data = pd.read_csv('Classification_Datasets/breast-cancer.csv')

# 10
#TARGET_COLUMN = 'y'
#DATASET_NAME = 'bank_marketing.csv'
#POS_LABEL = 'yes'
#NEG_LABEL = 'no'
#data = pd.read_csv('Classification_Datasets/bank_marketing.csv')

def preprocess_data(data, target_column, pos_label, neg_label):
    """
    Preprocesses the data:
    - Handles missing values separately for numerical and categorical columns
    - Encodes categorical variables (if any)
    - Scales numerical features
    - Encodes the target variable using POS_LABEL and NEG_LABEL
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
    # Convert target column to string for consistent processing
    data[target_column] = data[target_column].astype(str).str.strip().str.lower()

    # Convert POS_LABEL and NEG_LABEL to lowercase strings
    pos_label = str(pos_label).strip().lower()
    neg_label = str(neg_label).strip().lower()

    # Map target values to 1 and 0
    label_mapping = {pos_label: 1, neg_label: 0}

    # Check for unexpected labels
    unique_labels = set(data[target_column].unique())
    expected_labels = set(label_mapping.keys())
    unexpected_labels = unique_labels - expected_labels

    if unexpected_labels:
        print(f"Warning: Found unexpected labels in the target column '{target_column}': {unexpected_labels}")
        print(f"Expected labels are: {expected_labels}")
        # Optionally, handle unexpected labels (e.g., remove or map them)
        # For now, we'll proceed and map only the expected labels

    # Map the labels
    data[target_column] = data[target_column].map(label_mapping)

    # Drop rows with NaN in the target column after mapping (if any)
    data = data.dropna(subset=[target_column])

    # Convert target column to integer type
    data[target_column] = data[target_column].astype(int)

    # 3. Encode Categorical Variables
    if categorical_cols:
        data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
        print(f"One-Hot Encoded categorical columns: {categorical_cols}")
    else:
        print("No categorical columns to encode.")

    # 4. Scale Numerical Features
    if numerical_cols:
        scaler = StandardScaler()
        data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    else:
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

    # Class Balance (binary classification)
    target_values = data[target_column]
    class_counts = target_values.value_counts()
    features['num_classes'] = len(class_counts)
    class_balance = (class_counts / class_counts.sum()).to_dict()
    # Ensure class labels are strings for consistency in CSV
    class_balance = {str(k): v for k, v in class_balance.items()}
    features['class_balance'] = class_balance
    # Imbalance Ratio
    majority_class = class_counts.max()
    minority_class = class_counts.min()
    features['imbalance_ratio'] = majority_class / minority_class

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
    fraction = 0.2  # Increased from 10% to 20%
    coreset_size = int(len(X_train) * fraction)
    coreset_size = max(1, coreset_size)
    indices = np.random.choice(len(X_train), size=coreset_size, replace=False)
    return X_train.iloc[indices], y_train.iloc[indices]

def stratified_sampling_coreset(X_train, y_train):
    fraction = 0.2  # Increased from 10% to 20%
    coreset_size = int(len(X_train) * fraction)
    coreset_size = max(1, coreset_size)
    X_coreset, _, y_coreset, _ = train_test_split(
        X_train, y_train,
        train_size=coreset_size,
        stratify=y_train,
        random_state=RANDOM_STATE
    )
    return X_coreset, y_coreset

def kmeans_clustering_coreset(X_train, y_train):
    if len(X_train) > 100000:
        fraction = 0.01  # Use 1% for large datasets
    else:
        fraction = 0.05  # Increased from 1% to 5%
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
    fraction = 0.1  # Increased from 5% to 10%
    coreset_size = int(len(X_train) * fraction)
    coreset_size = max(1, coreset_size)
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_train)[:, 1]
    uncertainty = np.abs(proba - 0.5)
    indices = np.argsort(uncertainty)[:coreset_size]
    return X_train.iloc[indices], y_train.iloc[indices]

def importance_sampling_coreset(X_train, y_train):
    fraction = 0.1  # Increased from 5% to 10%
    coreset_size = int(len(X_train) * fraction)
    coreset_size = max(1, coreset_size)
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_train)[:, 1]
    importance = np.abs(proba - 0.5)
    importance += 1e-6  # To avoid zero probabilities
    importance /= importance.sum()
    indices = np.random.choice(len(X_train), size=coreset_size, replace=False, p=importance)
    return X_train.iloc[indices], y_train.iloc[indices]

def reservoir_sampling_coreset(X_train, y_train):
    fraction = 0.2  # Increased from 10% to 20%
    coreset_size = int(len(X_train) * fraction)
    coreset_size = max(1, coreset_size)
    n = len(X_train)
    indices = np.arange(n)
    np.random.shuffle(indices)
    selected_indices = indices[:coreset_size]
    return X_train.iloc[selected_indices], y_train.iloc[selected_indices]

def gradient_based_coreset(X_train, y_train):
    fraction = 0.1  # Adjust as needed
    total_coreset_size = int(len(X_train) * fraction)
    total_coreset_size = max(2, total_coreset_size)

    y_train = y_train.astype(int)

    initial_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    initial_model.fit(X_train, y_train)

    y_pred_proba = initial_model.predict_proba(X_train)[:, 1]

    epsilon = 1e-15
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
    loss_per_sample = - (y_train * np.log(y_pred_proba) + (1 - y_train) * np.log(1 - y_pred_proba))

    df = pd.DataFrame({
        'loss': loss_per_sample,
        'label': y_train
    })

    class_counts = y_train.value_counts()
    class_fractions = class_counts / len(y_train)
    coreset_sizes = (class_fractions * total_coreset_size).astype(int).clip(lower=1)

    coreset_indices = []

    for cls in y_train.unique():
        cls_indices = df[df['label'] == cls].index
        cls_losses = df.loc[cls_indices, 'loss']
        # Select samples with lowest loss
        cls_sorted_indices = cls_losses.sort_values(ascending=True).index
        num_samples = coreset_sizes[cls]
        selected_indices = cls_sorted_indices[:num_samples]
        coreset_indices.extend(selected_indices)

    # Adjust coreset size if necessary
    coreset_indices = coreset_indices[:total_coreset_size]

    X_coreset = X_train.loc[coreset_indices].reset_index(drop=True)
    y_coreset = y_train.loc[coreset_indices].reset_index(drop=True)

    # Check class distribution debug statement
    # coreset_class_distribution = y_coreset.value_counts(normalize=True)
    # print("Coreset class distribution:")
    # print(coreset_class_distribution)

    return X_coreset, y_coreset



def clustering_based_coreset(X_train, y_train):
    if len(X_train) > 100000:
        fraction = 0.01  # Use 1% for large datasets
    else:
        fraction = 0.05  # Increased from 1% to 5%
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
    Applies coreset selection, trains the Logistic Regression model, and evaluates it.
    """
    coreset_methods = {
        'none': no_coreset_selection,
        'random': random_sampling_coreset,
        'stratified': stratified_sampling_coreset,
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

    # Initialize the Logistic Regression model
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

    # Train the model on the coreset
    lr.fit(X_coreset, y_coreset)

    # Predict on the test set
    y_pred = lr.predict(X_test)
    y_pred_proba = lr.predict_proba(X_test)[:, 1]

    # Evaluate the model
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    # ROC AUC Score
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Calculate additional metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    if SHOW_PLOTS:
        # Plot the ROC Curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{coreset_method.capitalize()} (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve ({coreset_method.capitalize()})')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()

    # Return metrics for comparison
    return {
        'coreset_method': coreset_method.capitalize(),
        'coreset_size': len(X_coreset),
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def main():
    # Extract dataset features
    dataset_features = extract_dataset_features(data, TARGET_COLUMN)

    # Preprocess the data using POS_LABEL and NEG_LABEL
    data_preprocessed = preprocess_data(data, TARGET_COLUMN, POS_LABEL, NEG_LABEL)

    X = data_preprocessed.drop(TARGET_COLUMN, axis=1)
    y = data_preprocessed[TARGET_COLUMN]

    # Split the data into training and testing sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # Define coreset methods to test
    coreset_methods = [
        'none', 'random', 'stratified', 'kmeans', 'uncertainty',
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

    # Identify the best coreset method based on ROC AUC (excluding 'None')
    results_df_non_none = results_df[results_df['coreset_method'] != 'None']
    best_method_idx = results_df_non_none['roc_auc'].idxmax()
    best_method = results_df_non_none.loc[best_method_idx]

    # Get baseline ROC AUC
    baseline_roc_auc = results_df[results_df['coreset_method'] == 'None']['roc_auc'].values[0]
    performance_gain = best_method['roc_auc'] - baseline_roc_auc

    # Prepare evaluation metrics DataFrame
    evaluation_metrics = results_df.copy()
    evaluation_metrics.insert(0, 'dataset_name', DATASET_NAME)  # Move dataset_name to first column
    evaluation_metrics.insert(1, 'task_type', TASK_TYPE)  # Insert task_type as second column
    evaluation_metrics['baseline_roc_auc'] = baseline_roc_auc
    evaluation_metrics['performance_gain'] = evaluation_metrics['roc_auc'] - baseline_roc_auc

    # Define the columns to be used
    columns_order = [
        'dataset_name', 'task_type', 'coreset_method', 'coreset_size',
        'roc_auc', 'precision', 'recall', 'f1_score',
        'baseline_roc_auc', 'performance_gain'
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
        bars = plt.bar(results_df['coreset_method'], results_df['roc_auc'], color=bar_colors)
        plt.xlabel('Coreset Method')
        plt.ylabel('ROC AUC Score')
        plt.title(f'Model Performance by Coreset Method on {DATASET_NAME}')
        plt.ylim(0.5, 1)
        plt.grid(axis='y')

        # Annotate bars with ROC AUC scores
        for bar, score in zip(bars, results_df['roc_auc']):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.005, f'{score:.4f}', ha='center', va='bottom')

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        # Only write to CSV files if SHOW_PLOTS is False

        # Write evaluation metrics to CSV
        evaluation_metrics_file = 'evaluation_metrics.csv'

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

        # Write dataset characteristics to meta_dataset.csv
        meta_dataset_file = 'meta_dataset.csv'
        meta_dataset_columns_order = [
            'dataset_name', 'task_type', 'num_instances', 'num_features',
            'num_numerical_features', 'num_categorical_features', 'feature_type',
            'num_classes', 'class_balance', 'imbalance_ratio', 'missing_values',
            'missing_value_percentage', 'dimensionality', 'mean_correlation',
            'max_correlation', 'feature_redundancy', 'mean_of_means',
            'variance_of_means', 'mean_of_variances', 'variance_of_variances',
            'mean_skewness', 'mean_kurtosis', 'outlier_percentage', 'data_sparsity',
            'best_coreset_method'  # Removed performance_gain
        ]

        # Add best coreset method to dataset_features
        dataset_features['best_coreset_method'] = best_method['coreset_method']
        # Removed 'performance_gain' from dataset_features

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

        # Write to meta_dataset.csv
        meta_dataset_df.to_csv(
            meta_dataset_file,
            mode='a',
            index=False,
            header=header_meta
        )

if __name__ == "__main__":
    main()
