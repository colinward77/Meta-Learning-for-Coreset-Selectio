# Import necessary libraries
import pandas as pd
import numpy as np

# For data splitting and model training
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# For model evaluation
from sklearn.metrics import accuracy_score

# For data scaling and encoding
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier

# For clustering-based sampling
from sklearn.cluster import MiniBatchKMeans, KMeans

# For plotting
import matplotlib.pyplot as plt
import seaborn as sns

# For handling warnings
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42

# Load the dataset
data_path = '/Users/ahmed/Desktop/Machine Learning/top_chess_players.csv'
data = pd.read_csv(data_path)

# Define the target column as the first column
TARGET_COLUMN = data.columns[0]

# Data Preprocessing Function
def preprocess_data(data, target_column):
    """
    Preprocess the dataset: scales numerical features, encodes categorical variables,
    and handles missing values.
    """
    # Handle missing values by filling with median for numerical columns
    for col in data.select_dtypes(include=['float64', 'int64']).columns:
        data[col].fillna(data[col].median(), inplace=True)

    # Encode categorical columns
    for col in data.select_dtypes(include=['object']).columns:
        if col != target_column:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))

    # Encode the target column
    target_encoder = LabelEncoder()
    data[target_column] = target_encoder.fit_transform(data[target_column].astype(str))

    # Initialize the scaler
    scaler = StandardScaler()

    # Scale numerical features (excluding the target column)
    numerical_features = [col for col in data.select_dtypes(include=['float64', 'int64']).columns if col != target_column]
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    return data

# Coreset Selection Techniques
def no_coreset_selection(X_train, y_train):
    return X_train, y_train

def random_sampling_coreset(X_train, y_train, fraction=0.1):
    coreset_size = int(len(X_train) * fraction)
    indices = np.random.choice(len(X_train), size=coreset_size, replace=False)
    return X_train.iloc[indices], pd.Series(y_train).iloc[indices]

def stratified_sampling_coreset(X_train, y_train, fraction=0.1):
    coreset_size = int(len(X_train) * fraction)
    X_coreset, _, y_coreset, _ = train_test_split(
        X_train, pd.Series(y_train),
        train_size=coreset_size,
        stratify=y_train,
        random_state=RANDOM_STATE
    )
    return X_coreset, y_coreset

def kmeans_clustering_coreset(X_train, y_train, fraction=0.05):
    coreset_size = max(1, int(len(X_train) * fraction))
    kmeans = KMeans(n_clusters=coreset_size, random_state=RANDOM_STATE)
    kmeans.fit(X_train)
    labels = kmeans.labels_
    indices = [np.where(labels == i)[0][0] for i in range(coreset_size)]
    return X_train.iloc[indices], pd.Series(y_train).iloc[indices]

def uncertainty_sampling_coreset(X_train, y_train, fraction=0.05):
    coreset_size = int(len(X_train) * fraction)
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, multi_class='multinomial')
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_train)
    uncertainty = 1 - np.max(proba, axis=1)
    indices = np.argsort(uncertainty)[-coreset_size:]
    return X_train.iloc[indices], pd.Series(y_train).iloc[indices]

def minibatch_kmeans_coreset(X_train, y_train, fraction=0.05):
    coreset_size = max(1, int(len(X_train) * fraction))
    kmeans = MiniBatchKMeans(n_clusters=coreset_size, random_state=RANDOM_STATE, batch_size=1000)
    kmeans.fit(X_train)
    labels = kmeans.labels_
    indices = []
    for i in range(coreset_size):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) > 0:
            indices.append(cluster_indices[0])
        else:
            indices.append(np.random.randint(0, len(X_train)))
    return X_train.iloc[indices], pd.Series(y_train).iloc[indices]

def reservoir_sampling_coreset(X_train, y_train, fraction=0.05):
    coreset_size = max(1, int(len(X_train) * fraction))
    indices = []
    for i, _ in enumerate(X_train):
        if i < coreset_size:
            indices.append(i)
        else:
            j = np.random.randint(0, i + 1)
            if j < coreset_size:
                indices[j] = i
    return X_train.iloc[indices], pd.Series(y_train).iloc[indices]

def gradient_based_coreset(X_train, y_train, fraction=0.05):
    coreset_size = int(len(X_train) * fraction)
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, multi_class='multinomial')
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_train)
    epsilon = 1e-15
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
    y_train_one_hot = pd.get_dummies(y_train).to_numpy()
    loss_per_sample = -np.sum(y_train_one_hot * np.log(y_pred_proba), axis=1)
    indices = np.argsort(loss_per_sample)[-coreset_size:]
    return X_train.iloc[indices], pd.Series(y_train).iloc[indices]

# Train and Evaluate Function
def train_and_evaluate(X_train, y_train, X_test, y_test, coreset_method):
    X_coreset, y_coreset = coreset_method(X_train, y_train)
    model = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    model.fit(X_coreset, y_coreset)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)




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

# Main Execution
def main():
    # Preprocess the data
    data_preprocessed = preprocess_data(data, TARGET_COLUMN)

    # Split the data into features (X) and target (y)
    X = data_preprocessed.drop(TARGET_COLUMN, axis=1)
    y = data_preprocessed[TARGET_COLUMN]

    # Remove classes with fewer than 2 instances
    class_counts = y.value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    X = X[y.isin(valid_classes)]
    y = y[y.isin(valid_classes)]

    # Split into training and testing sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    # Define coreset methods to test
    coreset_methods = {
        'None': no_coreset_selection,
        'Random Sampling': random_sampling_coreset,
        'Stratified Sampling': stratified_sampling_coreset,
        'KMeans Clustering': kmeans_clustering_coreset,
        'Uncertainty Sampling': uncertainty_sampling_coreset,
        'MiniBatch KMeans': minibatch_kmeans_coreset,
        'Reservoir Sampling': reservoir_sampling_coreset,
        'Gradient-Based Sampling': gradient_based_coreset,
    }

    # Evaluate each coreset method
    results = []
    for method_name, method_function in coreset_methods.items():
        accuracy = train_and_evaluate(X_train, y_train, X_test, y_test, method_function)
        results.append({'Coreset Method': method_name, 'Accuracy': accuracy})
        print(f"{method_name}: Accuracy = {accuracy:.4f}")

    # Create a DataFrame for results
    results_df = pd.DataFrame(results)

    # Plot the results
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df, x='Coreset Method', y='Accuracy', palette='husl')
    plt.title('Accuracy by Coreset Selection Technique')
    plt.xlabel('Coreset Method')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.show()

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
