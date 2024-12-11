import pandas as pd
import numpy as np
import os

# For data splitting and model training
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# For model evaluation (regression)
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
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

# For imputing missing values
from sklearn.impute import SimpleImputer

# Set random seed for reproducibility
RANDOM_STATE = 42

# Flag to control whether plots are displayed
SHOW_PLOTS = True  # Set to True to display plots, False for data collection

# Task type - now Regression
TASK_TYPE = 'Regression'  # Adjusted from 'Binary Classification' to 'Regression'

# Dataset selection

# Uncomment the desired dataset and adjust accordingly

# 1
#TARGET_COLUMN = 'Salary'
#DATASET_NAME = 'Salary_dataset.csv'
#data = pd.read_csv('Regression_Datasets/Salary_dataset.csv')

# 2
#TARGET_COLUMN = 'Performance Index'
#DATASET_NAME = 'Student_Performance.csv'
#data = pd.read_csv('Regression_Datasets/Student_Performance.csv')

# 3
#TARGET_COLUMN = 'footlgth'
#DATASET_NAME = 'possum.csv'
#data = pd.read_csv('Regression_Datasets/possum.csv')

# 4
#TARGET_COLUMN = 'Ice Cream Sales (units)'
#DATASET_NAME = 'Ice_cream selling data.csv'
#data = pd.read_csv('Regression_Datasets/Ice_cream selling data.csv')

# 5
#TARGET_COLUMN = 'price'
#DATASET_NAME = 'kc_house_data.csv'
#data = pd.read_csv('Regression_Datasets/kc_house_data.csv')

# 6
#TARGET_COLUMN = 'quality'
#DATASET_NAME = 'winequality-red.csv'
#data = pd.read_csv('Regression_Datasets/winequality-red.csv')

# 7
#TARGET_COLUMN = 'Weekly_Sales'
#DATASET_NAME = 'Walmart.csv'
#data = pd.read_csv('Regression_Datasets/Walmart.csv')

# 8
#TARGET_COLUMN = 'Weight'
#DATASET_NAME = 'heightweight.csv'
#data = pd.read_csv('Regression_Datasets/heightweight.csv')

# 9
#TARGET_COLUMN = 'Urban Population'
#DATASET_NAME = 'population.csv'
#data = pd.read_csv('Regression_Datasets/population.csv')

# 10
#TARGET_COLUMN = 'Strength'
#DATASET_NAME = 'concrete.csv'
#data = pd.read_csv('Regression_Datasets/concrete_data.csv')

# 11
# TARGET_COLUMN = 'Scores'
# DATASET_NAME = 'score.csv'
# data = pd.read_csv('Regression_Datasets/score.csv')

# 12
#TARGET_COLUMN = 'Car Purchase Amount'
#DATASET_NAME = 'Car_Purchasing_Data.csv'
#data = pd.read_csv('Regression_Datasets/Car_Purchasing_Data.csv')

# 13
#TARGET_COLUMN = 'salary(in thousands)'
#DATASET_NAME = 'Experience-Salary.csv'
#data = pd.read_csv('Regression_Datasets/Experience-Salary.csv')

# 14
#TARGET_COLUMN = 'sales'
#DATASET_NAME = 'Marketing_Data.csv'
#data = pd.read_csv('Regression_Datasets/Marketing_Data.csv')

# 15
#TARGET_COLUMN = 'Son'
#DATASET_NAME = 'fathersonheight.csv'
#data = pd.read_csv('Regression_Datasets/fathersonheight.csv')

# 16
#TARGET_COLUMN = 'Life expectancy '
#DATASET_NAME = 'Life Expectancy Data.csv'
#data = pd.read_csv('Regression_Datasets/Life Expectancy Data.csv')

# 17
#TARGET_COLUMN = 'sales'
#DATASET_NAME = 'Advertising.csv'
#data = pd.read_csv('Regression_Datasets/Advertising.csv')

# 18
#TARGET_COLUMN = 'ESE'
#DATASET_NAME = 'AttendanceMarksSA.csv'
#data = pd.read_csv('Regression_Datasets/AttendanceMarksSA.csv')

# 19
#TARGET_COLUMN = 'PerFemEmploy'
#DATASET_NAME = 'FemEmployment.csv'
#data = pd.read_csv('Regression_Datasets/FemEmployment.csv')

# 20
#TARGET_COLUMN = 'Weight'
#DATASET_NAME = 'Fish_dataset.csv'
#data = pd.read_csv('Regression_Datasets/Fish_dataset.csv')

# 21
#TARGET_COLUMN = 'price'
#DATASET_NAME = 'ParisHousing.csv'
#data = pd.read_csv('Regression_Datasets/ParisHousing.csv')

# 22
#TARGET_COLUMN = 'Income'
#DATASET_NAME = 'AnnualHouseIncome.csv'
#data = pd.read_csv('Regression_Datasets/AnnualHouseIncome.csv')

# 23
#TARGET_COLUMN = 'Gold_Price'
#DATASET_NAME = 'GoldUP.csv'
#data = pd.read_csv('Regression_Datasets/GoldUP.csv')

# 24
#TARGET_COLUMN = 'Combat Power'
#DATASET_NAME = 'pokemon.csv'
#data = pd.read_csv('Regression_Datasets/pokemon.csv')

# 25
#TARGET_COLUMN = 'median_house_value'
#DATASET_NAME = 'Calihousing.csv'
#data = pd.read_csv('Regression_Datasets/Calihousing.csv')


def preprocess_data(data, target_column):
    """
    Preprocesses the data for regression:
    - Handles missing values
    - Encodes categorical variables
    - Scales numerical features
    - Ensures target variable is numeric
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

    # 2. Ensure the target variable is numeric
    data[target_column] = pd.to_numeric(data[target_column], errors='coerce')
    # Drop rows where target is NaN after conversion
    data = data.dropna(subset=[target_column])

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
    For regression, we set num_classes, class_balance, and imbalance_ratio to None.
    """
    features = {}

    # Basic Dataset Information
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

    # For regression tasks, set class-related metrics to None
    if TASK_TYPE == 'Regression':
        features['num_classes'] = None
        features['class_balance'] = None
        features['imbalance_ratio'] = None
    else:
        # If classification were used, original code would apply. Kept for minimal changes.
        target_values = data[target_column]
        class_counts = target_values.value_counts()
        features['num_classes'] = len(class_counts)
        class_balance = (class_counts / class_counts.sum()).to_dict()
        class_balance = {str(k): v for k, v in class_balance.items()}
        features['class_balance'] = class_balance
        majority_class = class_counts.max()
        minority_class = class_counts.min()
        features['imbalance_ratio'] = majority_class / minority_class

    # Missing Values
    #features['missing_values'] = data.isnull().sum().sum()
    #features['missing_value_percentage'] = features['missing_values'] / (
    #        features['num_instances'] * features['num_features'])

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


def importance_sampling_coreset(X_train, y_train):
    # For regression, use residuals as importance weights
    fraction = 0.1  # Increased from 5% to 10%
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
    fraction = 0.2  # Increased from 10% to 20%
    coreset_size = int(len(X_train) * fraction)
    coreset_size = max(1, coreset_size)
    n = len(X_train)
    indices = np.arange(n)
    np.random.shuffle(indices)
    selected_indices = indices[:coreset_size]
    return X_train.iloc[selected_indices], y_train.iloc[selected_indices]


def gradient_based_coreset(X_train, y_train):
    """
    For regression, select samples with the highest squared error loss.
    """
    fraction = 0.1  # Adjust as needed
    total_coreset_size = int(len(X_train) * fraction)
    total_coreset_size = max(2, total_coreset_size)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)

    # Squared error loss per sample
    loss_per_sample = (y_train - y_pred) ** 2

    # Convert to NumPy array for positional indexing
    loss_array = loss_per_sample.to_numpy()
    sorted_indices = np.argsort(loss_array)[-total_coreset_size:]
    selected_indices = sorted_indices
    X_coreset = X_train.iloc[selected_indices].reset_index(drop=True)
    y_coreset = y_train.iloc[selected_indices].reset_index(drop=True)
    return X_coreset, y_coreset


def farthest_point_sampling_coreset(X_train, y_train):
    """
    Selects samples using Farthest Point Sampling to maximize diversity.

    Args:
        X_train (DataFrame): Training features.
        y_train (Series): Training target.

    Returns:
        DataFrame, Series: Selected coreset.
    """
    fraction = 0.075  # Adjust the fraction as needed
    coreset_size = int(len(X_train) * fraction)
    coreset_size = max(1, coreset_size)

    X = X_train.values
    n_samples, n_features = X.shape

    if n_samples == 0:
        raise ValueError("X_train has no samples.")

    # Initialize with a random point
    np.random.seed(RANDOM_STATE)
    initial_idx = np.random.randint(0, n_samples)
    selected_indices = [initial_idx]

    # Compute distances from the initial point
    if n_features == 1:
        distances = np.abs(X - X[initial_idx]).flatten()
    else:
        distances = np.linalg.norm(X - X[initial_idx], axis=1)

    for _ in range(coreset_size - 1):
        # Select the point with the maximum distance to the selected set
        next_idx = np.argmax(distances)
        selected_indices.append(next_idx)

        # Update the distances
        if n_features == 1:
            new_distances = np.abs(X - X[next_idx]).flatten()
        else:
            new_distances = np.linalg.norm(X - X[next_idx], axis=1)
        distances = np.minimum(distances, new_distances)

    return X_train.iloc[selected_indices], y_train.iloc[selected_indices]


def leverage_sampling_coreset(X_train, y_train):
    """
    Selects samples with the highest leverage scores.

    Args:
        X_train (DataFrame): Training features.
        y_train (Series): Training target.

    Returns:
        DataFrame, Series: Selected coreset.
    """
    fraction = 0.1  # Adjust as needed
    coreset_size = int(len(X_train) * fraction)
    coreset_size = max(1, coreset_size)

    # Ensure all data is float64
    try:
        X_numeric = X_train.astype(float).values  # This will raise an error if conversion fails
    except ValueError as e:
        print("Error converting X_train to float:", e)
        print("Data Types in X_train:")
        print(X_train.dtypes)
        raise

    # Add intercept term
    X_with_intercept = np.hstack([np.ones((X_numeric.shape[0], 1)), X_numeric])

    # Compute Hat matrix
    try:
        hat_matrix = X_with_intercept @ np.linalg.pinv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T
    except np.linalg.LinAlgError:
        # In case of singular matrix, fallback to using k-means
        print("Singular matrix encountered in leverage calculation. Falling back to KMeans sampling.")
        return kmeans_clustering_coreset(X_train, y_train)

    leverage_scores = np.diag(hat_matrix)

    # Select samples with highest leverage
    indices = np.argsort(leverage_scores)[-coreset_size:]
    return X_train.iloc[indices], y_train.iloc[indices]


def clustering_based_coreset(X_train, y_train):
    # Same logic as classification, just without class concept
    if len(X_train) > 100000:
        fraction = 0.01
    else:
        fraction = 0.05
    coreset_size = int(len(X_train) * fraction)
    coreset_size = max(1, coreset_size)
    if SHOW_PLOTS:
        print(f"Using coreset size {coreset_size} for MiniBatchKMeans clustering.")

    kmeans = MiniBatchKMeans(n_clusters=coreset_size, random_state=RANDOM_STATE, batch_size=10000)
    kmeans.fit(X_train)
    labels = kmeans.labels_

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
        'importance': importance_sampling_coreset,
        'reservoir': reservoir_sampling_coreset,
        'gradient': gradient_based_coreset,
        'clustering': clustering_based_coreset,
        'leverage': leverage_sampling_coreset,  # Retained
        'farthest': farthest_point_sampling_coreset  # New method
    }

    if coreset_method not in coreset_methods:
        raise ValueError("Invalid coreset selection method.")

    X_coreset, y_coreset = coreset_methods[coreset_method](X_train, y_train)

    # Use Linear Regression for regression tasks
    lr = LinearRegression()
    lr.fit(X_coreset, y_coreset)

    y_pred = lr.predict(X_test)

    # Compute regression metrics
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

    # Return regression metrics
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

    # Split the data into training and testing sets without stratification for regression
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Reset indices to ensure positional indexing from 0
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Convert all columns to numeric, coercing errors to NaN
    X_train = X_train.apply(pd.to_numeric, errors='coerce')
    X_test = X_test.apply(pd.to_numeric, errors='coerce')

    # Handle NaN values if any
    # Example: Impute with mean
    imputer = SimpleImputer(strategy='mean')
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    # Define coreset methods to test (removed 'residual' and 'density', added 'farthest')
    coreset_methods = [
        'none', 'random', 'kmeans',
        'importance', 'reservoir', 'gradient', 'clustering',
        'leverage', 'farthest'
    ]

    # Store results for comparison
    results = []

    # Loop over coreset methods
    for method in coreset_methods:
        print(f"\nApplying coreset method: {method}")
        result = train_and_evaluate(X_train, y_train, X_test, y_test, method)
        results.append(result)

    # Create a DataFrame to compare results
    results_df = pd.DataFrame(results)

    # Identify the best coreset method based on MSE (lowest MSE is best)
    results_df_non_none = results_df[results_df['coreset_method'] != 'None']
    if not results_df_non_none.empty:
        best_method_idx = results_df_non_none['mse'].idxmin()
        best_method = results_df_non_none.loc[best_method_idx]
    else:
        best_method = None

    # Get baseline MSE
    if 'None' in results_df['coreset_method'].values:
        baseline_mse = results_df[results_df['coreset_method'] == 'None']['mse'].values[0]
    else:
        baseline_mse = None

    # Prepare evaluation metrics DataFrame
    evaluation_metrics = results_df.copy()
    evaluation_metrics.insert(0, 'dataset_name', DATASET_NAME)  # Move dataset_name to first column
    evaluation_metrics.insert(1, 'task_type', TASK_TYPE)  # Insert task_type as second column

    # Calculate performance gain for each method
    if baseline_mse is not None:
        evaluation_metrics['baseline_mse'] = baseline_mse
        # performance_gain = baseline_mse - method_mse
        evaluation_metrics['performance_gain'] = evaluation_metrics['baseline_mse'] - evaluation_metrics['mse']
    else:
        evaluation_metrics['baseline_mse'] = None
        evaluation_metrics['performance_gain'] = None

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
        for bar, score in zip(bars, results_df['mse']):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.005 * yval, f'{score:.4f}', ha='center', va='bottom')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        # Write evaluation metrics to a new file for regression
        evaluation_metrics_file = 'evaluation_metrics_regression.csv'

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

        # Write dataset characteristics to meta_dataset.csv (same file as classification)
        meta_dataset_file = 'meta_dataset.csv'
        meta_dataset_columns_order = [
            'dataset_name', 'task_type', 'num_instances', 'num_features',
            'num_numerical_features', 'num_categorical_features', 'feature_type',
            'num_classes', 'class_balance', 'imbalance_ratio',
            #'missing_values',
            #'missing_value_percentage',
            'dimensionality', 'mean_correlation',
            'max_correlation', 'feature_redundancy', 'mean_of_means',
            'variance_of_means', 'mean_of_variances', 'variance_of_variances',
            'mean_skewness', 'mean_kurtosis', 'outlier_percentage', 'data_sparsity',
            'best_coreset_method'
        ]

        if best_method is not None:
            dataset_features['best_coreset_method'] = best_method['coreset_method']
        else:
            dataset_features['best_coreset_method'] = None

        # Convert dataset_features to DataFrame
        meta_dataset_df = pd.DataFrame([dataset_features])

        # Ensure all columns are present
        for col in meta_dataset_columns_order:
            if col not in meta_dataset_df.columns:
                meta_dataset_df[col] = None

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
