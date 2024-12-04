import pandas as pd
import numpy as np

# For data splitting and model training
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# For model evaluation
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    log_loss
)

# For data scaling and encoding
from sklearn.preprocessing import StandardScaler, LabelEncoder

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

# Define the target column
# 1
# TARGET_COLUMN = 'Class'
# 2
# TARGET_COLUMN = 'Bankrupt?'
# 3
# TARGET_COLUMN = 'loan'
# 4
# TARGET_COLUMN = ' loan_status'
# 5
# TARGET_COLUMN = 'Exited'
# 6
# TARGET_COLUMN = 'Attrition'
# 7
# TARGET_COLUMN = 'Class'
# 8
# TARGET_COLUMN = 'class'
# 9
# TARGET_COLUMN = 'diagnosis'
# 10
TARGET_COLUMN = 'y'

# Load the dataset
# 1
# data = pd.read_csv('creditcard.csv')
# 2
# data = pd.read_csv('bankrupcy.csv')
# 3
# data = pd.read_csv('bank.csv')
# 4
# data = pd.read_csv('loan_approval_dataset.csv')
# 5
# data = pd.read_csv('Churn_Modelling.csv')
# 6
# data = pd.read_csv('employee_atrr.csv')
# 7
# data = pd.read_csv('Raisin_Dataset.csv')
# 8
# data = pd.read_csv('mushroom_cleaned.csv')
# 9
# data = pd.read_csv('breast-cancer.csv')
# 10
data = pd.read_csv('bank_marketing.csv')


def preprocess_data(data, target_column):
    """
    Preprocesses the data:
    - Handles missing values separately for numerical and categorical columns
    - Encodes categorical variables (if any)
    - Scales numerical features
    - Handles the target variable
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

    # For numerical columns, fill missing values with mean
    if numerical_cols:
        data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].mean())

    # For categorical columns, fill missing values with mode
    if categorical_cols:
        data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])

    # 2. Handle the Target Variable

    # If the target column is categorical, map it to numerical values
    if data[target_column].dtype == 'object':
        # Create a mapping of unique categories to integers
        unique_categories = data[target_column].unique()
        category_mapping = {category: idx for idx, category in enumerate(unique_categories)}
        data[target_column] = data[target_column].map(category_mapping)
        # Store the mapping if needed later
        # Handle any unmapped values if necessary
        data[target_column] = data[target_column].fillna(0)
    else:
        # Ensure target variable is of integer type
        data[target_column] = data[target_column].astype(int)

    # 3. Encode Categorical Variables

    if categorical_cols:
        # Use One-Hot Encoding for categorical variables
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



# Dataset Feature Extraction Function
def extract_dataset_features(data, target_column, task_type=None):
    """
    Extracts features of the dataset that might influence coreset selection methods.
    """
    features = {}

    # Determine target variable type and number of unique values
    target_values = data[target_column]
    num_unique_targets = target_values.nunique()
    target_dtype = target_values.dtype

    # Classification or Regression
    if task_type is not None:
        features['task_type'] = task_type
    else:
        if num_unique_targets <= 20:
            features['task_type'] = 'Classification'
        elif pd.api.types.is_numeric_dtype(target_values):
            features['task_type'] = 'Regression'
        else:
            features['task_type'] = 'Classification'

    # Number of Instances and Features
    features['num_instances'] = data.shape[0]
    features['num_features'] = data.shape[1] - 1  # Exclude target column

    # Data Types
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col != target_column]
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    features['num_numerical_features'] = len(numerical_cols)
    features['num_categorical_features'] = len(categorical_cols)

    # Class Balance (for classification)
    if features['task_type'] == 'Classification':
        class_counts = target_values.value_counts()
        features['num_classes'] = len(class_counts)
        features['class_balance'] = (class_counts / class_counts.sum()).to_dict()

    # Missing Values
    features['missing_values'] = data.isnull().sum().sum()

    # Dimensionality
    features['dimensionality'] = features['num_features'] / features['num_instances']

    # Correlation Matrix (only for numerical features)
    if len(numerical_cols) > 1:
        corr_matrix = data[numerical_cols].corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        features['mean_correlation'] = upper_triangle.stack().mean()
    else:
        features['mean_correlation'] = 0.0

    return features



# Coreset Selection Techniques

def no_coreset_selection(X_train, y_train):
    return X_train, y_train

def random_sampling_coreset(X_train, y_train):
    fraction = 0.1  # Use 10% of the dataset
    coreset_size = int(len(X_train) * fraction)
    coreset_size = max(1, coreset_size)
    indices = np.random.choice(len(X_train), size=coreset_size, replace=False)
    return X_train.iloc[indices], y_train.iloc[indices]

def stratified_sampling_coreset(X_train, y_train):
    fraction = 0.1  # Use 10% of the dataset
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
    fraction = 0.01  # Use 1% of the dataset
    coreset_size = int(len(X_train) * fraction)
    coreset_size = max(1, coreset_size)
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
    fraction = 0.05  # Use 5% of the dataset
    coreset_size = int(len(X_train) * fraction)
    coreset_size = max(1, coreset_size)
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_train)[:, 1]
    uncertainty = np.abs(proba - 0.5)
    indices = np.argsort(uncertainty)[:coreset_size]
    return X_train.iloc[indices], y_train.iloc[indices]

def importance_sampling_coreset(X_train, y_train):
    fraction = 0.05  # Use 5% of the dataset
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
    fraction = 0.1  # Use 10% of the dataset
    coreset_size = int(len(X_train) * fraction)
    coreset_size = max(1, coreset_size)
    n = len(X_train)
    indices = np.arange(n)
    np.random.shuffle(indices)
    selected_indices = indices[:coreset_size]
    return X_train.iloc[selected_indices], y_train.iloc[selected_indices]

def gradient_based_coreset(X_train, y_train):
    fraction = 0.05  # Use 5% of the dataset
    total_coreset_size = int(len(X_train) * fraction)
    total_coreset_size = max(2, total_coreset_size)  # Ensure at least two samples

    # Initialize the Logistic Regression model
    initial_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    initial_model.fit(X_train, y_train)

    # Predict probabilities on the training set
    y_pred_proba = initial_model.predict_proba(X_train)

    # Compute loss per sample
    epsilon = 1e-15
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
    # For multi-class, use log loss per sample
    loss_per_sample = log_loss(y_train, y_pred_proba, labels=initial_model.classes_, sample_weight=None, normalize=False)

    # Create a DataFrame for easy manipulation
    df = pd.DataFrame({
        'loss': loss_per_sample,
        'label': y_train
    })

    # Calculate the number of samples to select per class
    class_counts = y_train.value_counts()
    class_fractions = class_counts / len(y_train)
    coreset_sizes = (class_fractions * total_coreset_size).astype(int)
    # Ensure at least one sample per class
    coreset_sizes = coreset_sizes.clip(lower=1)

    # Collect indices for the coreset
    coreset_indices = []

    for cls in y_train.unique():
        cls_indices = df[df['label'] == cls].index
        cls_losses = df.loc[cls_indices, 'loss']
        cls_sorted_indices = cls_losses.sort_values(ascending=False).index
        num_samples = coreset_sizes[cls]
        selected_indices = cls_sorted_indices[:num_samples]
        coreset_indices.extend(selected_indices)

    # Ensure total coreset size matches desired size (adjust if necessary)
    coreset_indices = coreset_indices[:total_coreset_size]

    # Get the coreset samples
    X_coreset = X_train.loc[coreset_indices]
    y_coreset = y_train.loc[coreset_indices]

    return X_coreset, y_coreset


def clustering_based_coreset(X_train, y_train):
    fraction = 0.01  # Use 1% of the dataset
    coreset_size = int(len(X_train) * fraction)
    coreset_size = max(1, coreset_size)
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

    print(f"\nCoreset Method: {coreset_method.capitalize()}")
    print(f"Coreset Size: {len(X_coreset)}")

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

    # Classification Report
    cr = classification_report(y_test, y_pred, zero_division=0)

    # ROC AUC Score
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Calculate additional metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cr)
    print(f"ROC AUC Score: {roc_auc:.4f}")

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
        'f1_score': f1,
        'confusion_matrix': cm
    }

# Main function to run the experiment
def main():
    # Extract dataset features
    dataset_features = extract_dataset_features(data, TARGET_COLUMN, task_type='Classification')

    # Preprocess the data
    data_preprocessed = preprocess_data(data, TARGET_COLUMN)

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
    print("\nComparison of Coreset Methods:")
    print(results_df[['coreset_method', 'coreset_size', 'roc_auc']])

    # Identify the best coreset method based on ROC AUC (including 'None')
    best_method_idx = results_df['roc_auc'].idxmax()
    best_method = results_df.loc[best_method_idx]
    print(
        f"\nBest Coreset Selection Method: {best_method['coreset_method']} with ROC AUC Score: {best_method['roc_auc']:.4f}")

    # Create a combined table
    combined_table = dataset_features.copy()
    combined_table['best_coreset_method'] = best_method['coreset_method']
    combined_table['best_method_roc_auc'] = best_method['roc_auc']
    combined_table['best_method_precision'] = best_method['precision']
    combined_table['best_method_recall'] = best_method['recall']
    combined_table['best_method_f1_score'] = best_method['f1_score']
    combined_table['best_method_coreset_size'] = best_method['coreset_size']
    combined_table['best_method_confusion_matrix'] = best_method['confusion_matrix'].tolist()

    # Add labels based on ROC AUC score
    if best_method['roc_auc'] >= 0.9:
        combined_table['performance_label'] = 'Excellent'
    elif best_method['roc_auc'] >= 0.8:
        combined_table['performance_label'] = 'Good'
    elif best_method['roc_auc'] >= 0.7:
        combined_table['performance_label'] = 'Fair'
    else:
        combined_table['performance_label'] = 'Poor'

    # Convert to DataFrame for display
    combined_df = pd.DataFrame([combined_table])

    # Display the combined table
    print("\nCombined Table:")
    print(combined_df)

    # Optionally, save the combined table to a CSV file
    combined_df.to_csv('combined_results.csv', index=False)

    # Plotting results
    plt.figure(figsize=(12, 6))
    bar_colors = sns.color_palette('hls', len(results_df))
    bars = plt.bar(results_df['coreset_method'], results_df['roc_auc'], color=bar_colors)
    plt.xlabel('Coreset Method')
    plt.ylabel('ROC AUC Score')
    plt.title('Model Performance by Coreset Method')
    plt.ylim(0.5, 1)
    plt.grid(axis='y')

    # Annotate bars with ROC AUC scores
    for bar, score in zip(bars, results_df['roc_auc']):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.01, f'{score:.4f}', ha='center', va='bottom')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
