

#Customer Segmentation, multiclass


# Import necessary libraries
import pandas as pd
import numpy as np

# For data splitting and model training
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# For model evaluation
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_auc_score,
    roc_curve
)

# For data scaling and encoding
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier


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

# Load the dataset
data_path = '/Users/ahmed/Desktop/Machine Learning/train.csv'
data = pd.read_csv(data_path)

# Define the target column
TARGET_COLUMN = 'Segmentation'

# Data Preprocessing Function for Multiclass Classification
def preprocess_data(data, target_column):
    """
    Preprocess the dataset: scales numerical features, encodes categorical variables,
    and handles missing values.
    """
    # Drop the ID column if present
    if 'ID' in data.columns:
        data.drop('ID', axis=1, inplace=True)

    # Handle missing values by filling with mode for categorical and median for numerical columns
    for col in data.select_dtypes(include=['object']).columns:
        data[col].fillna(data[col].mode()[0], inplace=True)

    for col in data.select_dtypes(include=['float64', 'int64']).columns:
        data[col].fillna(data[col].median(), inplace=True)

    # Initialize the scaler
    scaler = StandardScaler()

    # Encode categorical variables, excluding the target column
    label_encoders = {}
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != target_column]

    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

    # Encode the target column
    target_encoder = LabelEncoder()
    data[target_column] = target_encoder.fit_transform(data[target_column].astype(str))

    # Scale numerical features
    numerical_features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    numerical_features = [col for col in numerical_features if col != target_column]
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    return data, target_encoder

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
    coreset_size = int(len(X_train) * fraction)
    
    # Ensure coreset_size is at least 1
    coreset_size = max(1, coreset_size)
    
    # Fit KMeans clustering
    kmeans = KMeans(n_clusters=coreset_size, random_state=RANDOM_STATE)
    kmeans.fit(X_train)
    labels = kmeans.labels_
    
    # Select one sample from each cluster
    indices = [np.where(labels == i)[0][0] for i in range(coreset_size)]
    
    # Ensure y_train is a Pandas Series
    y_train = pd.Series(y_train).reset_index(drop=True)
    
    return X_train.iloc[indices], y_train.iloc[indices]




from sklearn.linear_model import LogisticRegression
from sklearn.cluster import MiniBatchKMeans

def uncertainty_sampling_coreset(X_train, y_train, fraction=0.05):
    """
    Selects samples with the highest uncertainty based on Logistic Regression probabilities.
    """
    coreset_size = int(len(X_train) * fraction)
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, multi_class='multinomial')
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_train)
    uncertainty = 1 - np.max(proba, axis=1)  # Uncertainty is highest where max probability is lowest
    indices = np.argsort(uncertainty)[-coreset_size:]
    y_train = pd.Series(y_train).reset_index(drop=True)
    return X_train.iloc[indices], y_train.iloc[indices]

def minibatch_kmeans_coreset(X_train, y_train, fraction=0.05):
    """
    Selects samples using MiniBatchKMeans clustering.
    """
    coreset_size = int(len(X_train) * fraction)
    coreset_size = max(1, coreset_size)
    kmeans = MiniBatchKMeans(n_clusters=coreset_size, random_state=RANDOM_STATE, batch_size=1000)
    kmeans.fit(X_train)
    labels = kmeans.labels_
    indices = [np.where(labels == i)[0][0] for i in range(coreset_size)]
    y_train = pd.Series(y_train).reset_index(drop=True)
    return X_train.iloc[indices], y_train.iloc[indices]

def reservoir_sampling_coreset(X_train, y_train, fraction=0.05):
    """
    Selects samples using reservoir sampling.
    """
    coreset_size = int(len(X_train) * fraction)
    coreset_size = max(1, coreset_size)
    indices = []
    for i, _ in enumerate(X_train):
        if i < coreset_size:
            indices.append(i)
        else:
            j = np.random.randint(0, i + 1)
            if j < coreset_size:
                indices[j] = i
    y_train = pd.Series(y_train).reset_index(drop=True)
    return X_train.iloc[indices], y_train.iloc[indices]

def gradient_based_coreset(X_train, y_train, fraction=0.05):
    """
    Selects samples based on the highest gradient magnitudes (loss impact).
    """
    coreset_size = int(len(X_train) * fraction)
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, multi_class='multinomial')
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_train)
    epsilon = 1e-15
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
    y_train_one_hot = pd.get_dummies(y_train).to_numpy()
    loss_per_sample = -np.sum(y_train_one_hot * np.log(y_pred_proba), axis=1)
    indices = np.argsort(loss_per_sample)[-coreset_size:]
    y_train = pd.Series(y_train).reset_index(drop=True)
    return X_train.iloc[indices], y_train.iloc[indices]






# Train and Evaluate Function
def train_and_evaluate(X_train, y_train, X_test, y_test, coreset_method):
    """
    Applies coreset selection, trains the HistGradientBoostingClassifier model, and evaluates it.
    """
    # Apply the coreset method
    X_coreset, y_coreset = coreset_method(X_train, y_train)

    # Train the model
    model = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    model.fit(X_coreset, y_coreset)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy
# Main function to run the experiment
def main():
    # Preprocess the data
    data_preprocessed, target_encoder = preprocess_data(data, TARGET_COLUMN)
    
    # Split the data into features (X) and target (y)
    X = data_preprocessed.drop(TARGET_COLUMN, axis=1)
    y = np.ravel(data_preprocessed[TARGET_COLUMN])

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
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.show()

if __name__ == "__main__":
    main()
