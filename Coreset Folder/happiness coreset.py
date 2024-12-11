#https://www.kaggle.com/datasets/priyanshusethi/happiness-classification-dataset

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('happydata.csv')

# Preprocessing
def preprocess_data(data):
    # Standardize numerical features excluding 'happy'
    scaler = StandardScaler()
    features = data.drop('happy', axis=1).columns
    data[features] = scaler.fit_transform(data[features])
    return data

data = preprocess_data(data)

# Features and target
X = data.drop('happy', axis=1)
y = data['happy']

# Coreset selection methods
def no_coreset_selection(X_train, y_train, coreset_size=None):
    return X_train, y_train

def random_sampling_coreset(X_train, y_train, coreset_size):
    indices = np.random.choice(len(X_train), size=coreset_size, replace=False)
    return X_train.iloc[indices], y_train.iloc[indices]

def stratified_sampling_coreset(X_train, y_train, coreset_size):
    X_coreset, _, y_coreset, _ = train_test_split(X_train, y_train, train_size=coreset_size, stratify=y_train, random_state=42)
    return X_coreset, y_coreset

def kmeans_clustering_coreset(X_train, y_train, coreset_size):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=coreset_size, random_state=42)
    kmeans.fit(X_train)
    centers = pd.DataFrame(kmeans.cluster_centers_, columns=X_train.columns)
    
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=1).fit(X_train)
    nearest_indices = nbrs.kneighbors(centers, return_distance=False).flatten()
    y_coreset = y_train.iloc[nearest_indices]
    
    return centers, y_coreset

def uncertainty_sampling_coreset(X_train, y_train, coreset_size):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_train)[:, 1]
    uncertainty = np.abs(proba - 0.5)
    indices = np.argsort(uncertainty)[:coreset_size]
    return X_train.iloc[indices], y_train.iloc[indices]

def importance_sampling_coreset(X_train, y_train, coreset_size):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_train)[:, 1]
    importance = np.abs(proba - 0.5)
    importance /= importance.sum()
    indices = np.random.choice(len(X_train), size=coreset_size, replace=False, p=importance)
    return X_train.iloc[indices], y_train.iloc[indices]

def reservoir_sampling_coreset(X_train, y_train, coreset_size):
    n = len(X_train)
    indices = list(range(n))
    np.random.shuffle(indices)
    selected_indices = indices[:coreset_size]
    return X_train.iloc[selected_indices], y_train.iloc[selected_indices]

def train_and_evaluate(X_train, y_train, X_test, y_test, coreset_method, coreset_size=None):
    coreset_methods = {
        'none': no_coreset_selection,
        'random': random_sampling_coreset,
        'stratified': stratified_sampling_coreset,
        'kmeans': kmeans_clustering_coreset,
        'uncertainty': uncertainty_sampling_coreset,
        'importance': importance_sampling_coreset,
        'reservoir': reservoir_sampling_coreset
    }
    
    X_coreset, y_coreset = coreset_methods[coreset_method](X_train, y_train, coreset_size)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_coreset, y_coreset)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nCoreset Method: {coreset_method}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(cr)
    print(f"ROC AUC Score: {roc_auc:.4f}")
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{coreset_method} (AUC = {roc_auc:.4f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    #plt.show()
    
    return {'coreset_method': coreset_method, 'roc_auc': roc_auc}

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Evaluate all methods
coreset_methods = ['none', 'random', 'stratified', 'kmeans', 'uncertainty', 'importance', 'reservoir']
coreset_size = 10  # Example size

results = []
for method in coreset_methods:
    results.append(train_and_evaluate(X_train, y_train, X_test, y_test, method, coreset_size))

# Compare results
results_df = pd.DataFrame(results)
print("\nComparison of Coreset Methods:")
print(results_df)
