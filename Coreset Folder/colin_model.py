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
   roc_auc_score,
   roc_curve,
   precision_score,
   recall_score,
   f1_score
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
TARGET_COLUMN = 'Class'
# 2
#TARGET_COLUMN = 'Bankrupt?'


# Load the dataset
# 1
data = pd.read_csv('creditcard.csv')
# 2
#data = pd.read_csv('bankruptcy.csv')


# Data Preprocessing Function
def preprocess_data(data):
   """
   Scales numerical features and encodes categorical variables, excluding the target column.
   """


   # Initialize the scaler
   scaler = StandardScaler()


   # Scale 'Amount' and 'Time' features if they exist
   if 'Amount' in data.columns:
       data['scaled_amount'] = scaler.fit_transform(data[['Amount']])
       data.drop('Amount', axis=1, inplace=True)
   if 'Time' in data.columns:
       data['scaled_time'] = scaler.fit_transform(data[['Time']])
       data.drop('Time', axis=1, inplace=True)


   # Encode categorical variables if any, excluding the target column
   label_encoders = {}
   categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
   categorical_cols = [col for col in categorical_cols if col != TARGET_COLUMN]
   for col in categorical_cols:
       le = LabelEncoder()
       data[col] = le.fit_transform(data[col].astype(str))
       label_encoders[col] = le


   # Convert target column to integer if necessary
   if data[TARGET_COLUMN].dtype != np.int64:
       data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(int)


   # Standardize numerical features, excluding the target column
   numerical_features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
   numerical_features = [col for col in numerical_features if col != TARGET_COLUMN]
   data[numerical_features] = scaler.fit_transform(data[numerical_features])


   # Rearrange columns
   cols = data.columns.tolist()
   if 'scaled_amount' in cols and 'scaled_time' in cols:
       cols = ['scaled_amount', 'scaled_time'] + [col for col in cols if col not in ['scaled_amount', 'scaled_time']]
   data = data[cols]


   # Check unique values in the target column
   print(f"Unique values in target column '{TARGET_COLUMN}':", data[TARGET_COLUMN].unique())


   return data




# Dataset Feature Extraction Function
def extract_dataset_features(data, target_column):
   """
   Extracts features of the dataset that might influence coreset selection methods.
   """
   features = {}
   # Classification or Regression
   if data[target_column].nunique() <= 20 and data[target_column].dtype in [np.int64, np.int32]:
       features['task_type'] = 'Classification'
   else:
       features['task_type'] = 'Regression'


   # Number of Instances and Features
   features['num_instances'] = data.shape[0]
   features['num_features'] = data.shape[1] - 1  # Exclude target column


   # Data Types
   features['num_numerical_features'] = len(data.select_dtypes(include=['float64', 'int64']).columns.tolist())
   features['num_categorical_features'] = len(data.select_dtypes(include=['object', 'category']).columns.tolist())


   # Class Balance (for classification)
   if features['task_type'] == 'Classification':
       class_counts = data[target_column].value_counts()
       features['num_classes'] = len(class_counts)
       features['class_balance'] = (class_counts / class_counts.sum()).to_dict()


   # Missing Values
   features['missing_values'] = data.isnull().sum().sum()


   # Dimensionality
   features['dimensionality'] = features['num_features'] / features['num_instances']


   # Correlation Matrix (only for numerical features)
   numerical_features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
   corr_matrix = data[numerical_features].corr().abs()
   upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
   features['mean_correlation'] = upper_triangle.stack().mean()


   return features


# Coreset Selection Techniques


def no_coreset_selection(X_train, y_train):
   return X_train, y_train


#no run
def random_sampling_coreset(X_train, y_train):
   fraction = 0.1  # Use 10% of the dataset
   coreset_size = int(len(X_train) * fraction)
   coreset_size = max(1, coreset_size)
   indices = np.random.choice(len(X_train), size=coreset_size, replace=False)
   return X_train.iloc[indices], y_train.iloc[indices]

#no run
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

#run
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


#run
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

#run
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
   fraction = 0.05  # Use 10% of the dataset
   coreset_size = int(len(X_train) * fraction)
   coreset_size = max(1, coreset_size)
   n = len(X_train)
   indices = np.arange(n)
   np.random.shuffle(indices)
   selected_indices = indices[:coreset_size]
   return X_train.iloc[selected_indices], y_train.iloc[selected_indices]


def gradient_based_coreset(X_train, y_train):
   fraction = 0.05  # Use 5% of the dataset
   coreset_size = int(len(X_train) * fraction)
   coreset_size = max(1, coreset_size)
   initial_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
   initial_model.fit(X_train, y_train)
   y_pred_proba = initial_model.predict_proba(X_train)[:, 1]
   epsilon = 1e-15
   y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
   loss_per_sample = - (y_train * np.log(y_pred_proba) + (1 - y_train) * np.log(1 - y_pred_proba))
   coreset_indices = np.argsort(loss_per_sample)[-coreset_size:]
   return X_train.iloc[coreset_indices], y_train.iloc[coreset_indices]


def clustering_based_coreset(X_train, y_train):
   fraction = 0.05  # Use 1% of the dataset
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
   #plt.show()


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
   # Preprocess the data
   data_preprocessed = preprocess_data(data)


   # Extract dataset features
   dataset_features = extract_dataset_features(data_preprocessed, TARGET_COLUMN)


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


   # Identify the best coreset method based on ROC AUC (excluding 'none')
   coreset_only_df = results_df[results_df['coreset_method'] != 'None']
   best_method_idx = coreset_only_df['roc_auc'].idxmax()
   best_method = coreset_only_df.loc[best_method_idx]
   print(f"\nBest Coreset Selection Method: {best_method['coreset_method']} with ROC AUC Score: {best_method['roc_auc']:.4f}")


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
   bar_colors = sns.color_palette('husl', len(results_df))
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

