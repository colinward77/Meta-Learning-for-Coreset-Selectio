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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import logging

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

#np.random.seed(1)

# Flag to control whether plots are displayed
SHOW_PLOTS = True  # Set to True to display plots, False for data collection

TASK_TYPE = 'Multi-Class Classification'

HAS_NULL_CLASS = False

# 1
#TARGET_COLUMN = 'admission'
#DATASET_NAME = 'MBA.csv'
#data = pd.read_csv('MultiClass_Datasets/MBA.csv')
#HAS_NULL_CLASS = True

# 2
#TARGET_COLUMN = 'esrb_rating'
#DATASET_NAME = 'Video_games_esrb_rating.csv'
#data = pd.read_csv('MultiClass_Datasets/Video_games_esrb_rating.csv')

# 3
#TARGET_COLUMN = 'species'
#DATASET_NAME = 'fish_data.csv'
#data = pd.read_csv('MultiClass_Datasets/fish_data.csv')

# 4
#TARGET_COLUMN = 'Index'
#DATASET_NAME = 'bmi.csv'
#data = pd.read_csv('MultiClass_Datasets/bmi.csv')

# 5
#TARGET_COLUMN = 'Drug'
#DATASET_NAME = 'drug200.csv'
#data = pd.read_csv('MultiClass_Datasets/drug200.csv')

# 6
#TARGET_COLUMN = 'salary'
#DATASET_NAME = 'Employee Attrition.csv'
#data = pd.read_csv('MultiClass_Datasets/Employee Attrition.csv')

# 7
#TARGET_COLUMN = 'Class'
#DATASET_NAME = 'Dry_Beans_Dataset.csv'
#data = pd.read_csv('MultiClass_Datasets/Dry_Beans_Dataset.csv')

# 8 - reduced features and rows
#TARGET_COLUMN = 'Credit_Score'
#DATASET_NAME = 'Credit_score_cleaned_data.csv'
#data = pd.read_csv('MultiClass_Datasets/Credit_score_cleaned_data.csv', nrows=50000)

# 9 - reduced rows
#TARGET_COLUMN = 'Target'
#DATASET_NAME = 'MultiClass_1m.csv'
#data = pd.read_csv('MultiClass_Datasets/MultiClass_1m.csv', nrows=100000)

# 10
#TARGET_COLUMN = 'label'
#DATASET_NAME = 'edm_genre.csv'
#data = pd.read_csv('MultiClass_Datasets/edm_genre.csv')

# 11
#TARGET_COLUMN = 'review_score'
#DATASET_NAME = 'cleaned_reviews.csv'
#data = pd.read_csv('MultiClass_Datasets/cleaned_reviews.csv')

# 12
#TARGET_COLUMN = 'Default'
#DATASET_NAME = 'creditdefault_cleaned.csv'
#data = pd.read_csv('MultiClass_Datasets/creditdefault_cleaned.csv')

# 13
#TARGET_COLUMN = 'Drug'
#DATASET_NAME = 'DrugOther.csv'
#data = pd.read_csv('MultiClass_Datasets/DrugOther.csv')

# 14
#TARGET_COLUMN = 'class'
#DATASET_NAME = 'dermatology_database_1.csv'
#data = pd.read_csv('MultiClass_Datasets/dermatology_database_1.csv')

# 15
#TARGET_COLUMN = 'species'
#DATASET_NAME = 'IRIS.csv'
#data = pd.read_csv('MultiClass_Datasets/IRIS.csv')

#----following not added to use for model testing --

# 16
#TARGET_COLUMN = 'class'
#DATASET_NAME = 'bodyPerformance.csv'
#data = pd.read_csv('MetaModel_Test_Sets/bodyPerformance.csv')

# 17
#TARGET_COLUMN = 'NObeyesdad'
#DATASET_NAME = 'obesitypred_eatinghabits_physical.csv'
#data = pd.read_csv('MetaModel_Test_Sets/obesitypred_eatinghabits_physical.csv')

# 18
TARGET_COLUMN = 'genre'
DATASET_NAME = 'all_audio_features_modified.csv'
data = pd.read_csv('MetaModel_Test_Sets/all_audio_features_modified.csv')


def preprocess_data(data, target_column, handle_unexpected='drop'):
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

    if target_column in numerical_cols:
        numerical_cols.remove(target_column)
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)

    logger.info(f"Numerical columns: {numerical_cols}")
    logger.info(f"Categorical columns: {categorical_cols}")

    # Impute numerical columns
    if numerical_cols:
        imputer_num = SimpleImputer(strategy='mean')
        data[numerical_cols] = imputer_num.fit_transform(data[numerical_cols])
        logger.info("Imputed missing values in numerical columns with mean.")
    else:
        logger.info("No numerical columns to impute.")

    # Impute categorical columns
    if categorical_cols:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        data[categorical_cols] = imputer_cat.fit_transform(data[categorical_cols])
        logger.info("Imputed missing values in categorical columns with mode.")
    else:
        logger.info("No categorical columns to impute.")

    if HAS_NULL_CLASS:
        data[target_column] = data[target_column].fillna('null_class')

    le = LabelEncoder()
    data[target_column] = le.fit_transform(data[target_column])
    logger.info("Encoded target column using LabelEncoder for multi-class classification.")

    # One-hot encode categorical features
    if categorical_cols:
        data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
        logger.info(f"One-Hot Encoded categorical columns: {categorical_cols}")
    else:
        logger.info("No categorical columns to encode.")

    # Scale numerical features
    if numerical_cols:
        scaler = StandardScaler()
        data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
        logger.info("Scaled numerical features using StandardScaler.")
    else:
        logger.info("No numerical columns to scale.")

    return data


def extract_dataset_features(data, target_column):
    features = {}

    features['dataset_name'] = DATASET_NAME
    features['task_type'] = TASK_TYPE
    features['num_instances'] = data.shape[0]
    features['num_features'] = data.shape[1] - 1  # Exclude target column

    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numerical_cols:
        numerical_cols.remove(target_column)
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    features['num_numerical_features'] = len(numerical_cols)
    features['num_categorical_features'] = len(categorical_cols)

    if features['num_numerical_features'] > 0 and features['num_categorical_features'] > 0:
        features['feature_type'] = 'Mixed'
    elif features['num_numerical_features'] > 0:
        features['feature_type'] = 'Numerical'
    else:
        features['feature_type'] = 'Categorical'

    target_values = data[target_column]
    class_counts = target_values.value_counts()
    features['num_classes'] = len(class_counts)
    class_balance = (class_counts / class_counts.sum()).to_dict()
    class_balance = {str(k): v for k, v in class_balance.items()}
    features['class_balance'] = class_balance

    majority_class = class_counts.max()
    minority_class = class_counts.min()
    features['imbalance_ratio'] = majority_class / minority_class

    features['dimensionality'] = features['num_features'] / features['num_instances']

    if len(numerical_cols) > 1:
        corr_matrix = data[numerical_cols].corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        features['mean_correlation'] = upper_triangle.stack().mean()
        features['max_correlation'] = upper_triangle.stack().max()
        high_corr_pairs = upper_triangle.stack().apply(lambda x: x > 0.8).sum()
        features['feature_redundancy'] = high_corr_pairs
    else:
        features['mean_correlation'] = 0.0
        features['max_correlation'] = 0.0
        features['feature_redundancy'] = 0

    if numerical_cols:
        feature_means = data[numerical_cols].mean()
        feature_variances = data[numerical_cols].var()
        features['mean_of_means'] = feature_means.mean()
        features['variance_of_means'] = feature_means.var()
        features['mean_of_variances'] = feature_variances.mean()
        features['variance_of_variances'] = feature_variances.var()
        features['mean_skewness'] = data[numerical_cols].skew().mean()
        features['mean_kurtosis'] = data[numerical_cols].kurtosis().mean()
    else:
        features['mean_of_means'] = 0.0
        features['variance_of_means'] = 0.0
        features['mean_of_variances'] = 0.0
        features['variance_of_variances'] = 0.0
        features['mean_skewness'] = 0.0
        features['mean_kurtosis'] = 0.0

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

    total_elements = data.shape[0] * data.shape[1]
    zero_elements = (data == 0).sum().sum()
    features['data_sparsity'] = zero_elements / total_elements

    return features


def ensure_class_coverage(X_coreset, y_coreset, X_train, y_train):
    """
    Ensures that all classes present in y_train are also in y_coreset.
    If any are missing, add one sample from each missing class.
    """
    classes_in_full = np.unique(y_train)
    classes_in_coreset = np.unique(y_coreset)
    missing_classes = set(classes_in_full) - set(classes_in_coreset)

    for cls in missing_classes:
        cls_indices = y_train[y_train == cls].index
        if len(cls_indices) > 0:
            chosen_idx = np.random.choice(cls_indices)
            X_coreset = pd.concat([X_coreset, X_train.loc[[chosen_idx]]])
            y_coreset = pd.concat([y_coreset, y_train.loc[[chosen_idx]]])

    X_coreset = X_coreset.reset_index(drop=True)
    y_coreset = y_coreset.reset_index(drop=True)
    return X_coreset, y_coreset


def no_coreset_selection(X_train, y_train):
    # Returns full dataset, no need to ensure coverage
    return X_train, y_train

def random_sampling_coreset(X_train, y_train):
    fraction = 0.2
    coreset_size = int(len(X_train) * fraction)
    coreset_size = max(1, coreset_size)
    indices = np.random.choice(len(X_train), size=coreset_size, replace=False)
    X_coreset = X_train.iloc[indices]
    y_coreset = y_train.iloc[indices]

    # Ensure class coverage
    X_coreset, y_coreset = ensure_class_coverage(X_coreset, y_coreset, X_train, y_train)
    return X_coreset, y_coreset

def stratified_sampling_coreset(X_train, y_train):
    # Stratified should ensure coverage by design, but only if coreset_size > number_of_classes
    fraction = 0.2
    coreset_size = int(len(X_train) * fraction)
    coreset_size = max(1, coreset_size)
    X_coreset, _, y_coreset, _ = train_test_split(
        X_train, y_train,
        train_size=coreset_size,
        stratify=y_train,
        random_state=RANDOM_STATE
    )
    # Typically ensures coverage if coreset_size >= number_of_classes
    return X_coreset, y_coreset

def kmeans_clustering_coreset(X_train, y_train):
    if len(X_train) > 100000:
        fraction = 0.01
    else:
        fraction = 0.05
    coreset_size = int(len(X_train) * fraction)
    coreset_size = max(1, coreset_size)

    kmeans = KMeans(n_clusters=coreset_size, random_state=RANDOM_STATE)
    kmeans.fit(X_train)
    labels = kmeans.labels_

    X_list = []
    y_list = []
    for i in range(coreset_size):
        indices_in_cluster = np.where(labels == i)[0]
        if len(indices_in_cluster) == 0:
            continue
        idx = np.random.choice(indices_in_cluster)
        X_list.append(X_train.iloc[idx])
        y_list.append(y_train.iloc[idx])
    X_coreset = pd.DataFrame(X_list)
    y_coreset = pd.Series(y_list)

    X_coreset, y_coreset = ensure_class_coverage(X_coreset, y_coreset, X_train, y_train)
    return X_coreset, y_coreset

def uncertainty_sampling_coreset(X_train, y_train):
    fraction = 0.1
    coreset_size = int(len(X_train) * fraction)
    coreset_size = max(1, coreset_size)
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_train)
    sorted_proba = np.sort(proba, axis=1)
    uncertainty = sorted_proba[:, -1] - sorted_proba[:, -2] if proba.shape[1] > 1 else np.abs(proba[:, 0] - 0.5)
    indices = np.argsort(uncertainty)[:coreset_size]
    X_coreset = X_train.iloc[indices]
    y_coreset = y_train.iloc[indices]

    X_coreset, y_coreset = ensure_class_coverage(X_coreset, y_coreset, X_train, y_train)
    return X_coreset, y_coreset

def importance_sampling_coreset(X_train, y_train):
    fraction = 0.1
    coreset_size = int(len(X_train) * fraction)
    coreset_size = max(1, coreset_size)
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_train)
    sorted_proba = np.sort(proba, axis=1)
    importance = sorted_proba[:, -1] - sorted_proba[:, -2] if proba.shape[1] > 1 else np.abs(proba[:, 0] - 0.5)
    importance = np.clip(importance, 1e-6, None)
    importance /= importance.sum()
    indices = np.random.choice(len(X_train), size=coreset_size, replace=False, p=importance)
    X_coreset = X_train.iloc[indices]
    y_coreset = y_train.iloc[indices]

    X_coreset, y_coreset = ensure_class_coverage(X_coreset, y_coreset, X_train, y_train)
    return X_coreset, y_coreset

def reservoir_sampling_coreset(X_train, y_train):
    fraction = 0.2
    coreset_size = int(len(X_train) * fraction)
    coreset_size = max(1, coreset_size)
    n = len(X_train)
    indices = np.arange(n)
    np.random.shuffle(indices)
    selected_indices = indices[:coreset_size]
    X_coreset = X_train.iloc[selected_indices]
    y_coreset = y_train.iloc[selected_indices]

    X_coreset, y_coreset = ensure_class_coverage(X_coreset, y_coreset, X_train, y_train)
    return X_coreset, y_coreset

def gradient_based_coreset(X_train, y_train):
    fraction = 0.1
    total_coreset_size = int(len(X_train) * fraction)
    total_coreset_size = max(2, total_coreset_size)

    initial_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    initial_model.fit(X_train, y_train)

    y_pred_proba = initial_model.predict_proba(X_train)
    epsilon = 1e-15
    y_one_hot = pd.get_dummies(y_train)
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
    loss_per_sample = -(y_one_hot * np.log(y_pred_proba)).sum(axis=1)

    df = pd.DataFrame({
        'loss': loss_per_sample,
        'label': y_train
    }, index=X_train.index)

    class_counts = y_train.value_counts()
    class_fractions = class_counts / len(y_train)
    coreset_sizes = (class_fractions * total_coreset_size).astype(int).clip(lower=1)

    coreset_indices = []
    for cls in y_train.unique():
        cls_indices = df[df['label'] == cls].index
        cls_losses = df.loc[cls_indices, 'loss']
        cls_sorted_indices = cls_losses.sort_values(ascending=True).index
        num_samples = coreset_sizes[cls]
        selected_indices = cls_sorted_indices[:num_samples]
        coreset_indices.extend(selected_indices)
    coreset_indices = coreset_indices[:total_coreset_size]

    X_coreset = X_train.loc[coreset_indices].reset_index(drop=True)
    y_coreset = y_train.loc[coreset_indices].reset_index(drop=True)

    # Even though we tried to distribute samples by class, ensure_class_coverage to be safe
    X_coreset, y_coreset = ensure_class_coverage(X_coreset, y_coreset, X_train, y_train)
    return X_coreset, y_coreset

def clustering_based_coreset(X_train, y_train):
    if len(X_train) > 100000:
        fraction = 0.01
    else:
        fraction = 0.05
    coreset_size = int(len(X_train) * fraction)
    coreset_size = max(1, coreset_size)

    kmeans = MiniBatchKMeans(n_clusters=coreset_size, random_state=RANDOM_STATE, batch_size=10000)
    kmeans.fit(X_train)
    labels = kmeans.labels_

    X_list = []
    y_list = []
    for i in range(coreset_size):
        indices_in_cluster = np.where(labels == i)[0]
        if len(indices_in_cluster) == 0:
            continue
        idx = np.random.choice(indices_in_cluster)
        X_list.append(X_train.iloc[idx])
        y_list.append(y_train.iloc[idx])
    X_coreset = pd.DataFrame(X_list)
    y_coreset = pd.Series(y_list)

    X_coreset, y_coreset = ensure_class_coverage(X_coreset, y_coreset, X_train, y_train)
    return X_coreset, y_coreset

def train_and_evaluate(X_train, y_train, X_test, y_test, coreset_method):
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

    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_coreset, y_coreset)

    y_pred = lr.predict(X_test)
    y_pred_proba = lr.predict_proba(X_test)

    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')

    if SHOW_PLOTS:
        # Skipping plotting individual class ROC curves for simplicity
        pass

    return {
        'coreset_method': coreset_method.capitalize(),
        'coreset_size': len(X_coreset),
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def main():

    if HAS_NULL_CLASS:
        data_preprocessed = preprocess_data(data, TARGET_COLUMN)
        dataset_features = extract_dataset_features(data_preprocessed, TARGET_COLUMN)
    else:
        dataset_features = extract_dataset_features(data, TARGET_COLUMN)
        data_preprocessed = preprocess_data(data, TARGET_COLUMN)


    X = data_preprocessed.drop(TARGET_COLUMN, axis=1)
    y = data_preprocessed[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    coreset_methods = [
        'none', 'random', 'stratified', 'kmeans', 'uncertainty',
        'importance', 'reservoir', 'gradient', 'clustering'
    ]

    results = []
    for method in coreset_methods:
        result = train_and_evaluate(X_train, y_train, X_test, y_test, method)
        results.append(result)

    results_df = pd.DataFrame(results)

    results_df_non_none = results_df[results_df['coreset_method'] != 'None']
    if not results_df_non_none.empty:
        best_method_idx = results_df_non_none['roc_auc'].idxmax()
        best_method = results_df_non_none.loc[best_method_idx]
    else:
        best_method = results_df.iloc[0]

    baseline_roc_auc = results_df[results_df['coreset_method'] == 'None']['roc_auc'].values[0]
    performance_gain = best_method['roc_auc'] - baseline_roc_auc

    evaluation_metrics = results_df.copy()
    evaluation_metrics.insert(0, 'dataset_name', DATASET_NAME)
    evaluation_metrics.insert(1, 'task_type', TASK_TYPE)
    evaluation_metrics['baseline_roc_auc'] = baseline_roc_auc
    evaluation_metrics['performance_gain'] = evaluation_metrics['roc_auc'] - baseline_roc_auc

    columns_order = [
        'dataset_name', 'task_type', 'coreset_method', 'coreset_size',
        'roc_auc', 'precision', 'recall', 'f1_score',
        'baseline_roc_auc', 'performance_gain'
    ]

    for col in columns_order:
        if col not in evaluation_metrics.columns:
            evaluation_metrics[col] = None

    evaluation_metrics = evaluation_metrics[columns_order]

    print("\nEvaluation Metrics for All Coreset Methods:")
    print(evaluation_metrics[columns_order])

    if SHOW_PLOTS:
        plt.figure(figsize=(12, 6))
        bar_colors = sns.color_palette('hls', len(results_df))
        bars = plt.bar(results_df['coreset_method'], results_df['roc_auc'], color=bar_colors)
        plt.xlabel('Coreset Method')
        plt.ylabel('ROC AUC Score')
        plt.title(f'Model Performance by Coreset Method on {DATASET_NAME}')
        plt.ylim(0.5, 1)
        plt.grid(axis='y')

        for bar, score in zip(bars, results_df['roc_auc']):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.005, f'{score:.4f}', ha='center', va='bottom')

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        evaluation_metrics_file = 'evaluation_metrics_multi_classification.csv'
        if not os.path.exists(evaluation_metrics_file) or os.stat(evaluation_metrics_file).st_size == 0:
            header_eval = True
        else:
            header_eval = False
        evaluation_metrics.to_csv(
            evaluation_metrics_file,
            mode='a',
            index=False,
            header=header_eval
        )

        meta_dataset_file = 'meta_dataset.csv'
        meta_dataset_columns_order = [
            'dataset_name', 'task_type', 'num_instances', 'num_features',
            'num_numerical_features', 'num_categorical_features', 'feature_type',
            'num_classes', 'class_balance', 'imbalance_ratio',
            'dimensionality', 'mean_correlation',
            'max_correlation', 'feature_redundancy', 'mean_of_means',
            'variance_of_means', 'mean_of_variances', 'variance_of_variances',
            'mean_skewness', 'mean_kurtosis', 'outlier_percentage', 'data_sparsity',
            'best_coreset_method'
        ]

        dataset_features['best_coreset_method'] = best_method['coreset_method']
        meta_dataset_df = pd.DataFrame([dataset_features])
        for col in meta_dataset_columns_order:
            if col not in meta_dataset_df.columns:
                meta_dataset_df[col] = None
        meta_dataset_df = meta_dataset_df[meta_dataset_columns_order]

        if not os.path.exists(meta_dataset_file) or os.stat(meta_dataset_file).st_size == 0:
            header_meta = True
        else:
            header_meta = False

        meta_dataset_df.to_csv(
            meta_dataset_file,
            mode='a',
            index=False,
            header=header_meta
        )

if __name__ == "__main__":
    main()