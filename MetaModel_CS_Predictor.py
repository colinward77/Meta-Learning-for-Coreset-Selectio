import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin


# 1 -res
#USER_DATASET_PATH = 'MetaModel_Test_Sets/Customertravel.csv'
#USER_TASK_TYPE = 'Binary Classification'
#USER_DATASET_TARGET_COLUMN = 'Target'

# 2 -grad
#USER_DATASET_PATH = 'MetaModel_Test_Sets/social_ads.csv'
#USER_TASK_TYPE = 'Binary Classification'
#USER_DATASET_TARGET_COLUMN = 'Purchased'

# 3 -res
#USER_DATASET_PATH = 'MetaModel_Test_Sets/plant_growth_data.csv'
#USER_TASK_TYPE = 'Binary Classification'
#USER_DATASET_TARGET_COLUMN = 'Growth_Milestone'

# 4 -grad
#USER_DATASET_PATH = 'MetaModel_Test_Sets/diabetes_dataset.csv'
#USER_TASK_TYPE = 'Binary Classification'
#USER_DATASET_TARGET_COLUMN = 'Outcome'

# 5 -res
#USER_DATASET_PATH = 'MetaModel_Test_Sets/customer_purchase_data.csv'
#USER_TASK_TYPE = 'Binary Classification'
#USER_DATASET_TARGET_COLUMN = 'PurchaseStatus'

# 6 -grad
#USER_DATASET_PATH = 'MetaModel_Test_Sets/happydata.csv'
#USER_TASK_TYPE = 'Binary Classification'
#USER_DATASET_TARGET_COLUMN = 'happy'

# 7 -strat
#USER_DATASET_PATH = 'MetaModel_Test_Sets/riceClassification.csv'
#USER_TASK_TYPE = 'Binary Classification'
#USER_DATASET_TARGET_COLUMN = 'Class'

# 8 - rando
#USER_DATASET_PATH = 'MetaModel_Test_Sets/CO2 Emissions.csv'
#USER_TASK_TYPE = 'Regression'
#USER_DATASET_TARGET_COLUMN = 'CO2 Emissions(g/km)'

# 9 - res
USER_DATASET_PATH = 'MetaModel_Test_Sets/all_audio_features_modified.csv'
USER_TASK_TYPE = 'Multi-Class Classification'
USER_DATASET_TARGET_COLUMN = 'genre'

# Path to the meta-data CSV file
META_DATA_PATH = 'meta_dataset.csv'


def extract_dataset_features(data, target_column, dataset_name, task_type):
    features = {}
    DATASET_NAME = dataset_name
    TASK_TYPE = task_type

    features['dataset_name'] = DATASET_NAME
    features['task_type'] = TASK_TYPE
    features['num_instances'] = data.shape[0]
    features['num_features'] = data.shape[1] - 1

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
    minority_class = class_counts.min() if len(class_counts) > 1 else majority_class
    features['imbalance_ratio'] = majority_class / minority_class if minority_class != 0 else 1.0

    features['dimensionality'] = features['num_features'] / features['num_instances'] if features['num_instances'] > 0 else 0

    # Correlation, statistical properties, etc. remain the same
    numerical_cols = [c for c in numerical_cols if c in data.columns]  # just a safety check
    if len(numerical_cols) > 1:
        corr_matrix = data[numerical_cols].corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        features['mean_correlation'] = upper_triangle.stack().mean() if not upper_triangle.stack().empty else 0.0
        features['max_correlation'] = upper_triangle.stack().max() if not upper_triangle.stack().empty else 0.0
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
        features['outlier_percentage'] = outliers / total_values if total_values > 0 else 0.0
    else:
        features['outlier_percentage'] = 0.0

    total_elements = data.shape[0] * data.shape[1]
    zero_elements = (data == 0).sum().sum()
    features['data_sparsity'] = zero_elements / total_elements if total_elements > 0 else 0.0

    return features


class TaskTypeWeighter(BaseEstimator, TransformerMixin):
    def __init__(self, task_type_col='task_type'):
        self.task_type_col = task_type_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        def indicator_func(t):
            if t == 'Regression':
                return 10
            elif t == 'Multi-Class Classification':
                return 5
            else:
                return 1
        X['task_type_indicator'] = X[self.task_type_col].apply(indicator_func)
        return X


class ClassBalanceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column='class_balance'):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        numeric_min = []
        numeric_max = []

        for i, val in enumerate(X[self.column]):
            task_type = X.iloc[i]['task_type']
            if task_type == 'Regression':
                # Already handled by pre-processing (no NaNs)
                # If still encountered, default to 1.0
                numeric_min.append(1.0)
                numeric_max.append(1.0)
                continue

            if isinstance(val, dict):
                dict_val = val
            elif isinstance(val, str):
                val = val.strip()
                if val == "{}" or val == "":
                    dict_val = {}
                else:
                    dict_val = eval(val)
            else:
                dict_val = {}

            if len(dict_val) == 0:
                numeric_min.append(1.0)
                numeric_max.append(1.0)
            else:
                proportions = list(dict_val.values())
                numeric_min.append(min(proportions))
                numeric_max.append(max(proportions))

        X['class_balance_min'] = numeric_min
        X['class_balance_max'] = numeric_max

        X = X.drop(columns=[self.column])
        return X


def main(full_train_mode=False):
    df = pd.read_csv(META_DATA_PATH)

    # Before fitting the pipeline, fix regression rows:
    # For regression tasks, we know num_classes, class_balance, imbalance_ratio might be empty.
    # Set them to defaults:
    regression_mask = (df['task_type'] == 'Regression')

    # If class_balance is empty string or NaN, set to "{}"
    df.loc[regression_mask, 'class_balance'] = "{}"
    # If num_classes, imbalance_ratio are empty or NaN, set them to numeric defaults
    # Convert columns to numeric if needed:
    numeric_cols_to_check = ['num_classes', 'imbalance_ratio']
    for col in numeric_cols_to_check:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # For regression tasks, set num_classes=1, imbalance_ratio=1.0 if NaN
    df.loc[regression_mask, 'num_classes'] = df.loc[regression_mask, 'num_classes'].fillna(1.0)
    df.loc[regression_mask, 'imbalance_ratio'] = df.loc[regression_mask, 'imbalance_ratio'].fillna(1.0)

    # For any other columns that could be empty and numeric, ensure they are numeric and fill NaN with a default
    # Just in case:
    # Identify numeric columns:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Fill any remaining NaNs in numeric columns with a safe default (e.g., 1.0)
    # If you really don't want to impute arbitrarily, you must ensure no column is NaN at this point.
    # For now, let's do a final fill:
    df[numeric_cols] = df[numeric_cols].fillna(1.0)

    y = df['best_coreset_method']
    X = df.drop('best_coreset_method', axis=1)

    le = LabelEncoder()
    y = le.fit_transform(y)

    categorical_features = ['dataset_name', 'task_type', 'feature_type']
    numeric_features = [c for c in X.columns if c not in categorical_features and c not in ['best_coreset_method', 'class_balance']]

    pipeline = Pipeline(steps=[
        ('class_balance_transformer', ClassBalanceTransformer(column='class_balance')),
        ('task_type_weighter', TaskTypeWeighter(task_type_col='task_type')),
        ('column_transformer', ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
                ('num', StandardScaler(),
                 numeric_features + ['class_balance_min', 'class_balance_max', 'task_type_indicator'])
            ]
        )),
        # No imputer here since we handled NaNs preemptively in the dataframe
        ('classifier', RandomForestClassifier(random_state=42, n_estimators=21, min_samples_split=4, class_weight='balanced'))
    ])

    if not full_train_mode:
        param_grid = {
            'classifier__n_estimators': [18, 19, 20, 21, 22],
            'classifier__max_depth': [None, 5, 10, 15],
            'classifier__min_samples_split': [2, 4, 6]
        }
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(pipeline, param_grid, cv=kf, scoring='accuracy')
        grid_search.fit(X, y)
        print("Best parameters:", grid_search.best_params_)
        print("Best cross-validation accuracy:", grid_search.best_score_)
    else:
        pipeline.fit(X, y)
        user_data = pd.read_csv(USER_DATASET_PATH)
        single_instance_features = extract_dataset_features(
            user_data,
            target_column=USER_DATASET_TARGET_COLUMN,
            dataset_name="user_dataset",
            task_type=USER_TASK_TYPE
        )
        single_instance_df = pd.DataFrame([single_instance_features])
        for col in X.columns:
            if col not in single_instance_df.columns:
                single_instance_df[col] = 0
        if 'class_balance' not in single_instance_df.columns:
            single_instance_df['class_balance'] = "{}"
        single_instance_df = single_instance_df[X.columns]
        prediction = pipeline.predict(single_instance_df)
        predicted_class = le.inverse_transform(prediction)[0]
        print("Predicted best coreset method for the user-defined dataset:", predicted_class)

if __name__ == "__main__":
    # Set full_train_mode to True or False as desired
    # If False: Runs GridSearchCV for hyperparameter tuning
    # If True: Train on full dataset and then test on a single user-defined instance
    main(full_train_mode=False)
    # main(full_train_mode=True)