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
USER_DATASET_PATH = 'MetaModel_Test_Sets/Customertravel.csv'
USER_TASK_TYPE = 'Binary Classification'
USER_DATASET_TARGET_COLUMN = 'Target'

# 2 -grad
#USER_DATASET_PATH = 'MetaModel_Test_Sets/social_ads.csv'
#USER_TASK_TYPE = 'Binary Classification'
#USER_DATASET_TARGET_COLUMN = 'Purchased'

# 3 -res
#USER_DATASET_PATH = 'MetaModel_Test_Sets/plant_growth_data.csv'
#USER_TASK_TYPE = 'Binary Classification'
#USER_DATASET_TARGET_COLUMN = 'Growth_Milestone'

# 4 -res
#USER_DATASET_PATH = 'MetaModel_Test_Sets/diabetes_dataset.csv'
#USER_TASK_TYPE = 'Binary Classification'
#USER_DATASET_TARGET_COLUMN = 'Outcome'

# 5 -grad
#USER_DATASET_PATH = 'MetaModel_Test_Sets/customer_purchase_data.csv'
#USER_TASK_TYPE = 'Binary Classification'
#USER_DATASET_TARGET_COLUMN = 'PurchaseStatus'

# 6 -res
#USER_DATASET_PATH = 'MetaModel_Test_Sets/happydata.csv'
#USER_TASK_TYPE = 'Binary Classification'
#USER_DATASET_TARGET_COLUMN = 'happy'

# 7 -strat
#USER_DATASET_PATH = 'MetaModel_Test_Sets/riceClassification.csv'
#USER_TASK_TYPE = 'Binary Classification'
#USER_DATASET_TARGET_COLUMN = 'Class'


# Path to the meta-data CSV file
META_DATA_PATH = 'meta_dataset.csv'


def extract_dataset_features(data, target_column, dataset_name, task_type):
    features = {}

    DATASET_NAME = dataset_name
    TASK_TYPE = task_type

    # Number of Instances and Features
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

    # Feature Type Indicator
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

    # Dimensionality
    features['dimensionality'] = features['num_features'] / features['num_instances'] if features[
                                                                                             'num_instances'] > 0 else 0

    # Correlation (for numerical only)
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

    # Statistical Properties
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

    # Outliers
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

    # Data Sparsity
    total_elements = data.shape[0] * data.shape[1]
    zero_elements = (data == 0).sum().sum()
    features['data_sparsity'] = zero_elements / total_elements if total_elements > 0 else 0.0

    return features


#########################################
# Custom Transformer to add task_type weighting
#########################################
class TaskTypeWeighter(BaseEstimator, TransformerMixin):
    def __init__(self, task_type_col='task_type'):
        self.task_type_col = task_type_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['task_type_indicator'] = X[self.task_type_col].apply(lambda x: 10 if x == 'Regression' else 1)
        return X


#########################################
# ClassBalanceTransformer to convert class_balance dict-string to numeric features
#########################################
class ClassBalanceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column='class_balance'):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        numeric_min = []
        numeric_max = []
        for val in X[self.column]:
            # Check the type of val
            if isinstance(val, dict):
                dict_val = val
            elif isinstance(val, str):
                val = val.strip()
                if val == "{}" or val == "":
                    dict_val = {}
                else:
                    dict_val = eval(val)  # Use with caution, but we trust the data format
            else:
                dict_val = {}

            if len(dict_val) == 0:
                # No classes: assume single class
                numeric_min.append(1.0)
                numeric_max.append(1.0)
            else:
                proportions = dict_val.values()
                numeric_min.append(min(proportions))
                numeric_max.append(max(proportions))

        X['class_balance_min'] = numeric_min
        X['class_balance_max'] = numeric_max
        print(numeric_min)
        print(numeric_max)


        # Drop original class_balance column
        X = X.drop(columns=[self.column])
        return X


#########################################
# Main Code
#########################################
def main(full_train_mode=False):
    # Load the metadata dataset (50 instances)
    df = pd.read_csv(META_DATA_PATH)

    # Separate target
    y = df['best_coreset_method']
    X = df.drop('best_coreset_method', axis=1)

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(y)

    categorical_features = ['dataset_name', 'task_type', 'feature_type']

    # We'll specify numeric features after transformations. We'll assume everything not categorical
    # and not 'best_coreset_method' or 'class_balance' is numeric.
    # After transformations, we have class_balance_min, class_balance_max, task_type_indicator as numeric too.
    numeric_features = [c for c in X.columns if
                        c not in categorical_features and c not in ['best_coreset_method', 'class_balance']]

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
        ('classifier', RandomForestClassifier(
            random_state=42,
            n_estimators=55,
            min_samples_split=8
        ))
    ])

    if not full_train_mode:
        # Hyperparameter tuning with GridSearchCV
        param_grid = {
            'classifier__n_estimators': [54, 55, 56],
            'classifier__max_depth': [None],
            'classifier__min_samples_split': [6, 8, 10],
        }

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(pipeline, param_grid, cv=kf, scoring='accuracy')
        grid_search.fit(X, y)

        print("Best parameters:", grid_search.best_params_)
        print("Best cross-validation accuracy:", grid_search.best_score_)
    else:
        # Train on full dataset
        pipeline.fit(X, y)

        # Load user-defined dataset
        user_data = pd.read_csv(USER_DATASET_PATH)

        # Extract features from user dataset
        single_instance_features = extract_dataset_features(
            user_data,
            target_column=USER_DATASET_TARGET_COLUMN,
            dataset_name="user_dataset",
            task_type=USER_TASK_TYPE
        )
        single_instance_df = pd.DataFrame([single_instance_features])  # single row DataFrame

        # Ensure all columns used in training are present in single_instance_df
        for col in X.columns:
            if col not in single_instance_df.columns:
                single_instance_df[col] = 0

        if 'class_balance' not in single_instance_df.columns:
            single_instance_df['class_balance'] = "{}"

        # Reorder columns to match training data columns
        single_instance_df = single_instance_df[X.columns]

        # Predict on single instance
        prediction = pipeline.predict(single_instance_df)
        predicted_class = le.inverse_transform(prediction)[0]
        print("Predicted best coreset method for the user-defined dataset:", predicted_class)


if __name__ == "__main__":
    # Set full_train_mode to True or False as desired
    # If False: Runs GridSearchCV for hyperparameter tuning
    # If True: Train on full dataset and then test on a single user-defined instance
    main(full_train_mode=True)
    # main(full_train_mode=True)