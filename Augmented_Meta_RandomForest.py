from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# This whole script was written by Ahmed.
#
# This was the best performing model through the
# use of Random Forest and data augmentation.
# Ahmed also included tree plotting, a useful feature
# to understand the decision-making of the model.

# Load dataset
file_path = '/Users/ahmed/Desktop/Machine Learning/final_augmented.csv'
df = pd.read_csv(file_path)

# Drop columns with all missing values
df = df.dropna(axis=1, how='all')

# Drop rows where the target 'best_coreset_method' is missing
df = df.dropna(subset=['best_coreset_method'])

# Fill missing numerical values with the median
numerical_cols = df.select_dtypes(include=[np.number]).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

# Fill missing categorical values with the mode
categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

# Assign unique group IDs (every two rows are grouped together)
df['group_id'] = np.arange(len(df)) // 2

# Select features and target
X = df.select_dtypes(include=[np.number]).drop(columns=['group_id'])
y = df['best_coreset_method']
group_ids = df['group_id']

# Encode the target variable
y_encoded = pd.factorize(y)[0]


# Function to split data while keeping group integrity
def group_split(X, y, group_ids, test_size):
    # Combine features, target, and group IDs
    combined = pd.DataFrame(X)
    combined['target'] = y
    combined['group_id'] = group_ids
    # Split groups into train and test
    unique_groups = combined['group_id'].unique()
    train_groups, test_groups = train_test_split(unique_groups, test_size=test_size, random_state=22)
    train_data = combined[combined['group_id'].isin(train_groups)]
    test_data = combined[combined['group_id'].isin(test_groups)]
    # Split back into features and target
    X_train = train_data.drop(columns=['target', 'group_id'])
    y_train = train_data['target']
    X_test = test_data.drop(columns=['target', 'group_id'])
    y_test = test_data['target']
    return X_train, X_test, y_train, y_test


# Function to train and evaluate the Random Forest model
def evaluate_model(test_size):
    X_train, X_test, y_train, y_test = group_split(X, y_encoded, group_ids, test_size=test_size)
    model = RandomForestClassifier(n_estimators=73, random_state=22, max_depth=50)  # Limit depth for simplicity
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, model


# Evaluate the model for 10%, 20%, and 30% test sizes
accuracy_10, _ = evaluate_model(test_size=0.1)
accuracy_20, model_20 = evaluate_model(test_size=0.2)
accuracy_30, _ = evaluate_model(test_size=0.3)

# Print accuracies
print(f'Accuracy with 10% testing data: {accuracy_10:.2f}')
print(f'Accuracy with 20% testing data: {accuracy_20:.2f}')
print(f'Accuracy with 30% testing data: {accuracy_30:.2f}')


# Function to plot a customized decision tree
def plot_custom_tree(model, feature_names, class_names):
    plt.figure(figsize=(15, 8))
    tree = model.estimators_[0]  # Select one tree (4th tree) in the forest
    plot_tree(tree, feature_names=feature_names, class_names=class_names, filled=True, rounded=True, fontsize=8)

    # Modify the text of each node to remove 'samples', 'gini', and 'value' lines
    for item in plt.gca().texts:
        text = item.get_text()
        lines = text.split('\n')

        # Remove 'gini' and 'value' lines
        lines = [line for line in lines if 'gini' not in line and 'value' not in line]

        # Format 'samples' line to include class name
        for i, line in enumerate(lines):
            if 'samples' in line:
                sample_count = line.split('=')[1].strip()
                # Find the class line and format it
                for j, class_line in enumerate(lines):
                    if 'class' in class_line:
                        class_name = class_line.split('=')[1].strip()
                        lines[j] = f'Class: {class_name} @ {sample_count} samples'
                        lines.pop(i)  # Remove the original 'samples' line
                        break

        # Update the node text
        item.set_text('\n'.join(lines))

    plt.show()


# Plot the customized tree for the 20% test size
plot_custom_tree(model_20, X.columns, np.unique(y).astype(str))