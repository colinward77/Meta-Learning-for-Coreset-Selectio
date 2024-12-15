import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# Load the dataset (update the path if necessary)
file_path = "/Users/ahmed/Desktop/Meta-Learning-for-Coreset-Selectio/augmented_meta_dataset.csv"
df = pd.read_csv(file_path)

# Prepare features and labels
X = df.drop(columns=['best_coreset_method'])
y = df['best_coreset_method']

# Impute any missing values with the mean for numeric columns
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Initialize a dictionary to store accuracies
rf_accuracies = {}

# Define different test sizes
test_sizes = [0.1, 0.3, 0.4]

# Loop through each test size and evaluate Random Forest
for test_size in test_sizes:
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=test_size, random_state=42)
    
    # Train the Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, rf.predict(X_test))
    rf_accuracies[test_size] = accuracy

# Print the results
for test_size, acc in rf_accuracies.items():
    print(f"Test Size: {test_size*100:.0f}%, Accuracy: {acc:.2f}")
