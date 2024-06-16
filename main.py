import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load the dataset
file_path = 'path_to_your_file.csv'  # Update this with your file path
triage_data = pd.read_csv(file_path)

# Impute missing numerical values with the mean
numerical_features = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'acuity']
imputer = SimpleImputer(strategy='mean')
triage_data[numerical_features] = imputer.fit_transform(triage_data[numerical_features])

# Impute missing categorical values with the most frequent value
categorical_features = ['pain']
imputer = SimpleImputer(strategy='most_frequent')
triage_data[categorical_features] = imputer.fit_transform(triage_data[categorical_features])

# Encode categorical variables
label_encoder = LabelEncoder()
triage_data['pain'] = label_encoder.fit_transform(triage_data['pain'])
triage_data['chiefcomplaint'] = label_encoder.fit_transform(triage_data['chiefcomplaint'])

# Convert 'acuity' to categorical
triage_data['acuity'] = triage_data['acuity'].astype(int)

# Prepare features and target variable
X = triage_data.drop(columns=['subject_id', 'stay_id', 'acuity'])
y = triage_data['acuity']

# Encode target variable
label_encoder_acuity = LabelEncoder()
y_encoded = label_encoder_acuity.fit_transform(y)

# Increase data by repeating samples to balance classes
class_counts = y.value_counts()
max_count = class_counts.max()

balanced_X = pd.DataFrame()
balanced_y = pd.Series(dtype=int)

for class_value in class_counts.index:
    class_X = X[y == class_value]
    class_y = pd.Series([class_value] * len(class_X), dtype=int)  # Use the actual class value for the labels

    repeated_class_X = class_X.sample(max_count, replace=True, random_state=42)
    repeated_class_y = pd.Series([class_value] * max_count, dtype=int)

    balanced_X = pd.concat([balanced_X, repeated_class_X])
    balanced_y = pd.concat([balanced_y, repeated_class_y])

# Split the balanced data into training and testing sets
X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced = train_test_split(balanced_X, balanced_y,
                                                                                        test_size=0.2, random_state=42)

# Train a Random Forest Classifier on the balanced data
rf_model_balanced = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
rf_model_balanced.fit(X_train_balanced, y_train_balanced)

# Predict on the test set
y_pred_balanced = rf_model_balanced.predict(X_test_balanced)

# Evaluate the balanced model
accuracy_balanced = accuracy_score(y_test_balanced, y_pred_balanced)
classification_report_balanced = classification_report(y_test_balanced, y_pred_balanced)
confusion_matrix_balanced = confusion_matrix(y_test_balanced, y_pred_balanced)

print(f'Accuracy: {accuracy_balanced}')
print('Classification Report:')
print(classification_report_balanced)
print('Confusion Matrix:')
print(confusion_matrix_balanced)

# Save the model and label encoder
joblib.dump(rf_model_balanced, 'triage_model.pkl')
joblib.dump(label_encoder_acuity, 'label_encoder_acuity.pkl')
