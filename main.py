import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

# Load the NSL-KDD dataset
train_data = pd.read_csv('/Users/farazsaeed/Downloads/archive-2/KDDTrain+_20Percent.txt', header=None)
test_data = pd.read_csv('/Users/farazsaeed/Downloads/archive-2/KDDTrain+_20Percent.txt', header=None)

# Check the number of columns
print(f'Train data columns: {train_data.shape[1]}')  # This should show 43 instead of 42
print(f'Test data columns: {test_data.shape[1]}')

# If there is an extra column (e.g., 43 columns instead of 42), drop the last column
if train_data.shape[1] == 43:
    train_data = train_data.drop(train_data.columns[-1], axis=1)
if test_data.shape[1] == 43:
    test_data = test_data.drop(test_data.columns[-1], axis=1)

# NSL-KDD dataset columns (42 features)
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
]

# Assign column names to data
train_data.columns = columns
test_data.columns = columns

# Combine training and test data labels to ensure the LabelEncoder knows all possible labels
combined_labels = pd.concat([train_data['label'], test_data['label']], axis=0)

# Encode categorical variables: 'protocol_type', 'service', 'flag'
categorical_cols = ['protocol_type', 'service', 'flag']
label_encoder = LabelEncoder()

# Apply the same encoder to both train and test
for col in categorical_cols:
    train_data[col] = label_encoder.fit_transform(train_data[col])
    test_data[col] = label_encoder.transform(test_data[col])

# Create a label encoder for the 'label' column using the combined labels (to handle unseen labels)
label_encoder = LabelEncoder()
label_encoder.fit(combined_labels)

# Transform the 'label' column for both train and test sets
train_data['label'] = label_encoder.transform(train_data['label'])
test_data['label'] = label_encoder.transform(test_data['label'])

# For binary classification, map any non-'normal' labels to 1 (attack), and 'normal' to 0
train_data['label'] = train_data['label'].apply(lambda x: 1 if label_encoder.inverse_transform([x])[0] != 'normal' else 0)
test_data['label'] = test_data['label'].apply(lambda x: 1 if label_encoder.inverse_transform([x])[0] != 'normal' else 0)

# Separate features and labels
X_train = train_data.drop('label', axis=1).values
y_train = train_data['label'].values

X_test = test_data.drop('label', axis=1).values
y_test = test_data['label'].values

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape input data for CNN (CNN requires 3D input: [samples, timesteps, features])
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report

# Build the CNN model
model = Sequential()

# First convolutional layer
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))

# Second convolutional layer
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

# Flatten the output for the fully connected layers
model.add(Flatten())

# Fully connected layer
model.add(Dense(128, activation='relu'))

# Dropout for regularization
model.add(Dropout(0.5))

# Output layer (binary classification)
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc}')

# Predict on the test set
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Generate classification report
print(classification_report(y_test, y_pred))
# Save the trained model to an HDF5 file
model.save('cnn_nids_model.h5')
print('Model saved as cnn_nids_model.h5')
import numpy as np
from tensorflow.keras.models import load_model

from CNN2 import X_train, scaler, X_test

# Load the saved model
model = load_model('cnn_nids_model.h5')
print('Model loaded for real-time detection')

# Simulate real-time detection (Here, you could integrate with live network data capture libraries like pyshark or scapy)
def real_time_detection(new_data):
    """
    This function simulates real-time detection using the pre-trained CNN model.
    'new_data' should be a 1D array (a single sample) representing the features of the network traffic.
    """
    # Ensure new_data has the same number of features as the training data
    assert len(new_data) == X_train.shape[1], "New data must have the same number of features as the training data"

    # Reshape and scale the new data (following the same preprocessing as the training data)
    new_data = np.array(new_data).reshape(1, -1)  # Reshape for a single sample
    new_data = scaler.transform(new_data)  # Apply the same scaling as the training data
    new_data = new_data.reshape(1, X_train.shape[1], 1)  # Reshape for CNN input (3D)

    # Predict using the loaded model
    prediction = model.predict(new_data)
    result = "Attack" if prediction > 0.5 else "Normal"
    print(f"Real-time prediction: {result}")

    return result

# Simulate new data for real-time detection (Replace this with live traffic in a real scenario)
sample_new_data = X_test[0].flatten()  # Taking the first test sample as an example for new data
real_time_detection(sample_new_data)
