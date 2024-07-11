from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf

# Load the saved model
model_path = 'model/lead.keras'  # Replace with your actual model path
model = load_model(model_path)

X_train = np.load('./lead_bin/X.npy')
B_train = np.load('./lead_bin/B.npy')
y_train = np.load('./lead_bin/y.npy')

print(X_train.shape)
print(B_train.shape)
print(y_train.shape)

# Take the first 8 elements from each array
X_train_first_8 = X_train[:1000]
B_train_first_8 = B_train[:1000]

# Concatenate along the second axis (columns)
combined = np.concatenate((X_train_first_8, B_train_first_8), axis=1)
print(combined)
print(combined.shape)  # Output shape should be (8, 57)
# Make predictions
predictions = model.predict([X_train_first_8, B_train_first_8])

y_test = y_train[:1000]

# Print the expected output
#print("Expected:\n", y_test)

# Convert predictions and expected output to class indices
predicted_classes = np.argmax(predictions, axis=1)
expected_classes = np.argmax(y_test, axis=1)

# Print the predicted classes and expected classes
print("Predicted classes:\n", predicted_classes)
print("Expected classes:\n", expected_classes)

# Optionally, calculate and print the accuracy
accuracy = np.mean(predicted_classes == expected_classes)
print("Accuracy:", accuracy)