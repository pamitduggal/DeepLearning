"""
Exercise 3: With more control using keras
Objective: Getting started with Keras and build a simple neural network.
1. Strat from (or redo) categorized data from the IRIS data set.
2. Using Keras, create a perceptron built as follows:
a. One dense layer of 16 neurons connected to a 4-neurons input layer. The
activation function of this layer must be a sigmoid
b. A output layer of 3 neurons with a softmax activation function.
c. Used a categorical crossentropy (log loss) objective function, the adam
optimizer and precompute the accuracy metric.
3. Train the network for 100 epochs
4. Compare the precision of the classification obtained with the logistic regression of
exercise 1.

"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

# 1. Load and prepare the Iris dataset
print("\n1. Loading and preparing Iris dataset...")
iris = load_iris()
X = iris.data
y = iris.target

# Split into train/test (75/25)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Transform labels to categorical (one-hot encoding)
y_train_cat = to_categorical(y_train, 3)
y_test_cat = to_categorical(y_test, 3)

print(f"Training data shape: {X_train.shape}")
print(f"Training labels shape: {y_train_cat.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Test labels shape: {y_test_cat.shape}")

# Also train a logistic regression for comparison
print("\nTraining logistic regression for comparison...")
log_clf = LogisticRegression(max_iter=200)
log_clf.fit(X_train, y_train)
log_test_score = log_clf.score(X_test, y_test)

# 2. Build the neural network architecture
print("\n2. Building Keras neural network...")
print("   Architecture:")
print("   - Input layer: 4 features (sepal length, sepal width, petal length, petal width)")
print("   - Hidden layer: 16 neurons with sigmoid activation")
print("   - Output layer: 3 neurons with softmax activation")

model = Sequential([
    # Input layer: 4 features
    # Hidden layer: 16 neurons with sigmoid activation
    Dense(16, activation='sigmoid', input_shape=(4,), name='hidden_layer'),

    # Output layer: 3 neurons (one for each class) with softmax activation
    # Softmax ensures the outputs sum to 1, so they can be interpreted as probabilities
    Dense(3, activation='softmax', name='output_layer')
])
print("\n3. Compiling the model...")
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # log loss
    metrics=['accuracy']
)

# Display the model architecture
print("\nModel Architecture:")
model.summary()

# 3. Train the network for 100 epochs
print("\n4. Training the model for 100 epochs...")
history = model.fit(
    X_train, y_train_cat,
    epochs=100,
    batch_size=32,  # 32 samples at a time
    validation_data=(X_test, y_test_cat),
    verbose=1
)

# Evaluate the model on test data
print("\n5. Evaluating the model...")
test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
train_loss, train_accuracy = model.evaluate(X_train, y_train_cat, verbose=0)

print(f"\nKeras Model Results:")
print(f"  Training accuracy: {train_accuracy:.4f}")
print(f"  Test accuracy: {test_accuracy:.4f}")

# 4. Compare with logistic regression from Exercise 2
print("\n6. Comparison with Logistic Regression:")
print("="*60)
print(f"Logistic Regression test accuracy: {log_test_score:.4f}")
print(f"Keras Neural Network test accuracy: {test_accuracy:.4f}")
print(f"Difference: {test_accuracy - log_test_score:.4f}")

"""
My Observation: The Keras model should achieve similar or slightly better accuracy
Both models work well on Iris because it's a relatively simple dataset
The neural network has more parameters but may not always outperform
the simpler logistic regression on this particular problem

Logistic Regression test accuracy: 0.9474
Keras Neural Network test accuracy: 0.8158
Difference: -0.1316
"""

