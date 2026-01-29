"""
Exercise 4 : Adapt a network to a given problem
Objective: Build a multilayer perceptron by changing its architecture to be adapted to a
given problem
1. Load the MNist dataset and visualize the different loaded vectors. Display the first
images of the dataset. What is this dataset about?
2. Modify the data to:
a. Linearize the data in two dimensions.
b. Normalize the data
c. Transform the class vector to be used by a neural network.
3. Train the architecture of the previous exercise to this new problem. Make sure to
adapt the input and output layer sizes.
4. Train the network of few epochs with a batch size of 128. Observe the network
score. What do you think about it?

"""

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np
import matplotlib.pyplot as plt


# 1. Load the MNIST dataset
# MNIST contains 70,000 images of handwritten digits (0-9)
print("\n1. Loading MNIST dataset...")
(X_train_mnist, y_train_mnist), (X_test_mnist, y_test_mnist) = mnist.load_data()

# Display information about dataset
print(f"Training data shape: {X_train_mnist.shape}")
print(f"Training labels shape: {y_train_mnist.shape}")
print(f"Test data shape: {X_test_mnist.shape}")
print(f"Test labels shape: {y_test_mnist.shape}")
print(f"Image dimensions: {X_train_mnist.shape[1]}x{X_train_mnist.shape[2]} pixels")
print(f"Number of classes: {len(np.unique(y_train_mnist))} (digits 0-9)")

# 2. Visualize the first few images
print("\n2. Displaying first 9 images from the dataset...")
fig, axes = plt.subplots(3, 3, figsize=(8, 8))
for i in range(9):
    row, col = i // 3, i % 3
    axes[row, col].imshow(X_train_mnist[i], cmap='gray')
    axes[row, col].set_title(f'Label: {y_train_mnist[i]}')
    axes[row, col].axis('off')
plt.tight_layout()
plt.savefig('Ex4_MNIST_samples.png', dpi=150, bbox_inches='tight')
plt.show()

""" My Observation: MNIST contains grayscale images of handwritten digits
 Each image shows a single digit (0-9) written by hand
 The images are already preprocessed and centered
"""

"""
 3. Modify the data:
 a. Linearize the data in two dimensions (flatten 28x28 to 784)
 b. Normalize the data (scale pixel values from 0-255 to 0-1)
 c. Transform the class vector to be used by a neural network

"""
print("\n3. Preprocessing the data...")

# a. Linearize: flatten 28x28 images to 784-dimensional vectors
print("   a. Flattening images (28x28 -> 784)...")
X_train_flat = X_train_mnist.reshape(X_train_mnist.shape[0], 28 * 28)
X_test_flat = X_test_mnist.reshape(X_test_mnist.shape[0], 28 * 28)

# b. Normalize: convert pixel values from 0-255 to 0-1 range
# This helps the neural network train faster and more stable
print("   b. Normalizing pixel values (0-255 -> 0-1)...")
X_train_flat = X_train_flat.astype('float32') / 255.0
X_test_flat = X_test_flat.astype('float32') / 255.0

# c. Transform labels to categorical (one-hot encoding)
# Instead of label 5, we get [0,0,0,0,0,1,0,0,0,0]
print("   c. Converting labels to categorical (one-hot encoding)...")
y_train_mnist_cat = to_categorical(y_train_mnist, 10)
y_test_mnist_cat = to_categorical(y_test_mnist, 10)

print(f"\nPreprocessed data shapes:")
print(f"  Flattened training data: {X_train_flat.shape}")
print(f"  Categorical training labels: {y_train_mnist_cat.shape}")
print(f"  Sample label (original): {y_train_mnist[0]}")
print(f"  Sample label (categorical): {y_train_mnist_cat[0]}")

# 4. Train the architecture from Exercise 3, but adapt it for MNIST
# Input layer: 784 neurons (instead of 4 for Iris)
# Output layer: 10 neurons (instead of 3 for Iris)

print("\n4. Building and training the neural network...")
print("   Architecture:")
print("   - Input layer: 784 features (flattened 28x28 image)")
print("   - Hidden layer: 16 neurons with sigmoid activation")
print("   - Output layer: 10 neurons (one for each digit 0-9) with softmax")

# Create a new model for MNIST
model_mnist = Sequential([
    # Hidden layer: 16 neurons with sigmoid activation
    # Input is now 784 features (flattened image)
    Dense(16, activation='sigmoid', input_shape=(784,), name='hidden_layer'),

    # Output layer: 10 neurons (one for each digit 0-9) with softmax
    Dense(10, activation='softmax', name='output_layer')
])

# Compile with same settings as Exercise 3
model_mnist.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Architecture:")
model_mnist.summary()

# 5. Train the network for a few epochs with batch size of 128
print("\n5. Training the network (batch size=128)...")
print("   (This may take a few minutes...)")
history_mnist = model_mnist.fit(
    X_train_flat, y_train_mnist_cat,
    epochs=10,  # start with 10 epochs to see initial results
    batch_size=128,  # process 128 samples at a time
    validation_data=(X_test_flat, y_test_mnist_cat),
    verbose=1
)

# 6. Evaluate the network and observe the score
print("\n6. Evaluating the network...")
train_loss_mnist, train_acc_mnist = model_mnist.evaluate(X_train_flat, y_train_mnist_cat, verbose=0)
test_loss_mnist, test_acc_mnist = model_mnist.evaluate(X_test_flat, y_test_mnist_cat, verbose=0)

print(f"\nMNIST Model Results (after 10 epochs):")
print(f"  Training accuracy: {train_acc_mnist:.4f}")
print(f"  Test accuracy: {test_acc_mnist:.4f}")

"""
 My Observation: The accuracy might be relatively low (maybe 70-85%)
 This is because MNIST is more complex than Iris, and we only have 16 hidden neurons
 The network needs more capacity (more neurons or layers) to learn the patterns
 Also, 10 epochs might not be enough - we may need more training time
"""
# Visualize some predictions
print("\n7. Visualizing some predictions...")
predictions = model_mnist.predict(X_test_flat[:9], verbose=0)
predicted_labels = np.argmax(predictions, axis=1)

fig, axes = plt.subplots(3, 3, figsize=(8, 8))
for i in range(9):
    row, col = i // 3, i % 3
    axes[row, col].imshow(X_test_mnist[i], cmap='gray')
    true_label = y_test_mnist[i]
    pred_label = predicted_labels[i]
    color = 'green' if true_label == pred_label else 'red'
    axes[row, col].set_title(f'True: {true_label}, Pred: {pred_label}', color=color)
    axes[row, col].axis('off')
plt.tight_layout()
plt.savefig('Ex4_MNIST_predictions.png', dpi=150, bbox_inches='tight')
plt.show()


