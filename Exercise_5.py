"""
Exercise 5: Deeper multi-layer perceptron
Objective: Create a multi-layer perceptron on a more complex image classification task.
1. Load, display and format the Cifar10 dataset
2. Build the mono-layer perceptron from the previous exercise. Train, evaluate and
display classification errors. Try to modify gently the network (no new layer but
larger hidden layer, different activation functions, etc.) and evaluate these
different networks. What do you think of the prediction you obtain?
3. Build the following multilayer perceptron:
a. First hidden layer of 512 neurons densely connected to the input layer, relu
activation function, dropout of 0.2 and batch normalization.
b. Second hidden layer of 512 neurons densely with relu activation function,
batch normalization and dropout of 0.2.
4. Run 50 epochs of training and evaluate the obtained network. What do you think
about the prediction score?
5. Save your model on your hard drive and build the accuracy and loss for each
training epoch.
6. Build the architecture! Comment on advantages of your choice and the
disadvantages.

"""

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint

import numpy as np
import matplotlib.pyplot as plt

# 1. Load, display and format the CIFAR-10 dataset
print("\n1. Loading CIFAR-10 dataset...")
(X_train_cifar, y_train_cifar), (X_test_cifar, y_test_cifar) = cifar10.load_data()

# CIFAR-10 has 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"Training data shape: {X_train_cifar.shape}")
print(f"Training labels shape: {y_train_cifar.shape}")
print(f"Test data shape: {X_test_cifar.shape}")
print(f"Test labels shape: {y_test_cifar.shape}")
print(f"Image dimensions: {X_train_cifar.shape[1]}x{X_train_cifar.shape[2]} pixels")
print(f"Number of color channels: {X_train_cifar.shape[3]} (RGB)")
print(f"Number of classes: {len(class_names)}")

# Display some sample images
print("\nDisplaying sample images from CIFAR-10...")
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i in range(10):
    row, col = i // 5, i % 5
    axes[row, col].imshow(X_train_cifar[i])
    axes[row, col].set_title(f'{class_names[y_train_cifar[i][0]]}')
    axes[row, col].axis('off')
plt.tight_layout()
plt.savefig('Ex5_CIFAR10_samples.png', dpi=150, bbox_inches='tight')
plt.show()

# Preprocess the data
print("\n2. Preprocessing CIFAR-10 data...")
# Flatten images: 32x32x3 = 3072 features
print("   Flattening images (32x32x3 -> 3072)...")
X_train_cifar_flat = X_train_cifar.reshape(X_train_cifar.shape[0], 32 * 32 * 3)
X_test_cifar_flat = X_test_cifar.reshape(X_test_cifar.shape[0], 32 * 32 * 3)

# Normalize pixel values to 0-1 range
print("   Normalizing pixel values (0-255 -> 0-1)...")
X_train_cifar_flat = X_train_cifar_flat.astype('float32') / 255.0
X_test_cifar_flat = X_test_cifar_flat.astype('float32') / 255.0

# Convert labels to categorical
print("   Converting labels to categorical")
y_train_cifar_cat = to_categorical(y_train_cifar, 10)
y_test_cifar_cat = to_categorical(y_test_cifar, 10)

print(f"\nPreprocessed data shapes:")
print(f"  Flattened training data: {X_train_cifar_flat.shape}")
print(f"  Categorical training labels: {y_train_cifar_cat.shape}")

# 2. Build the mono-layer perceptron (Model 1)
print("\n3. Building single-layer perceptron (baseline)")

model_cifar_simple = Sequential([
    Dense(16, activation='sigmoid', input_shape=(3072,), name='hidden_layer'),
    Dense(10, activation='softmax', name='output_layer')
])

model_cifar_simple.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Training simple model (16 neurons)...")
history_simple = model_cifar_simple.fit(
    X_train_cifar_flat, y_train_cifar_cat,
    epochs=20,
    batch_size=128,
    validation_data=(X_test_cifar_flat, y_test_cifar_cat),
    verbose=1
)

simple_train_acc = model_cifar_simple.evaluate(X_train_cifar_flat, y_train_cifar_cat, verbose=0)[1]
simple_test_acc = model_cifar_simple.evaluate(X_test_cifar_flat, y_test_cifar_cat, verbose=0)[1]

print(f"\nSimple Model Results:")
print(f"  Training accuracy: {simple_train_acc:.4f}")
print(f"  Test accuracy: {simple_test_acc:.4f}")

# Try a larger hidden layer
print("\nTrying larger hidden layer (128 neurons)...")
model_cifar_larger = Sequential([
    Dense(128, activation='relu', input_shape=(3072,), name='hidden_layer'),
    Dense(10, activation='softmax', name='output_layer')
])

model_cifar_larger.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Training larger model (128 neurons)...")
history_larger = model_cifar_larger.fit(
    X_train_cifar_flat, y_train_cifar_cat,
    epochs=20,
    batch_size=128,
    validation_data=(X_test_cifar_flat, y_test_cifar_cat),
    verbose=1
)

larger_train_acc = model_cifar_larger.evaluate(X_train_cifar_flat, y_train_cifar_cat, verbose=0)[1]
larger_test_acc = model_cifar_larger.evaluate(X_test_cifar_flat, y_test_cifar_cat, verbose=0)[1]

print(f"\nLarger Model Results (128 neurons):")
print(f"  Training accuracy: {larger_train_acc:.4f}")
print(f"  Test accuracy: {larger_test_acc:.4f}")

"""
 My Observation: The simple model has low accuracy (maybe 20-30%)
 CIFAR-10 is much harder than MNIST because:
 - Images are colored (3 channels vs 1) and Objects can appear in different positions, scales, orientations

 A larger hidden layer helps, but we need a deeper network for better results
"""

"""
 3. Build the following multilayer perceptron:
 a. First hidden layer of 512 neurons densely connected to the input layer,
    relu activation function, dropout of 0.2 and batch normalization
 b. Second hidden layer of 512 neurons densely with relu activation function,
      batch normalization and dropout of 0.2
"""
print("\n4. Building deeper multilayer perceptron...")
print("   Architecture:")
print("   - Input layer: 3072 features (flattened 32x32x3 image)")
print("   - Hidden layer 1: 512 neurons, ReLU, BatchNorm, Dropout(0.2)")
print("   - Hidden layer 2: 512 neurons, ReLU, BatchNorm, Dropout(0.2)")
print("   - Output layer: 10 neurons with softmax")

model_cifar_deep = Sequential([
    # First hidden layer: 512 neurons
    Dense(512, activation='relu', input_shape=(3072,), name='hidden_layer_1'),
    BatchNormalization(),  # Normalize activations to help training
    Dropout(0.2),  # Randomly set 20% of neurons to 0 during training (prevents overfitting)

    # Second hidden layer: 512 neurons
    Dense(512, activation='relu', name='hidden_layer_2'),
    BatchNormalization(),  # Normalize activations
    Dropout(0.2),  # Dropout again

    # Output layer: 10 classes
    Dense(10, activation='softmax', name='output_layer')
])

model_cifar_deep.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nDeep Model Architecture:")
model_cifar_deep.summary()

# 4. Run 50 epochs of training
print("\n5. Training deep model for 50 epochs...")


# Create a callback to save the best model during training
checkpoint = ModelCheckpoint(
    'best_cifar10_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

history_deep = model_cifar_deep.fit(
    X_train_cifar_flat, y_train_cifar_cat,
    epochs=50,
    batch_size=128,
    validation_data=(X_test_cifar_flat, y_test_cifar_cat),
    callbacks=[checkpoint],
    verbose=1
)

# Evaluate the deep network
deep_train_acc = model_cifar_deep.evaluate(X_train_cifar_flat, y_train_cifar_cat, verbose=0)[1]
deep_test_acc = model_cifar_deep.evaluate(X_test_cifar_flat, y_test_cifar_cat, verbose=0)[1]

print(f"\nDeep Model Results (after 50 epochs):")
print(f"  Training accuracy: {deep_train_acc:.4f}")
print(f"  Test accuracy: {deep_test_acc:.4f}")

"""
 My Observation: The deep model should achieve much better accuracy (maybe 40-50%)
 However, CIFAR-10 is still challenging for MLPs because:
 - MLPs don't capture spatial relationships well (they treat pixels independently)
 - Convolutional Neural Networks (CNNs) would be much better for images
 - But this shows how architecture depth and regularization help
"""
# 5. Save the model and build accuracy/loss plots
print("\n6. Saving model and plotting training history...")

# Save the final model
model_cifar_deep.save('cifar10_mlp_model.h5')
print("Model saved as 'cifar10_mlp_model.h5'")

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot accuracy
axes[0].plot(history_deep.history['accuracy'], label='Training Accuracy')
axes[0].plot(history_deep.history['val_accuracy'], label='Validation Accuracy')
axes[0].set_title('Model Accuracy Over Epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

# Plot loss
axes[1].plot(history_deep.history['loss'], label='Training Loss')
axes[1].plot(history_deep.history['val_loss'], label='Validation Loss')
axes[1].set_title('Model Loss Over Epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('Ex5_CIFAR10_training_history.png', dpi=150, bbox_inches='tight')
plt.show()

# 6. Build the architecture! Comment on advantages and disadvantages

"""
Architecture Analysis:

ADVANTAGES of this architecture:

1. Deep layers (512 neurons each) provide high capacity to learn complex patterns
2. ReLU activation helps with gradient flow and training speed
3. Batch Normalization stabilizes training and allows higher learning rates
4. Dropout (0.2) prevents overfitting by randomly disabling neurons during training
5. Dense layers can capture global patterns in the data

DISADVANTAGES of this architecture:

1. MLPs don't preserve spatial structure - they flatten images, losing 2D relationships
2. Very large number of parameters (512*3072 + 512*512 + 512*10 = ~1.8M parameters)
3. Computationally expensive - requires significant memory and processing power
4. For images, Convolutional Neural Networks (CNNs) would be much more efficient
5. May still struggle with CIFAR-10 because images have complex spatial patternS
6. Risk of overfitting despite dropout if not enough data or too many epochs

BETTER ALTERNATIVE:

For image classification, Convolutional Neural Networks (CNNs) are preferred
because they:
- Preserve spatial relationships through convolution operations
Share parameters across spatial locations (more efficient)
 translation-invariant (can recognize objects in different positions)
 Typically achieve 70-90% accuracy on CIFAR-10 vs 40-50% for MLPs

"""
