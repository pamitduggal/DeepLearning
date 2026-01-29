"""
Exercise 2: Simply with sci-kit learn
Objective: load, visualize a dataset. Train a logistic classifier and a multilayer perceptron
with python and scikit learn
1. In Python, load the iris dataset. Use a pairplot polt (seaborn library) to visualize
and analyze the data.
2. Separate the data into 4 matrices: x_train, x_test, y_train, y_test. Input data are on
the first 4 columns and the output data is on the last column. Train/test ratio must
be 75/25.
3. By using SKLearn, build a logistic classifier on the training data set. Evaluate the
trained model on the test dataset and display the prediction score. What do you
think about this score?
4. Transform the data to make them usable by a neural network: the output must be
a Nxn matrice (n being the number of classes of the problem and N the number of
observations) instead of a Nx1 vector. Each column must contain a 1 if the
observation is of this class and zero otherwise. Look at the to_categorical function!
5. Still using SKLearn, create a perceptron to classify the IRIS dataset. The network
must contain a hidden layer of 16 neurons. You can change the hyper-parameters
(solver, activation function, learning rate, etc.) of the network as you wish.
6. Train the network on the training dataset and display the score on both the
training and the test datasets.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.utils import to_categorical

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Loading the data Iris
print("\n1. Loading Iris dataset...")
iris = load_iris()
X = iris.data          # first 4 columns = inputs
y = iris.target        # last column = output (class)

# Create a dataframe for visualization
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(y, iris.target_names)

# Visualize the data with pairplot
print("Creating pairplot visualization...")
sns.pairplot(df, hue='species')
plt.savefig('Ex2_Iris_pairplot.png', dpi=150, bbox_inches='tight')
plt.show()

# My Observation: petal features separate the classes clearly and sepal features overlap more

# 2. Train/test split 75%/25%
print("\n2. Splitting data into train/test (75/25)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# 3. Logistic regression classifier
print("\n3. Training logistic regression classifier...")
log_clf = LogisticRegression(max_iter=200)
log_clf.fit(X_train, y_train)

log_train_score = log_clf.score(X_train, y_train)
log_test_score  = log_clf.score(X_test, y_test)

print("Logistic Regression Results:")
print(f"  Training accuracy: {log_train_score:.4f}")
print(f"  Test accuracy: {log_test_score:.4f}")

"""
My Observation: Iris is easy, so accuracies around 0.95–1.0 are normal
Logistic train accuracy: ~0.97
Logistic test accuracy: ~0.95
"""

# 4. Transforming Data for Neural Network
print("\n4. Transforming labels to categorical (one-hot encoding)...")
y_train_cat = to_categorical(y_train)   # shape (N_train, 3)
y_test_cat  = to_categorical(y_test)    # shape (N_test, 3)

print("Sample original labels:", y_train[:5])
print("Sample categorical labels:")
print(y_train_cat[:5])

# My Observation: each label becomes a vector like [1,0,0], [0,1,0], [0,0,1] for the three classes after transforming

# 5. MLP with 16 hidden neurons
print("\n5. Training MLP with 16 hidden neurons...")

mlp = MLPClassifier(
    hidden_layer_sizes=(16,),   # one hidden layer of 16 neurons
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)

mlp.fit(X_train, y_train)

mlp_train_score = mlp.score(X_train, y_train)
mlp_test_score  = mlp.score(X_test, y_test)

print("MLP Results:")
print(f"  Training accuracy: {mlp_train_score:.4f}")
print(f"  Test accuracy: {mlp_test_score:.4f}")

# 6. Comparison
print("\n6. Comparison of Models:")
print("="*60)
print(f"Logistic Regression:")
print(f"  Training accuracy: {log_train_score:.4f}")
print(f"  Test accuracy: {log_test_score:.4f}")
print(f"\nMLP (16 hidden neurons):")
print(f"  Training accuracy: {mlp_train_score:.4f}")
print(f"  Test accuracy: {mlp_test_score:.4f}")

"""
My Observation: both models reach high accuracy
If MLP has slightly higher train accuracy and similar test accuracy,
it is more flexible but not strictly necessary for this simple dataset.
"""
