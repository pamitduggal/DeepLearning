from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#1 Loading the data Iris
iris = load_iris()
X = iris.data          # first 4 columns = inputs
y = iris.target        # last column = output (class)

df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(y, iris.target_names)

sns.pairplot(df, hue='species')
plt.show()
# My Observation - petal features separate the classes clearly; sepal features overlap more (Ex1_Figure_2)

#2 Train/test split 75%/25%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

#3 Logistic regression classifier
log_clf = LogisticRegression(max_iter=200)
log_clf.fit(X_train, y_train)

log_train_score = log_clf.score(X_train, y_train)
log_test_score  = log_clf.score(X_test, y_test)

print("Logistic train accuracy:", log_train_score)
print("Logistic test accuracy:", log_test_score)

# My Observation- Iris is easy, so accuracies around 0.95–1.0 are normal (Ex2_Figure_2)
# Logistic train accuracy: 0.9732142857142857
# Logistic test accuracy: 0.9473684210526315

#4 Transforming Data for Neural Network
from tensorflow.keras.utils import to_categorical

y_train_cat = to_categorical(y_train)   # shape (N_train, 3)
y_test_cat  = to_categorical(y_test)    # shape (N_test, 3)

print(y_train[:5])
print(y_train_cat[:5])

# My Observation- each label becomes a vector like [1,0,0], [0,1,0], [0,0,1] for the three classes after transforming (Ex2_Figure_3)

#5 MLP with 16 hidden neurons

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

print("MLP train accuracy:", mlp_train_score)
print("MLP test accuracy:", mlp_test_score)

#6 Comparison

# Logistic: Logistic train accuracy: 0.9732142857142857 , Logistic test accuracy: 0.9473684210526315
# MLP: MLP train accuracy: 0.9732142857142857 , MLP test accuracy: 1.0
# My Observation- both models reach high accuracy
# if MLP has slightly higher train accuracy and similar test accuracy,
# it is more flexible but not strictly necessary for this simple dataset.(Ex2_Figure_4)
