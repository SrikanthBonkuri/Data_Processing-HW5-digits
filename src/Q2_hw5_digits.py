# Importing the libraries

from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
import numpy as np


# Read the data

digits = load_digits()
X = digits.data
y = digits.target

# Create the decision tree

clf = DecisionTreeClassifier()
clf = clf.fit(X, y)


# Investigate performance with a validation curve

param_range = np.arange(10)
train_scores, test_scores = validation_curve(DecisionTreeClassifier(), X, y, param_name="random_state", param_range=param_range, scoring='accuracy')
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)


# Plot the curve

plt.subplots(1, figsize=(7,7))
plt.plot(param_range, train_mean, label="Training score", color="black")
plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")
plt.title("Validation Curve With Decision Tree Classifier")
plt.xlabel("Class Weights")
plt.ylabel("Accuracy Score")
plt.tight_layout()
plt.legend(loc=7)
plt.savefig('../figs/Q2_Decision_Tree_Validation_Curve.png')