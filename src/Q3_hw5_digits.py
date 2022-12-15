# Importing the libraries

from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics

# Read the data

digits = load_digits()
X = digits.data
y = digits.target

# Split data for training and testing, and model y

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)
model = RandomForestClassifier()
model.fit(Xtrain, ytrain)
ypred = model.predict(Xtest)
print(metrics.classification_report(ypred, ytest))

# Investigate performance with a validation curve

param_range = np.arange(start=1, stop=1100, step=100)
train_scores, test_scores = validation_curve(RandomForestClassifier(), X, y, param_name="n_estimators", param_range=param_range, scoring="accuracy")
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
plt.title("Validation Curve With Random Forest Classifier")
plt.xlabel("n_estimators")
plt.ylabel("Accuracy Score")
plt.tight_layout()
plt.legend(loc=7)
plt.savefig('../figs/Q3_Random_Forest_Classifier_Validation.png')