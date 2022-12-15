# Importing the libraries

from sklearn.datasets import load_digits
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn import metrics

# Read the data

digits = load_digits()
X = digits.data
y = digits.target

print(X.shape)
print(y.shape)

# Split data for training and testing, and model y

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)
model = AdaBoostClassifier(learning_rate=4)
model.fit(Xtrain, ytrain)
ymodel = model.predict(Xtest)
print(ymodel)
print(metrics.classification_report(ymodel, ytest))


# Print results
print('\nTrain accuracy: {:.2f}'.format(accuracy_score(ytrain, model.predict(Xtrain))))
print('Test accuracy: {:.2f}\n'.format(accuracy_score(ytest, ymodel)))

# Investigate performance with a validation curve

#param_range = np.arange(start=0, stop=500, step=50)
param_range = np.arange(5)
train_scores, test_scores = validation_curve(AdaBoostClassifier(), X, y, param_name="learning_rate", param_range=param_range, scoring="accuracy")
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
plt.title("Validation Curve With Ada Boost Classifier")
plt.xlabel("Learning Rate")
plt.ylabel("Accuracy Score")
plt.tight_layout()
plt.legend(loc=4)
plt.savefig('../figs/Q4_Ada_Boost_Classifier_Validation.png')