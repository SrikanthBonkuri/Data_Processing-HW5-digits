# Importing the libraries

from sklearn.datasets import fetch_lfw_people
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

# Read the data

faces = fetch_lfw_people(min_faces_per_person=60, return_X_y=True)
X = faces.data
y = faces.target

# Split data for training and testing, and model y

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=42)
model = SVC(kernel='rbf', class_weight='balanced')
model.fit(Xtrain, ytrain)
ymodel = model.predict(Xtest)
print(ymodel)
print(metrics.classification_report(ymodel, ytest))
'''
param_grid = {'svc__C': [1, 5, 10, 50],
              'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}
grid = GridSearchCV(model, param_grid)
grid.fit(Xtrain, ytrain)
print(grid.best_params_)

# Investigate performance with a validation curve
#param_range = np.arange(start=0, stop=500, step=50)
param_range = np.arange(5)
train_scores, test_scores = validation_curve(SVC(), X, y, param_name="learning_rate", param_range=param_range, scoring="accuracy")
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot the curve
plt.cla()
plt.clf()
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
plt.savefig('figs/4_Ada_Boost_Classifier_Validation.png')'''