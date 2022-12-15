
# Importing the libraries

from sklearn.svm import SVC
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

#from sklearn.datasets import fetch_lfw_people

from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=60)


# Read the data

print(faces.images.shape)
print(faces.data.shape)

X = faces.data
y = faces.target

# Split data for training and testing, and model y

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=42)
model = SVC(kernel='rbf', class_weight=None)
model.fit(Xtrain, ytrain)
ymodel = model.predict(Xtest)
print(ymodel)
print(metrics.classification_report(ymodel, ytest))


# Print results
print('\nTrain accuracy: {:.2f}'.format(accuracy_score(ytrain, model.predict(Xtrain))))
print('Test accuracy: {:.2f}\n'.format(accuracy_score(ytest, ymodel)))

'''(param_grid = {'svc__C': [1, 5, 10, 50],
              'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}
grid = GridSearchCV(model, param_grid)
grid.fit(Xtrain, ytrain)
print(grid.best_params_)'''

from sklearn.model_selection import learning_curve

# Investigate performance with a validation curve


#param_range = np.arange(start=0, stop=500, step=50)
param_range = np.arange(20, 121, 30)
train_sizes, train_scores, test_scores = learning_curve(SVC(), X, y, train_sizes = param_range, cv = 5)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)


plt.subplots(1, figsize=(7,7))
plt.plot(param_range, train_mean, label="Training score", color="black")
plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")
plt.title("Validation Curve With SVC Classifier")
plt.xlabel("Train sizes")
plt.ylabel("")
plt.tight_layout()
plt.legend(loc=4)
plt.savefig('../figs/Q5_SVC_Classifier_Validation.png')