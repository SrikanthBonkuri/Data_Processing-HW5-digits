# Importing the libraries

from sklearn.datasets import load_digits
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#from sklearn.evaluation import plot
from sklearn.metrics import precision_recall_fscore_support

# Read the data
digits = load_digits()
X = digits.data
y = digits.target

# Train/test split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)

# Train and test the model
model = GaussianNB()
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)

# Print results
print('\nTrain accuracy: {:.2f}'.format(accuracy_score(ytrain, model.predict(Xtrain))))
print('Test accuracy: {:.2f}\n'.format(accuracy_score(ytest, y_model)))

print(confusion_matrix(ytest, y_model))

print(classification_report(ytest, y_model))

'''target_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

cr_nb = plot.ClassificationReport.from_raw_data(
    ytest, y_model, target_names=target_names
)

print(cr_nb) '''




# Plot the confusion matrix
mat = confusion_matrix(ytest, y_model)
sns.heatmap(mat, square=True, annot=True, cbar=False, cmap='OrRd')
plt.title('Gaussian Naive Bayes ')
plt.xlabel('predicted value')
plt.ylabel('true value')
#plt.savefig("figs/template.png")
plt.show()


plt.figure(figsize=(10, 10))
xticks = ['precision', 'recall', 'f1-score', 'support']
yticks = list(np.unique(ytest))
yticks += ['avg']

rep = np.array(precision_recall_fscore_support(ytest, y_model)).T
avg = np.mean(rep, axis=0)
avg[-1] = np.sum(rep[:, -1])
rep = np.insert(rep, rep.shape[0], avg, axis=0)

# Plot the classification report
sns.heatmap(rep, annot=True, cbar=False, 
                xticklabels=xticks, yticklabels=yticks, ax=None)

plt.title('Gaussian NB Report ')
plt.xlabel('Metrics')
plt.ylabel('Classes')
plt.savefig("../figs/Q1_NB_Class_report.png")
plt.show()
