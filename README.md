# HW5-digits-solution

## Instructions

This assignment builds on
[05.02 - Introducing Scikit-Learn](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.02-Introducing-Scikit-Learn.ipynb) by VanderPlas, which produces train/test accuracy of 0.86/0.83 for Gaussian Naive Bayes
classification of the digits dataset.
You can reproduce those results with
```
make template
```
which prints the train/test accuracy:
```
Train accuracy: 0.86
Test accuracy: 0.83
```
and plots the confusion matrix:

<img src="figs/template.png" width="500px">

Submit your solution in this README.md, including any relevant printed output and plots. 
Include a series of independent Python modules in `./src` -- one per question -- that can be run using the 
Makefile to reproduce all results that you report in this README.md. 
Figures should be saved to `./figs` and linked to the README.md as appropriate.
The code in your modules should be concise and nicely documented.
Each question is worth 2 points. 

## Question 1

Use the [classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) to investigate performance by class.
Describe the source of variation in the "support" column.
Is the variation problematic? Justify your assessment quantitatively.
Explain your conclusions in a maximum of 1 or 2 sentences maximum.

#### Result:

![Q1_NB_Class_report](https://user-images.githubusercontent.com/45035308/207872974-d3cb3904-8710-48f0-b032-80ab1eb3c618.png)


## Question 2

Use a decision tree classifier with the same data.
Investigate model performance with a validation curve.
Comment briefly (1 or 2 sentences, maximum) on the results, including a comparison with the results from Question 1.

#### Result:

Train accuracy: 1.00
Test accuracy: 0.84
![Q2_Decision_Tree_Validation_Curve](https://user-images.githubusercontent.com/45035308/207873136-0fe5c1bc-e8e6-4e70-b429-6a22ad78cc62.png)


## Question 3

In [5.08 Random Forests](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.08-Random-Forests.ipynb), VanderPlas performs digits classification with a random forest.
He uses `n_estimators=1000`.
Use a validation curve to investigate the choice of n_estimators.
Comment briefly on the results (including comparison with results above).

#### Result:

Train accuracy: 1.00
Test accuracy: 0.98
![Q3_Random_Forest_Classifier_Validation](https://user-images.githubusercontent.com/45035308/207873174-9babbd12-45b9-4136-a107-eb0c58e6fcd0.png)


## Question 4

Investigate use of
[AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html).
Boosting is discussed in Section 8.2.2 (p345) if ISLR2.
Look at the scikit-learn
[adaboost example](https://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_hastie_10_2.html) for ideas.
Comment briefly on results and your choice of hyperparameters (including comparison with results above).

#### Result:

Train accuracy: 0.55
Test accuracy: 0.57
![Q4_Ada_Boost_Classifier_Validation](https://user-images.githubusercontent.com/45035308/207873211-667af76b-85ea-472d-807a-117c94d7b766.png)


## Question 5

Adapt the use of SVC in cells 18-26 of
[Labeled Faces in the Wild demo in VanderPlas](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.07-Support-Vector-Machines.ipynb).
When selecting optimal hyperparameters, make sure that your range encompasses the
best value.
Comment briefly on results and your choice of hyperparameters (including comparison with results above).

#### Result:

Train accuracy: 0.93
Test accuracy: 0.75
![Q5_SVC_Classifier_Validation](https://user-images.githubusercontent.com/45035308/207873243-544ac9e0-fe27-41dd-bd2f-f92421003afd.png)


