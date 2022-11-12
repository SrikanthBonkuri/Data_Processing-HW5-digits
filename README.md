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

## Question 2

Use a decision tree classifier with the same data.
Investigate model performance with a validation curve.
Comment briefly (1 or 2 sentences, maximum) on the results, including a comparison with the results from Question 1.

## Question 3

In [5.08 Random Forests](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.08-Random-Forests.ipynb), VanderPlas performs digits classification with a random forest.
He uses `n_estimators=1000`.
Use a validation curve to investigate the choice of n_estimators.
Comment briefly on the results (including comparison with results above).

## Question 4

Investigate use of
[AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html).
Boosting is discussed in Section 8.2.2 (p345) if ISLR2.
Look at the scikit-learn
[adaboost example](https://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_hastie_10_2.html) for ideas.
Comment briefly on results and your choice of hyperparameters (including comparison with results above).

## Question 5

Adapt the use of SVC in cells 18-26 of
[Labeled Faces in the Wild demo in VanderPlas](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.07-Support-Vector-Machines.ipynb).
When selecting optimal hyperparameters, make sure that your range encompasses the
best value.
Comment briefly on results and your choice of hyperparameters (including comparison with results above).
