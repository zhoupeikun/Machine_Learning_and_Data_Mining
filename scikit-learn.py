# Import classes
import numpy as np
# import matplotlib.pyplot as plt
from sklearn import datasets

# Load and parse the data file
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
np.unique(iris_y)

# Split iris data in train and test data
# A random permutation, to split the data randomly
np.random.seed(0)
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test  = iris_X[indices[-10:]]
iris_y_test  = iris_y[indices[-10:]]

# Create figure
# plt.figure(1)

# Train a k-nearest-neighbor model
# Create and fit a k-nearest-neighbor classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_y_train)


# Evaluate model on test instances and comput test error
from sklearn.metrics import accuracy_score
print knn.predict(iris_X_test)
accuracy_score_knn = accuracy_score(iris_y_test, knn.predict(iris_X_test))
print "{:10.20f}".format(accuracy_score_knn)


# 1
# Linear SVC
# Create and fit a LinearSVC classifier
from sklearn.svm import SVC, LinearSVC
linearsvc = LinearSVC()
linearsvc = linearsvc.fit(iris_X_train, iris_y_train)
accuracy_score_linearsvc = accuracy_score(iris_y_test, linearsvc.predict(iris_X_test))
print "{:10.20f}".format(accuracy_score_linearsvc)

#
#  2
#  Light SVC
# from gat.classifiers import SVC_Light
# svc_light = SVC_Light(kernel = 'linear')
# svc_light = svc_light.fit(iris_X_train, iris_y_train)
# accuracy_score_svclight = accuracy_score(iris_y_test, svc_light.predict(iris_X_test))
# print "{:10.20f}".format(accuracy_score_svclight)


# 3
# Random forest classifier
# import nltk
# from nltk.classify.scikitlearn import SklearnClassifier
# random_forest = SklearnClassifier()
# random_forest.fit(iris_X_train, iris_y_train)
# accuracy_score_randomforest = accuracy_score(iris_y_test, random_forest.predict(iris_X_test))
# print "{:10.20f}".format(accuracy_score_randomforest)

# 4
# Ada boost classifer
# from sklearn.ensemble import AdaBoostClassifier
# ada_boost = AdaBoostClassifer()
# ada_boost.fit(iris_X_train, iris_y_train)
# accuracy_score_adaboost = accuracy_score(iris_y_test, ada_boost.predict(iris_X_test))
# print "{:10.20f}".format(accuracy_score_adaboost)

# 5
# LogisticRegression Classifier
from sklearn.linear_model import LogisticRegression
logistic_regression = LogisticRegression()
logistic_regression = logistic_regression.fit(iris_X_train, iris_y_train)
accuracy_score_logistic = accuracy_score(iris_y_test, logistic_regression.predict(iris_X_test))
print "{:10.20f}".format(accuracy_score_logistic)


# Cross-validation evaluation
from sklearn import cross_validation
from sklearn import metrics

#knn
scores_knn = cross_validation.cross_val_score(knn, iris.data, iris.target, scoring="f1_macro")
print "Knn cross_validation score: " + "{:10.20f}".format(scores_knn.mean())
#linear_SVC
scores_linearsvc = cross_validation.cross_val_score(linearsvc, iris.data, iris.target, scoring="f1_macro")
print "LinearSVC cross_validation score: " + "{:10.20f}".format(scores_linearsvc.mean())

logistic_regression = LogisticRegression()
logistic_regression = logistic_regression.fit(iris_X_train, iris_y_train)
scores_logistic = cross_validation.cross_val_score(logistic_regression, iris.data, iris.target, scoring="f1_macro")
print "LogisticRegression cross_validation score: " + "{:10.20f}".format(scores_logistic.mean())


