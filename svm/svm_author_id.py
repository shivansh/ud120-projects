#!/usr/bin/python
"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###

#########################################################
from sklearn import svm
from sklearn.metrics import accuracy_score

# Observations
# ~~~~~~~~~~~~
# Reducing the size of the features and labels to 1% reduces the accuracy to
# 88 % from 98 % (for a linear kernel).
#
# Accuracies with different values of C -
# C=10:    0.61
# C=100:   0.61
# C=1000:  0.82
# C=10000: 0.89
#
# Recordings with (kernel='rbf', C=10000) -
#   training time: 84.527 s
#   prediction time: 8.544 s
#   accuracy: 0.9908987485779295

clf = svm.SVC(kernel='rbf', C=10000)
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time() - t0, 3), "s"

t0 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time() - t0, 3), "s"

accuracy = accuracy_score(pred, labels_test)
print "accuracy:", accuracy

class_1_count = 0
for p in pred:
    if p == 1:
        class_1_count += 1

print "Events in Chris(1) class:", class_1_count
