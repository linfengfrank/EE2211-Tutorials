#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 16:24:42 2020

@author: thomas
"""

from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
    

# load data
iris_dataset = load_iris()

# split dataset
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], 
                                                    iris_dataset['target'], 
                                                    test_size=0.20, 
                                                    random_state=0)

# fit tree
dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4)
dtree = dtree.fit(X_train, y_train)

# predict
y_trainpred = dtree.predict(X_train)
y_testpred = dtree.predict(X_test)

# print accuracies
print("Training accuracy: ", metrics.accuracy_score(y_train, y_trainpred))
print("Test accuracy: ", metrics.accuracy_score(y_test, y_testpred))    

# Plot tree
tree.plot_tree(dtree)
plt.savefig('FigTut9_Q4.eps')
plt.show()
