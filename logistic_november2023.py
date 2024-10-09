# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 08:51:32 2023

@author: Gowtham S
"""

# Diabetes Dataset

import pandas as pd

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']

pima = pd.read_csv('diabetes.csv', header=0, names=col_names)

feature_cols = ['pregnant', 'insulin', 'bmi', 'age', 'glucose', 'bp', 'pedigree']

X = pima[feature_cols]

y = pima.label

####################################################################################


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)


####################################################################################

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(random_state=16)

logreg.fit(X_train, y_train)


y_pred = logreg.predict(X_test)

from sklearn import metrics

cnf = metrics.confusion_matrix(y_test, y_pred)

cnf


[True Positive, False Positive]
[False Negative, True Negative]


[115,  10]
[ 25,  42]









