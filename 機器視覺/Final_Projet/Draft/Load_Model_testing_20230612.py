# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 16:27:46 2023

@author: acer0
"""

import numpy as np
import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
import seaborn as sns

# Save Model Using Pickle
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle

# Save and Load Machine Learning Models in Python with scikit-learn
# https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/

os.chdir(r"D:/哲平/北科大_碩班_AI學程/碩一課程/314337 類神經網路/機器視覺/Final_Projet")
# TODO: 讀入has資料集
data = pd.read_csv('report.csv')    
data = data.drop(['File Name'], axis=1)

# train_test_split
train, test = train_test_split(data, train_size=0.8, test_size = 0.2, shuffle=True, stratify=data['label'], random_state=1)

train_features = train.iloc[:,:4]
train_target = train.iloc[:,4]

test_features = test.iloc[:,:4]
test_target = test.iloc[:,4]

filename = 'MLP_model.sav'

clf = pickle.load(open(filename, 'rb'))
# result = clf.score(test_features, test_target)
# print('Accuracy Score (Testing Data)', round(result*100, 2), '%')

AccurayScore = accuracy_score(test_target, clf.predict(test_features))*100
print('Accuracy Score (Testing Data)', round(AccurayScore, 2), '%')
print('Accuracy Score (Training Data)', round(accuracy_score(train_target, clf.predict(train_features))*100, 2), '%')
print('Accuracy Score (All Data)', round(accuracy_score(data.iloc[:,4], clf.predict(data.iloc[:,:4]))*100, 2), '%')
# CrossValidationScore = np.average(cross_val_score(clf, data.iloc[:,:3], data.iloc[:,3], cv=483))*100
# print('CrossValidationScore', round(CrossValidationScore,2), '%')

# print(len(clf.predict(test_features)), len(test_features))
clf.predict(test_features)

test['預測值'] = clf.predict(test_features)
df_predict_wrong = test[test['label'] != test['預測值']]
df_predict_wrong.reset_index(drop=True)

# confusion matrix
mat = pd.DataFrame(confusion_matrix(test['label'],  clf.predict(test_features)), index=['Label_NG', 'Label_OK'], columns=['Predict_NG', 'Predict_OK'])

print('NG')
precision = (mat['Predict_NG']['Label_NG']/sum(mat['Predict_NG']))*100
print('Precision精確率', round(precision,2), '%')
recall = (mat['Predict_NG']['Label_NG']/sum(mat.iloc[0]))*100
print('Recall', round(recall,2), '%')
F1 = 2 * (precision * recall) / (precision + recall)
print('F1-Score', round(F1,2), '%')

print('OK')
precision = (mat['Predict_OK']['Label_OK']/sum(mat['Predict_OK']))*100
print('Precision精確率', round(precision,2), '%')
recall = (mat['Predict_OK']['Label_OK']/sum(mat.iloc[1]))*100
print('Recall', round(recall,2), '%')
F1 = 2 * (precision * recall) / (precision + recall)
print('F1-Score', round(F1,2), '%')

mat = pd.concat([mat, pd.DataFrame({'Predict_NG':[sum(mat['Predict_NG'])], 'Predict_OK':[sum(mat['Predict_OK'])]}, index=['Sum'])])
mat['Sum'] = pd.Series([sum(mat.iloc[0]), sum(mat.iloc[1]), sum(mat.iloc[2])], index=['Label_NG', 'Label_OK', 'Sum'])
mat

# 只適用solver=adam或sgd
pd.DataFrame(clf.loss_curve_)
# 只適用solver=adam或sgd
plt.plot(clf.loss_curve_,)
plt.xlabel('Iteration')
plt.ylabel('loss/error')
plt.show()

print(f'最後的loss值:{round(clf.loss_,2)}')

