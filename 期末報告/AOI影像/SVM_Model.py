# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 09:24:35 2023

@author: acer0
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mlxtend.plotting import plot_decision_regions
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
import os
from sklearn.svm import SVC
# %matplotlib inline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

os.chdir(r"D:/哲平/北科大_碩班_AI學程/碩一課程\314337 類神經網路/期末報告/AOI影像")
# TODO: 讀入has資料集
data = pd.read_csv('hash_report.csv')    
data = data.drop(['File Name'], axis=1)
data['label'] = data['label'].replace('NG',0)
data['label'] = data['label'].replace('OK',1)

plt.figure(figsize=(14,10))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(data.astype(float).corr(), cmap = "BrBG", linewidths=0.1, square=True, linecolor='white', annot=True)

ahash = data['ahash'].astype(float) #'ahash'
phash = data['phash'].astype(float) #'dhash'
dhash = data['dhash'].astype(float) #'dhash'
label = data['label'].astype(object)

# 'ahash vs. dhash'
sns.scatterplot(x=ahash, y=dhash , hue=label)
plt.title('ahash vs. dhash')
plt.xlabel('ahash')
plt.ylabel('dhash')
plt.legend(loc='upper right')

data = data.drop(['phash'], axis=1)
features = data.iloc[:,:2]
label = data.iloc[:,2]

# # 使用 scikit-learn 提供的鳶尾花資料庫
# iris = load_iris()
# df = pd.DataFrame(iris['data'], columns = iris['feature_names'])
# df["target"] = iris["target"]
# df = df.drop(["petal width (cm)", "sepal width (cm)"], axis = 1)
# df

# clf = LinearSVC()
# clf = clf.fit(df.drop(["target"], axis = 1), df["target"])

# plot_decision_regions(X=np.array(df.drop(["target"], axis = 1)),
# y=np.array(df["target"]),
# clf=clf)

# SVM線性
clf = LinearSVC()
clf = clf.fit(features, label)
plot_decision_regions(X=np.array(features), y=np.array(label), clf=clf)

# SVM非線性
clf = SVC(kernel="rbf")
clf = clf.fit(features, label)
plot_decision_regions(X=np.array(features), y=np.array(label), clf=clf)

# 模型成果
AccurayScore = accuracy_score(label, clf.predict(features))*100
print('Accuracy Score (All Data)', round(AccurayScore, 2), '%')
# CrossValidationScore = np.average(cross_val_score(clf, features, label, cv=483))*100
# print('CrossValidationScore', round(CrossValidationScore,2), '%')

data['預測值'] = clf.predict(features)
df_predict_wrong = data[data['label'] != data['預測值']]
df_predict_wrong.reset_index(drop=True)

# confusion matrix
mat = pd.DataFrame(confusion_matrix(data['label'],  clf.predict(features)), index=['Label_NG', 'Label_OK'], columns=['Predict_NG', 'Predict_OK'])

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
