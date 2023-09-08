# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 15:44:17 2023

@author: acer0
"""

# 載入套件
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
# from mpl_toolkits.mplot3d import Axes3D

import pickle

def sigmoid(v):
    return 1 / (1 + np.exp(-v))

os.chdir(r"D:/哲平/北科大_碩班_AI學程/碩一課程/314337 類神經網路/機器視覺/Final_Projet")
# TODO: 讀入has資料集
data = pd.read_csv('report.csv')    
data = data.drop(['File Name'], axis=1)

# train_test_split
train, test = train_test_split(data, train_size=0.9, test_size = 0.1, shuffle=True, stratify=data['label'], random_state=1)

train_features = train.iloc[:,:4]
train_target = train.iloc[:,4]

test_features = test.iloc[:,:4]
test_target = test.iloc[:,4]

# 多層感知器分類
clf = MLPClassifier(hidden_layer_sizes=(8,), activation='logistic', solver='adam', learning_rate='constant', learning_rate_init=0.2, max_iter=200, early_stopping=True, batch_size=200) 
clf

clf.fit(train_features, train_target)

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

# 權重矩陣
# print(clf.coefs_)
# # 偏差向量 bias
# print(clf.intercepts_)
# print(clf.classes_)
# print(clf.n_layers_)
print(f'最後的loss值:{round(clf.loss_,2)}')

# 儲存模型
# save the model to disk
filename = 'MLP_Model.sav'
pickle.dump(clf, open(filename, 'wb'))

sys.exit()

W1 = clf.coefs_[0].tolist()
W1.insert(0,clf.intercepts_[0].tolist())
pd.DataFrame(W1, index=['Bias','Weight_ahash','Weight_phash', 'Weight_dhash'])

index = [f'W{i}' for i in range(0, 10)]
index.insert(0, 'bias')
W2 = clf.coefs_[1].tolist()
W2.insert(0,clf.intercepts_[1].tolist())
pd.DataFrame(W2, index=index)

# 視覺化（丟入150筆資料繪圖）
AllData_Features = pd.concat([train_features, test_features], ignore_index=True)
# 插入Bias
AllData_Features.insert(0, 'Bias', list([1]*6982))
# 轉成numpy array
AllData_Features = np.array(AllData_Features)

# 內積
HiddenLayer = list()
for _input in AllData_Features: 
  HiddenLayer.append(sigmoid(np.matmul(_input, W1))) # W1為隱藏層權重

# 插入Bias
df_HiddenLayer = pd.DataFrame(HiddenLayer, columns=[f'H{i}' for i in range(0,10)])
df_HiddenLayer.insert(0, 'Bias', list([1]*6982))
df_HiddenLayer

# 轉成numpy array
HiddenLayer = np.array(df_HiddenLayer)

# 內積 (沒有Sigmoid)
# Output_Predict = list()
# for _input in HiddenLayer:
#   Output_Predict.append(np.matmul(_input, W2))
# Output_Predict
   
# 內積 (有Sigmoid)
Output_Predict = list()
for _input in HiddenLayer:
  Output_Predict.append(sigmoid(np.matmul(_input, W2)))

df_OutputPredictLayer = pd.DataFrame(Output_Predict, columns=[f'predict{i}' for i in range(0,1)])

df_OutputPredictLayer['label'] = data['label']

# df_OutputPredictLayer = df_OutputPredictLayer[df_OutputPredictLayer['label']=='NG']

# 繪製直線
sns.scatterplot(x=df_OutputPredictLayer['predict0'], y=list([0]*6982), hue=df_OutputPredictLayer['label'])
plt.title('sepal length vs. sepal width')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(['NG', 'OK'], loc='upper right')

# 顯示圖形
plt.show()

