# -*- coding:utf-8 -*-
from sklearn import cross_validation,metrics
from sklearn import svm
#
# train_data,train_target = load(filename)#自定义加载数据函数，返回的是训练数据的数据项和标签项
# train_x,test_x,train_y,test_y = cross_validation.train_test_split(train_data,train_target,test_size=0.2,random_state=27)#把训练集按0.2的比例划分为训练集和验证集
# #start svm
# clf = svm.SVC(C=5.0)
# clf.fit(train_x,train_y)
# predict_prob_y = clf.predict_proba(test_x)#基于SVM对验证集做出预测，prodict_prob_y 为预测的概率
# #end svm ,start metrics
# test_auc = metrics.roc_auc_score(test_y,predict_prob_y)#验证集上的auc值
# print(test_auc)

import numpy as np
from sklearn import metrics

y = np.array([1,1,2,2])
pred = np.array([0.1,0.4,0.35,0.8])
fpr,tpr,thresholds = metrics.roc_curve(y,pred,pos_label=2)
print(metrics.auc(fpr,tpr))