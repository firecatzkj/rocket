# -*- coding:utf-8 -*-
# -*- coding: utf-8 -*-
from sklearn.model_selection import KFold
import numpy as np
from sklearn.ensemble import *
import autosklearn.classification
from autosklearn.metrics import *
from Util import *

alg = 'gradient_boosting'
# TODO:scikit-learn阶段()
print('sklearn使用 ' + alg + ' 算法选参开始\n==================================================')
trainSet, X_train, y_train, testSet, X_test, y_test = getData()
if alg == 'adaboost':
    mdoel = AdaBoostClassifier()
elif alg == 'gradient_boosting':
    mdoel = GradientBoostingClassifier()
elif alg == 'random_forest':
    mdoel = RandomForestClassifier()

kf = KFold(n_splits=10)

toAskVari = []  # 10 folds每次训练后保留的变量的交集
for train_index, test_index in kf.split(trainSet):
    X_train_sk, y_train_sk = X_train.loc[train_index], y_train.loc[train_index]
    mdoel.fit(X_train_sk, y_train_sk)
    liftVari = X_train.columns[np.where(mdoel.feature_importances_ > 0)].tolist()
    if len(toAskVari) == 0:
        toAskVari.extend(liftVari)
    else:
        toAskVari = list(set(toAskVari).intersection(set(liftVari)))

# TODO:auto-sklearn阶段(√)
print('auto-sklearn阶段开始,从上一步共获取' + str(len(toAskVari)) + '个变量。')
print('变量列表为:')
print(toAskVari)
print('=======================')
# TODO:进行进一步降维或做其他处理()
'''
'''
# TODO:训练阶段(√)
cls = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=1800,
    per_run_time_limit=1800,
    include_estimators=['adaboost', 'gradient_boosting', 'random_forest'],
    resampling_strategy='holdout',
    resampling_strategy_arguments={'train_size': 0.67},
    ml_memory_limit=4096
)
X_train_ask = X_train[toAskVari].copy()
y_train_ask = y_train.copy()

X_test_ask = X_test[toAskVari].copy()
y_test_ask = y_test.copy()

cls.fit(X_train_ask, y_train_ask, metric=roc_auc)
getReport(cls, trainSet, X_train_ask, y_train_ask, testSet, X_test_ask, y_test_ask)
