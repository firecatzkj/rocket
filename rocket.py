# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from lib.execute import Executor
from tools.mylogger import logger
from lib.select_funcs import drop_useless, drop_by_iv
from lib.judge_funcs import judge_auc_mean_std
from lib.utils import getReport
import autosklearn.classification
from autosklearn.metrics import *


class MyExecutor(Executor):
    def feature_select(self, sub_train_set, y):
        logger.info("Start filer Variables total: {}".format(len(sub_train_set.columns)))
        tmp1 = drop_useless(sub_train_set, 'pre_apply_no', 'book_date', 'book_mon')
        sub_train_set = sub_train_set[tmp1]
        tmp2 = drop_by_iv(sub_train_set, "fpd")
        logger.info("Stop filter Variables total: {}".format(len(tmp2)))
        return tmp2

    def judge_function_model(self, result):
        result = [i for i in result]
        model_score = []
        for i in result:
            current_auc_list = np.array(i)[:, 0]
            score = judge_auc_mean_std(current_auc_list.mean(), current_auc_list.std())
            model_score.append(score)
        best_model_index = pd.Series(model_score).idxmax()
        return result[best_model_index]


def main():
    df = pd.read_csv("data/hl_test_clean.csv", encoding="utf8")
    df['book_date'] = pd.to_datetime(df['book_date'])
    trainSet = df[(df['book_date'] >= '2017-04-01') & (df['book_date'] <= '2017-07-20')].reset_index(drop=True)
    testSet = df[(df['book_date'] >= '2017-07-20') & (df['book_date'] <= '2017-08-31')].reset_index(drop=True)

    logger.info("============================Data is ready!============================")
    clf = RandomForestClassifier(
        n_estimators=10,
        max_features=10,
        max_depth=4,
        min_samples_split=0.05,
    )
    myexe = MyExecutor(df, "fpd", clf)
    #print(myexe.get_result())

    leftVaris = myexe.get_result()
    leftVaris = leftVaris[leftVaris.values > 7].keys()
    X_train = trainSet[leftVaris].copy()
    y_train = trainSet['fpd'].copy()
    X_test = testSet[leftVaris].copy()
    y_test = testSet[leftVaris].copy()

    # AutoSklearn阶段:
    cls = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=62,
        per_run_time_limit=60,
        include_estimators=['adaboost'],
        resampling_strategy='holdout',
        resampling_strategy_arguments={'train_size': 0.67}
    )

    cls.fit(X_train, y_train, metric=roc_auc)
    getReport(cls, trainSet, X_train, y_train, testSet, X_test, y_test)


if __name__ == '__main__':
    main()
