# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lib.execute import Executor
from tools.mylogger import logger
from lib.select_funcs import drop_useless
from lib.judge_funcs import judge_auc_mean_std

class MyExecutor(Executor):
    def feature_select(self, sub_train_set, y):
        tmp = drop_useless(sub_train_set, 'pre_apply_no', 'book_date', 'book_mon')
        return tmp

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
    logger.info("============================Data is ready!============================")
    clf = DecisionTreeClassifier(
        # max_features=4,
        # min_weight_fraction_leaf=0.05,
        min_samples_split=0.05,
        max_leaf_nodes=20
    )
    myexe = MyExecutor(df, "fpd", clf)
    print(myexe.get_result())


if __name__ == '__main__':
    main()
