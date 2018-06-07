# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lib.execute import Executor
from tools.mylogger import logger
from lib.select_funcs import drop_useless


class MyExecutor(Executor):
    def feature_select(self, sub_train_set, y):
        tmp = drop_useless(sub_train_set, 'pre_apply_no', 'book_date', 'book_mon')
        return tmp

    def judge_function(self, result):
        """
        1. 测试AUC, KS表现
        2. 单个变量多次入选B1~B10
        3. 变量业务逻辑核查
        :param: train_all的返回值
        :return: 
        """
        # TODO: 对变量集对应的10组测试AUC均值和方差进行评判



def main():
    df = pd.read_csv("data/hl_test_clean.csv", encoding="utf8")
    logger.info("============================Data is ready!============================")
    clf = DecisionTreeClassifier(
        min_weight_fraction_leaf=0.05,
        min_samples_split=0.05,
    )
    myexe = MyExecutor(df, "fpd", clf)
    print(myexe.get_result())


if __name__ == '__main__':
    main()
