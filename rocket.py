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
        return drop_useless(sub_train_set, 'pre_apply_no', 'book_date', 'book_mon')


def main():
    df = pd.read_csv("data/hl_test_clean.csv", encoding="utf8")
    logger.info("============================Data is ready!============================")
    clf = DecisionTreeClassifier(
        min_weight_fraction_leaf=0.05,
        min_samples_split=0.05,
    )
    myexe = MyExecutor(df, "fpd", clf)
    myexe.train_all()


if __name__ == '__main__':
    main()
