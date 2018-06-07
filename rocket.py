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
        # tmp = [
        #     "hl_phone_silent_frequentcy",
        #     "hl_contact_early_morning_cnt_5m",
        #     "hl_transactions_min_5m",
        #     "hl_contact_night_pct",
        #     "hl_region_call_out_cnt_max_avg_call_in_time",
        #     "hl_transactions_total_amt_5m",
        #     "hl_call_cnt_mid_2m",
        #     "hl_contact_bank_callout_len_total",
        #     "fpd"
        # ]
        return tmp


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
