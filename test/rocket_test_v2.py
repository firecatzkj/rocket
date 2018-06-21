# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
#from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lib.execute import Executor
from tools.mylogger import logger
from lib.select_funcs import drop_useless, drop_by_iv
from lib.judge_funcs import judge_auc_mean_std
from lib.utils import getReport
from tpot import TPOTClassifier


class MyExecutor(Executor):
    def feature_select(self, sub_train_set, y):
        #logger.info("Start filer Variables total: {}".format(len(sub_train_set.columns)))
        #tmp1 = drop_useless(sub_train_set, 'pre_apply_no', 'book_date', 'book_mon')
        #sub_train_set = sub_train_set[tmp1]
        #tmp2 = drop_by_iv(sub_train_set, "fpd")
        #logger.info("Stop filter Variables total: {}".format(len(tmp2)))
        tmp2 = [
            'hl_call_domesitc_cnt_2m',
            'hl_contact_early_morning_cnt_5m',
            'hl_phone_silent_frequentcy',
            'hl_contact_night_pct',
            'hl_transactions_total_amt_5m',
            'hl_region_call_cnt_max_uniq_num_cnt',
            'hl_region_call_out_cnt_max_avg_call_in_time',
            'hl_contact_morning_cnt_5m',
            'hl_region_call_in_time_max_avg_call_in_time',
            'hl_transactions_total_amt_2m',
            'hl_contact_night_cnt_5m',
            'hl_phone_num_used_time_months',
            'hl_region_call_cnt_max_avg_callin_time',
            'hl_region_call_in_time_max_uniq_num_cnt',
            'hl_region_call_in_cnt_max_avg_call_out_time',
            'hl_transactions_min_5m',
            'hl_region_call_out_time_max_avg_call_out_time',
            'fpd'
        ]
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
    clf = XGBClassifier(learning_rate=0.01,
                        max_depth=7,
                        min_child_weight=15,
                        n_estimators=100,
                        nthread=1,
                        subsample=0.6500000000000001)
    myexe = MyExecutor(df, "fpd", clf)
    #leftVaris = myexe.get_result()
    #leftVaris = leftVaris[leftVaris.values > 7].keys()
    #print(leftVaris)
    leftVaris = [
            'hl_call_domesitc_cnt_2m',
            'hl_contact_early_morning_cnt_5m',
            'hl_phone_silent_frequentcy',
            'hl_contact_night_pct',
            'hl_transactions_total_amt_5m',
            'hl_region_call_cnt_max_uniq_num_cnt',
            'hl_region_call_out_cnt_max_avg_call_in_time',
            'hl_contact_morning_cnt_5m',
            'hl_region_call_in_time_max_avg_call_in_time',
            'hl_transactions_total_amt_2m',
            'hl_contact_night_cnt_5m',
            'hl_phone_num_used_time_months',
            'hl_region_call_cnt_max_avg_callin_time',
            'hl_region_call_in_time_max_uniq_num_cnt',
            'hl_region_call_in_cnt_max_avg_call_out_time',
            'hl_transactions_min_5m',
            'hl_region_call_out_time_max_avg_call_out_time'
        ]

    X_train = trainSet[leftVaris].copy()
    y_train = trainSet['fpd'].copy()
    X_test = testSet[leftVaris].copy()
    y_test = testSet['fpd'].copy()
    # AutoSklearn阶段:
    pipeline_optimizer = TPOTClassifier(generations=5,
                                        population_size=20,
                                        cv=4,
                                        random_state=42,
                                        verbosity=2)
    pipeline_optimizer.fit(X_train, y_train)
    # print(pipeline_optimizer.score(X_test, y_test))
    pipeline_optimizer.export('tpot_exported_pipeline.py')
    getReport(pipeline_optimizer, trainSet, X_train, y_train, testSet, X_test, y_test)


if __name__ == '__main__':
    main()
