# -*- coding:utf-8 -*-
import pandas as pd
from lib.utils import getReport
from xgboost import XGBClassifier


df = pd.read_csv("data/hl_test_clean.csv", encoding="utf8")
df['book_date'] = pd.to_datetime(df['book_date'])
trainSet = df[(df['book_date'] >= '2017-04-01') & (df['book_date'] <= '2017-07-20')].reset_index(drop=True)
testSet = df[(df['book_date'] >= '2017-07-20') & (df['book_date'] <= '2017-08-31')].reset_index(drop=True)
leftVaris = ['hl_call_domesitc_cnt_2m',
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
model = XGBClassifier(learning_rate=0.01,
                      max_depth=7,
                      min_child_weight=15,
                      n_estimators=100,
                      nthread=1,
                      subsample=0.6500000000000001)
model.fit(X_train, y_train)

print(model.feature_importances_)
getReport(model, trainSet, X_train, y_train, testSet, X_test, y_test)

