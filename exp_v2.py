# -*- coding:utf-8 -*-
import pandas as pd
from lib.utils import getReport
from sklearn.ensemble import GradientBoostingClassifier


df = pd.read_csv("data/hl_test_clean.csv", encoding="utf8")
df['book_date'] = pd.to_datetime(df['book_date'])
trainSet = df[(df['book_date'] >= '2017-04-01') & (df['book_date'] <= '2017-07-20')].reset_index(drop=True)
testSet = df[(df['book_date'] >= '2017-07-20') & (df['book_date'] <= '2017-08-31')].reset_index(drop=True)
leftVaris = [
    'hl_call_domesitc_cnt_2m',
    'hl_phone_silent_frequentcy',
    'hl_region_unique_num_max_uniq_num_cnt',
    'hl_contact_night_cnt_5m',
    'hl_phone_num_used_time_months',
    'hl_transactions_total_amt_5m',
    'hl_transactions_min_5m',
    'hl_transactions_min_2m',
    'hl_contact_early_morning_cnt_5m',
    'hl_contact_afternoon_cnt_5m',
    'hl_region_call_in_time_max_uniq_num_cnt',
    'hl_region_call_out_cnt_max_avg_call_out_time',
    'hl_region_unique_num_max_avg_call_in_time',
    'hl_region_call_in_cnt_max_avg_call_out_time',
    'hl_transactions_total_amt_2m',
    'hl_contact_contacts_amt_5m',
    'hl_region_call_out_cnt_max_avg_call_in_time',
    'hl_contact_loan_callout_len_avg',
    'hl_contact_morning_cnt_5m',
    'hl_region_call_out_cnt_max_uniq_num_cnt',
    'hl_callin_len_avg_5m',
    'hl_region_call_in_cnt_max_uniq_num_cnt',
    'hl_callin_len_avg_2m',
    'hl_transactions_max_2m',
    'hl_call_cnt_avg_2m',
    'hl_call_cnt_mid_2m',
    'hl_region_call_in_time_max_avg_call_in_time',
    'hl_contact_loan_callin_len_total',
    'hl_contact_weekend_cnt_5m',
    'hl_contact_bank_callout_len_avg',
    'hl_call_cnt_mid_5m',
    'hl_contact_night_pct',
    'hl_region_call_in_cnt_max_avg_call_in_time',
    'hl_callin_cnt_avg_2m',
    'hl_region_call_out_time_max_avg_call_out_time',
    'hl_region_call_out_time_max_uniq_num_cnt',
    'hl_region_call_cnt_max_uniq_num_cnt',
    'hl_region_call_cnt_max_avg_callin_time',
    'hl_region_call_in_cnt_max_call_in_time',
    'hl_region_unique_num_max_call_out_time',
    'hl_region_call_out_time_max_avg_call_in_time',
    'hl_region_call_in_time_max_avg_call_out_time',
    'hl_region_call_out_time_max_call_in_time',
    'hl_contact_noon_cnt_5m',
    'hl_callout_len_avg_5m',
    'hl_region_call_in_time_max_call_in_time',
    'hl_region_call_out_cnt_max_call_out_cnt',
    'hl_region_call_out_cnt_max_call_out_cnt_pct',
    'hl_region_call_out_cnt_max_call_out_time',
    'hl_region_call_out_time_max_call_out_time',
    'hl_region_unique_num_max_call_in_time',
    'hl_call_cnt_max_2m',
    'hl_region_unique_num_max_call_out_cnt',
    'hl_transactions_max_5m',
    'hl_region_call_cnt_max_avg_callout_time',
    'hl_contact_eachother_cnt',
    'hl_contact_bank_callin_len_avg',
    'hl_call_domesitc_cnt_5m',
    'hl_contact_workday_cnt_5m',
    'hl_region_call_in_cnt_max_call_in_time_pct',
    'hl_region_call_out_time_max_call_out_cnt',
    'hl_region_call_cnt_max_callin_time_pct',
    'hl_region_call_out_time_max_call_out_cnt_pct',
    'hl_region_unique_num_max_call_out_cnt_pct',
    'hl_region_call_in_cnt_max_call_in_cnt',
    'hl_region_call_in_cnt_max_call_out_time',
    'hl_region_contact_loc_amt',
    'hl_call_cnt_max_5m',
    'hl_region_call_out_cnt_max_call_in_time_pct',
    'hl_region_call_out_cnt_max_call_in_time',
    'hl_region_unique_num_max_avg_call_out_time',
    'hl_region_call_in_time_max_call_in_time_pct',
    'hl_region_unique_num_max_call_in_time_pct',
    'hl_region_call_out_time_max_call_in_time_pct',
    'hl_callout_cnt_avg_2m',
    'hl_call_cnt_avg_5m',
    'hl_region_call_in_cnt_max_call_out_cnt',
    'hl_region_call_cnt_max_callout_cnt',
    'hl_region_call_in_time_max_call_in_cnt']

X_train = trainSet[leftVaris].copy()
y_train = trainSet['fpd'].copy()
X_test = testSet[leftVaris].copy()
y_test = testSet['fpd'].copy()

model = GradientBoostingClassifier(learning_rate=0.001,
                                   max_depth=8,
                                   max_features=0.7000000000000001,
                                   min_samples_leaf=20,
                                   min_samples_split=11,
                                   n_estimators=100,
                                   subsample=0.2)
model.fit(X_train, y_train)
print(model.feature_importances_)
getReport(model, trainSet, X_train, y_train, testSet, X_test, y_test)

