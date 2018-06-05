# -*- coding:utf-8 -*-
import pandas as pd
cols = [
    "hl_transactions_min_5m",
    "hl_contact_bank_callout_len_avg",
    "hl_contact_early_morning_cnt_5m",
    "hl_phone_silent_frequentcy",
    "hl_contact_loan_num",
    "hl_region_unique_num_max_uniq_num_cnt"]
df_check = pd.read_table("model_online.data", sep="\t")
df_offline = pd.read_table("model_offline.data", sep="\t")
df_offline = df_offline.drop_duplicates(["hl_object_id"])

res = pd.merge(df_check, df_offline, how="left", left_on="object_id", right_on="hl_object_id")
for c in cols:
    diff = res[c] - res[str(c) + "_off"]
    res[str(c) + "_diff"] = diff
res.to_csv("result.csv", index=False)
pd.DataFrame.to_csv()



a = [
    'hl_call_early_morning_pct_5m',
    'hl_contact_bank_callout_cnt_avg',
    'hl_contact_bank_callout_cnt_avg_off',
    'hl_contact_credit_num',
    'hl_contact_credit_num_off',
    'hl_contact_linkman_callout_time_pct_2m',
    'hl_contact_linkman_callout_time_pct_2m_off',
    'hl_contact_loan_callout_len_total',
    'hl_contact_loan_callout_len_total_off',
    'hl_contact_loan_num',
    'hl_contact_loan_num_off',
    'hl_contact_morning_cnt_5m',
    'hl_contact_morning_cnt_5m_off',
    'hl_phone_silent_frequentcy',
    'hl_phone_silent_frequentcy_off',
    'hl_region_unique_num_max_avg_call_in_time',
    'hl_region_unique_num_max_avg_call_in_time_off',
    'hl_smses_cnt_sum_5m',
    'hl_smses_cnt_sum_5m_off',
    'objectid',
    'hl_object_id'
]










