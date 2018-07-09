# -*- coding:utf-8 -*-
import pandas as pd
from copy import copy
from xgboost.sklearn import XGBClassifier
from lib.utils import getReport
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from xgboost import XGBClassifier









import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from scipy import stats

def ks_score(model, X_train, y_train):
    X_pred = model.predict_proba(X_train)
    X_train['prob'] = X_pred[:, 1]
    good = X_train[X_train['fpd'] == 0]
    bad = X_train[X_train['fpd'] == 1]
    trainKS = stats.ks_2samp(good['prob'], bad['prob']).statistic
    return trainKS






def run():
    df = pd.read_csv("../analysis/clean_data_filtered.csv", encoding="utf8")
    trainSet = df[
        df["book_mon"].isin(['2017-05', '2017-06', '2017-07'])
    ].reset_index(drop=True)

    # 这里不一样,训练集用过滤过噪声的,测试集用原来的样本
    df_test = pd.read_csv("../data/model_mix/clean_data.csv", encoding="utf8")
    testSet = df_test[
        df_test["book_mon"] == "2017-08"
        ].reset_index(drop=True)

    leftVaris = [
        "hl_contact_bank_callin_cnt_avg",
        "td_cnt_offloan_1m",
        "hl_call_early_morning_pct_5m",
        "hl_transactions_min_2m",
        "hl_call_cnt_avg_5m",
        "hl_call_domesitc_cnt_2m",
        "hl_transactions_total_amt_5m",
        "hl_region_call_out_cnt_max_avg_call_out_time",
        "hl_contact_loan_callout_len_avg",
        "hl_contact_linkman_callin_cnt_pct_5m",
        "credit",
        "CRD_LINE_SUM",
        "CRD_CNT_INTEGRAL_000_CSM_3M",
        "hl_call_cnt_mid_5m",
        "hl_contact_bank_callin_cnt_total",
        "hl_callin_len_avg_5m",
        "hl_call_domesitc_cnt_5m",
        "hl_transactions_max_2m",
        "CRD_LINE_MAX",
        "hl_smses_send_cnt_5m"
    ]
    X_train = trainSet[leftVaris].copy()
    y_train = trainSet['fpd'].copy()
    X_test = testSet[leftVaris].copy()
    y_test = testSet['fpd'].copy()

    ##################################################
    model = XGBClassifier(
        learning_rate=0.13,
        max_depth=2,
        #min_child_weight=10,
        n_estimators=86,
        #nthread=1,
        subsample=0.25
    )



    #  max_depth=2, n_estimators=90  ==> 训练集ks 0.539 测试集ks 0.265
    # {'max_depth':2 ,'learning_rate': 0.12, 'n_estimators': 86} ==> 训练集ks 0.554 测试集ks 0.262


    # param_test = {
    #     # 'n_estimators': list(range(50, 100, 2)),
    #     # 'max_depth': list(range(2, 6, 1)),
    #     'learning_rate': list(np.arange(0.1, 0.7, 0.01)),
    #     'n_estimators': list(range(60, 100, 2))
    # }
    # print(1)
    # grid_search = GridSearchCV(
    #     estimator=model,
    #     param_grid=param_test,
    #     scoring='roc_auc',
    #     cv=5,
    #     verbose=9,
    #     n_jobs=2
    # )
    # grid_search.fit(X_train, y_train)
    # print(grid_search.grid_scores_)
    # print(grid_search.best_params_)
    # print(grid_search.best_score_)

    model.fit(X_train, y_train)
    getReport(model, trainSet, X_train, y_train, testSet, X_test, y_test)


if __name__ == '__main__':
    # 设置boosting迭代计算次数
    run()

    # 训练集ks
    # 0.458162015571691
    # 测试集ks
    # 0.15044400557941692
    # KS之差为: 0.30771800999227406

    # 训练集ks
    # 0.5449479722330581
    # 测试集ks
    # 0.08852490712480493
    # : 0.4564230651082532KS之差为