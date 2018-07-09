# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from copy import copy
from sklearn.ensemble import RandomForestClassifier
from lib.execute import Executor
from tools.mylogger import logger
from lib.select_funcs import drop_useless, drop_by_iv
from lib.judge_funcs import judge_auc_mean_std
from lib.utils import getReport
from tpot import TPOTClassifier


class MyExecutor(Executor):
    def feature_select(self, sub_train_set, y):
        logger.info("Start filer Variables total: {}".format(len(sub_train_set.columns)))
        tmp1 = [
            "fpd",
            "td_cnt_offloan_1m",
            "hl_contact_linkman_callin_cnt_pct_5m",
            "hl_contact_credit_callin_len_avg",
            "hl_contact_credit_callin_len_total",
            "CRD_CNT_TOP2AMT_3M",
            "hl_smses_receive_cnt_avg_2m",
            "CRD_CNT_TOP1AMT_3M",
            "CRD_CNT_TOP3CNT_3M",
            "CRD_CNT_INTEGRAL_000_CSM_3M",
            "hl_call_early_morning_pct_5m",
            "hl_smses_receive_cnt_avg_5m",
            "hl_phone_silent_frequentcy",
            "td_cnt_offloan_3m",
            "td_ip_cnt_deviceid_1m",
            "hl_contact_bank_callin_cnt_total",
            "td_cnt_all_1w",
            "CRD_CNT_TOP2CNT_3M",
            "hl_contact_credit_callout_len_avg",
            "CRD_RATIO_TOP2_HIGH_AMT_3M",
            "info_base",
            "info_contact",
            "CRD_CNT_INTEGRAL_00_CSM_3M",
            "hl_smses_send_cnt_5m",
            "CRD_LINE_MAX",
            "hl_contact_bank_callin_cnt_avg",
            "info_addr",
            "CRD_RATIO_TOP2AMT_3M",
            "hl_contact_loan_callout_cnt_avg",
            "hl_contact_loan_callout_len_avg",
            "hl_contact_early_morning_cnt_5m",
            "CRD_AMT_TOP3_HIGH_AMT_3M",
            "hl_region_call_in_cnt_max_uniq_num_cnt",
            "hl_region_call_cnt_max_uniq_num_cnt",
            "hl_contact_contacts_amt_5m",
            "CRD_LINE_SUM",
            "CRD_RATIO_TOP3_HIGH_AMT_3M",
            "hl_call_cnt_mid_5m",
            "hl_callin_cnt_avg_5m",
            "hl_contact_night_cnt_5m",
            "hl_contact_morning_cnt_5m",
            "hl_contact_noon_cnt_5m",
            "hl_region_call_out_cnt_max_uniq_num_cnt",
            "hl_region_unique_num_max_uniq_num_cnt",
            "hl_call_domesitc_cnt_2m",
            "hl_callin_cnt_avg_2m",
            "hl_callout_cnt_avg_5m",
            "hl_region_call_in_time_max_uniq_num_cnt",
            "hl_call_cnt_avg_5m",
            "hl_call_cnt_max_2m",
            "hl_contact_weekend_cnt_5m",
            "hl_region_call_cnt_max_callin_cnt",
            "hl_region_call_out_cnt_max_call_in_cnt",
            "hl_transactions_max_5m",
            "hl_region_call_out_time_max_uniq_num_cnt",
            "hl_call_cnt_avg_2m",
            "hl_call_cnt_mid_2m",
            "hl_contact_bank_callin_len_avg",
            "hl_region_call_in_cnt_max_call_in_cnt",
            "credit",
            "hl_contact_most_frequentcy_pct_5m",
            "hl_contact_workday_cnt_5m",
            "hl_phone_num_used_time_months",
            "hl_region_call_in_cnt_max_call_out_cnt",
            "hl_region_call_in_time_max_call_out_cnt",
            "td_cnt_all_3m_mode_1m",
            # "hl_contact_night_pct",
            "hl_region_unique_num_max_call_out_cnt",
            "hl_callout_cnt_avg_2m",
            "hl_region_call_out_time_max_call_out_cnt",
            "hl_call_cnt_5m",
            "hl_call_cnt_max_5m",
            "hl_call_domesitc_cnt_5m",
            "hl_contact_afternoon_cnt_5m",
            "hl_region_call_in_time_max_call_in_cnt",
            "hl_region_call_out_cnt_max_call_out_cnt",
            "hl_region_call_out_time_max_call_in_cnt",
            "hl_region_unique_num_max_call_in_cnt",
            "hl_smses_cnt_sum_5m",
            "hl_transactions_max_2m",
            "hl_callin_len_avg_2m",
            "hl_callin_len_avg_5m",
            "hl_callout_len_avg_2m",
            "hl_contact_bank_callout_len_avg",
            "hl_region_call_cnt_max_avg_callin_time",
            "hl_region_call_cnt_max_avg_callout_time",
            "hl_region_call_cnt_max_callin_time",
            "hl_region_call_cnt_max_callout_time_pct",
            "hl_region_call_in_cnt_max_avg_call_in_time",
            "hl_region_call_in_cnt_max_avg_call_out_time",
            "hl_region_call_in_cnt_max_call_in_cnt_pct",
            "hl_region_call_in_cnt_max_call_in_time",
            "hl_region_call_in_time_max_avg_call_in_time",
            "hl_region_call_in_time_max_avg_call_out_time",
            "hl_region_call_in_time_max_call_in_cnt_pct",
            "hl_region_call_in_time_max_call_in_time",
            "hl_region_call_out_cnt_max_avg_call_in_time",
            "hl_region_call_out_cnt_max_avg_call_out_time",
            "hl_region_call_out_cnt_max_call_in_time",
            "hl_region_call_out_cnt_max_call_in_time_pct",
            "hl_region_call_out_time_max_avg_call_in_time",
            "hl_region_call_out_time_max_call_out_time",
            "hl_region_unique_num_max_avg_call_in_time",
            "hl_region_unique_num_max_avg_call_out_time",
            "hl_region_unique_num_max_call_in_cnt_pct",
            "hl_region_unique_num_max_call_in_time_pct",
            "hl_transactions_min_2m",
            "hl_transactions_min_5m",
            "hl_transactions_total_amt_2m",
            "hl_transactions_total_amt_5m",
            "CRD_AMT_TOP2CNT_3M",
            "CRD_AMT_TOP1AMT_3M",
            "CRD_AMT_TOP3AMT_3M",
            "operator",
        ]
        return tmp1

    def judge_function_model(self, result, n_var):
        """
        变量筛选方法:
            1. 模型组间权重:  (模型i_AUC_MEAN / 模型i_AUC_STD) / SUM(模型_AUC_MEAN / 模型_AUC_STD)
            2. 单个变量权重: SUM(模型组间权重 * 该变量在该组模型中出现的次数) 
            3. 所有100个变量集的计算结果加和排序
        :param result: 结果集
        :param n_var: 筛选出的变量的个数
        :return: 筛选出来的变量
        """
        result = [i for i in result]
        model_score = []
        for i in result:
            current_auc_list = np.array(i)[:, 0]
            score = judge_auc_mean_std(current_auc_list.mean(), current_auc_list.std())
            model_score.append(score)
        model_score_sum = sum(model_score)
        model_weight = list(map(lambda x: round(x / model_score_sum, 8), model_score))
        var_weight_collection = []
        for mw, single_res in zip(model_weight, result):
            single_var_res = self.get_variable_cnt(single_res[1])
            single_var_res["var_weignt"] = single_var_res["cnt"] * mw
            var_weight_collection.append(single_var_res)
        var_weight_collection = pd.concat(var_weight_collection)

        var_weight_result = []
        for ss in var_weight_collection.groupby(by="variable"):
            tmp = {
                "variable": ss[0],
                "weight_sum": ss[1]["var_weignt"].sum()
            }
            var_weight_result.append(copy(tmp))
        var_weight_result = pd.DataFrame(var_weight_result).sort_values(by="weight_sum", ascending=False)
        return var_weight_result["variable"][0: n_var]


def main():
    df = pd.read_csv("analysis/clean_data_filtered.csv", encoding="utf8")
    trainSet = df[
        df["book_mon"].isin(['2017-05', '2017-06', '2017-07'])
    ].reset_index(drop=True)

    # 这里不一样,训练集用过滤过噪声的,测试集用原来的样本
    df_test = pd.read_csv("data/model_mix/clean_data.csv", encoding="utf8")
    testSet = df_test[
        df_test["book_mon"] == "2017-08"
    ].reset_index(drop=True)
    logger.info("============================Data is ready!============================")

    clf = RandomForestClassifier(
        n_estimators=10,
        max_features=10,
        max_depth=4,
        min_samples_split=0.05,
    )
    myexe = MyExecutor(trainSet, "fpd", clf)
    leftVaris = myexe.get_result(15)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(leftVaris)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    X_train = trainSet[leftVaris].copy()
    y_train = trainSet['fpd'].copy()
    X_test = testSet[leftVaris].copy()
    y_test = testSet['fpd'].copy()
    # AutoSklearn阶段:
    pipeline_optimizer = TPOTClassifier(generations=20,
                                        population_size=10,
                                        cv=3,
                                        random_state=42,
                                        n_jobs=1,
                                        verbosity=2)
    pipeline_optimizer.fit(X_train, y_train)
    pipeline_optimizer.export('tpot_exported_pipeline.py')
    getReport(pipeline_optimizer, trainSet, X_train, y_train, testSet, X_test, y_test)

if __name__ == '__main__':
    main()
