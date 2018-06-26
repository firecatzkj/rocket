# -*- coding:utf-8 -*-
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import random
import os
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
        tmp1 = drop_useless(sub_train_set, 'pre_apply_no', 'book_date', 'book_mon')
        return tmp1
        # sub_train_set = sub_train_set[tmp1]
        # tmp2 = drop_by_iv(sub_train_set, "fpd", 2)
        # logger.info("Stop filter Variables total: {}".format(len(tmp2)))
        # return tmp2

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


def main(kwargs):
    df = pd.read_csv("data/model_mix/clean_data.csv", encoding="utf8")
    trainSet = df[
        df["book_mon"].isin(['2017-05', '2017-06', '2017-07'])
    ].reset_index(drop=True)
    testSet = df[
        df["book_mon"] == "2017-08"
    ].reset_index(drop=True)

    logger.info("============================Data is ready!============================")
    # clf = RandomForestClassifier(
    #     n_estimators=10,
    #     max_features=10,
    #     max_depth=4,
    #     min_samples_split=0.05,
    # )
    clf = kwargs["feature_model"]
    myexe = MyExecutor(df, "fpd", clf)
    leftVaris = myexe.get_result(kwargs["feature_num"])
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(leftVaris)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    X_train = trainSet[leftVaris].copy()
    y_train = trainSet['fpd'].copy()
    X_test = testSet[leftVaris].copy()
    y_test = testSet['fpd'].copy()
    # AutoSklearn阶段:
    pipeline_optimizer = TPOTClassifier(
        generations=int(kwargs["generations"]),
        population_size=int(kwargs["population_size"]),
        #offspring_size=kwargs["offspring_size"],
        #mutation_rate=kwargs["mutation_rate"],
        #crossover_rate=kwargs["crossover_rate"],
        scoring=kwargs["scoring"],
        cv=int(kwargs["cv"]),
        subsample=float(kwargs["subsample"]),
        n_jobs=int(kwargs["n_jobs"]),
        #max_time_mins=kwargs["max_time_mins"],  # max_eval_time_seconds = max(int(self.max_eval_time_mins * 60), 1)
        max_eval_time_mins=int(kwargs["max_eval_time_mins"]),
        random_state=random.randint(1, 100)
    )
    pipeline_optimizer.fit(X_train, y_train)
    trainKS, testKS, abs_trainKS_testKS, trainAUC, testAUC, abs_trainAUC_testAUC = \
        getReport(pipeline_optimizer, trainSet, X_train, y_train, testSet, X_test, y_test)

    # 记录结果
    if kwargs["uid"] is os.listdir("tpot_result/"):
        os.removedirs("tpot_result/{}".format(kwargs['uid']))
    os.mkdir("tpot_result/{}".format(kwargs["uid"]))
    pipeline_optimizer.export('tpot_result/{}/tpot_exported_pipeline.py'.format(kwargs["uid"]))
    with open('tpot_result/{}/vars'.format(kwargs["uid"]), "w+") as f1:
        f1.write(str(leftVaris))
    report = pd.DataFrame([{
        "trainKS": trainKS,
        "testKS": testKS,
        "abs_trainKS_testKS": abs_trainKS_testKS,
        "trainAUC": trainAUC,
        "testAUC": testAUC,
        "abs_trainAUC_testAUC": abs_trainAUC_testAUC
    }, ])
    report.to_csv('tpot_result/{}/report.csv'.format(kwargs['uid']), index=False, encoding="utf8")


def batch_run():
    prama_df = pd.read_csv("batch_test_tpot.csv", encoding="utf8")
    feature_model = RandomForestClassifier(
        n_estimators=10,
        max_features=5,
        max_depth=3,
        min_samples_split=0.05,
    )
    for i in range(len(prama_df)):
        tmp = dict(prama_df.iloc[i])
        for j in tmp.keys():
            if pd.isna(tmp[j]):
                tmp[j] = None
        tmp["feature_model"] = feature_model
        print(tmp)
        main(tmp)


if __name__ == '__main__':
    batch_run()
