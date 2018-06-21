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
        tmp1 = drop_useless(sub_train_set, 'pre_apply_no', 'book_date', 'book_mon')
        return tmp1
        # sub_train_set = sub_train_set[tmp1]
        # tmp2 = drop_by_iv(sub_train_set, "fpd", 2)
        # logger.info("Stop filter Variables total: {}".format(len(tmp2)))
        #return tmp2

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

        print(var_weight_collection.columns)
        var_weight_collection.to_csv("var_w_collection.csv")
        var_weight_result = []
        for ss in var_weight_collection.groupby(by="variable"):
            tmp = {
                "variable": ss[0],
                "weight_sum": ss[1]["var_weignt"].sum()
            }
            var_weight_result.append(copy(tmp))
        var_weight_result = pd.DataFrame(var_weight_result).sort_values(by="weight_sum", ascending=False)
        var_weight_result.to_csv("var_weight_result.csv", index=False)
        return var_weight_result["variable"][0: n_var]


def main():
    df = pd.read_csv("data/hl_test_clean.csv", encoding="utf8")
    df['book_date'] = pd.to_datetime(df['book_date'])
    trainSet = df[(df['book_date'] >= '2017-04-01') & (df['book_date'] <= '2017-07-20')].reset_index(drop=True)
    testSet = df[(df['book_date'] >= '2017-07-20') & (df['book_date'] <= '2017-08-31')].reset_index(drop=True)
    logger.info("============================Data is ready!============================")
    clf = RandomForestClassifier(
        n_estimators=10,
        max_features=10,
        max_depth=4,
        min_samples_split=0.05,
    )
    myexe = MyExecutor(df, "fpd", clf)
    leftVaris = myexe.get_result(20)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(leftVaris)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    X_train = trainSet[leftVaris].copy()
    y_train = trainSet['fpd'].copy()
    X_test = testSet[leftVaris].copy()
    y_test = testSet['fpd'].copy()
    # AutoSklearn阶段:
    pipeline_optimizer = TPOTClassifier(generations=5,
                                        population_size=20,
                                        cv=4,
                                        random_state=42,
                                        n_jobs=1,
                                        verbosity=2)
    pipeline_optimizer.fit(X_train, y_train)
    pipeline_optimizer.export('tpot_exported_pipeline.py')
    getReport(pipeline_optimizer, trainSet, X_train, y_train, testSet, X_test, y_test)

if __name__ == '__main__':
    main()
