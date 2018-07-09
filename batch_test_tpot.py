# -*- coding:utf-8 -*-
import pandas as pd
import random
import os
from sklearn.ensemble import RandomForestClassifier
from lib.execute import Executor
from tools.mylogger import logger
from lib.select_funcs import drop_useless, drop_by_iv
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
