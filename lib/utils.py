# -*- coding:utf-8 -*-
import pandas as pd
import math
from sklearn.utils import shuffle
from tools.mylogger import logger
from copy import copy
from sklearn import tree
from scipy import stats
from sklearn.metrics import *


# 随机打乱数据
def data_split(data, threads, split_flag):
    data = shuffle(pd.DataFrame(data))
    total = len(data)
    sep = round(total / threads, 2)
    split_point = []
    for i in range(threads + 1):
        split_point.append(int(i * sep))
    logger.info("CutPoint: " + str(split_point))
    split_res = []
    for j in range(len(split_point) - 1):
        t = []
        t.append(split_point[j])
        t.append(split_point[j + 1])
        split_res.append(t)
    res = []
    for s in split_res:
        content = data[s[0]:s[1]]
        if len(content) != 0:
            res.append(content)
    res_final = []
    for ss in range(len(res)):
        tmp = res[ss]
        tmp = pd.DataFrame(tmp)
        tmp[split_flag] = ss
        res_final.append(tmp)
    return res_final


# 自动分箱
def smbinning_test(df, Y, x):
    y = df[Y]
    x_test = df[[x, ]]
    mytree = tree.DecisionTreeClassifier(
        max_features=1,
        # min_weight_fraction_leaf=0.05,
        min_samples_split=0.05,
        criterion="entropy",
        max_leaf_nodes=5)
    mytree.fit(x_test, y)
    cutpoint = mytree.tree_.threshold
    print(x)
    print(cutpoint)
    print("=====================")
    return cutpoint[cutpoint != -2]


def calc_columns(single_result, tmp, Y, df):
    single_result["total"] = len(tmp)
    single_result["good"] = len(tmp[tmp[Y] == 0])
    single_result["bad"] = len(tmp[tmp[Y] == 1])
    single_result["goodDistr"] = round((single_result["good"] / len(df[df[Y] == 0])), 6)
    single_result["badDistr"] = round((single_result["bad"] / len(df[df[Y] == 1])), 6)
    single_result["distr"] = round((len(tmp) / len(df)), 6)
    try:
        single_result["badRate"] = round((single_result["bad"] / single_result["total"]), 6)
    except:
        single_result["badRate"] = None
    try:
        single_result["Odds"] = round((single_result["bad"] / single_result["good"]), 6)
    except:
        single_result["Odds"] = None
    try:
        single_result["WOE"] = round(math.log(single_result["goodDistr"] / single_result["badDistr"]), 6)
    except:
        single_result["WOE"] = None
    try:
        single_result["IV"] = round((single_result["goodDistr"] - single_result["badDistr"]) * single_result["WOE"], 4)
    except:
        single_result["IV"] = None
    return copy(single_result)


def calc_iv(df, Y, x, cuts):
    # 排序很重要,因为后面算区间的时候需要索引+1 -1
    cuts = list(cuts)
    cuts.sort()
    single_sample = df[[Y, x]]
    single_sample = pd.DataFrame(single_sample)
    result = []
    base_single_result = dict.fromkeys([
        "cutpoints",
        "total",
        "good",
        "bad",
        "goodDistr",
        "badDistr",
        "distr",
        "badRate",
        "Odds",
        "WOE",
        "IV"
    ], None)

    for i in cuts:
        single_result = copy(base_single_result)
        if i == min(cuts):
            tmp = single_sample.dropna()
            single_result["cutpoints"] = "<={}".format(i)
            tmp = tmp[tmp[x] <= i]
        else:
            tmp = single_sample.dropna()
            single_result["cutpoints"] = "<={}".format(i)
            tmp = tmp[
                (tmp[x] > cuts[cuts.index(i) - 1]) &
                (tmp[x] <= i)]
        result.append(calc_columns(single_result, tmp, Y, df))

    tmp_max = single_sample.dropna()
    tmp_max = tmp_max[tmp_max[x] > max(cuts)]
    single_result_max = calc_columns(copy(base_single_result), tmp_max, Y, df)
    single_result_max["cutpoints"] = ">{}".format(max(cuts))
    result.append(single_result_max)

    # 计算missing和total
    # missing
    tmp_missing = single_sample[pd.isnull(single_sample[x])]
    single_result_missing = calc_columns(copy(base_single_result), tmp_missing, Y, df)
    single_result_missing["cutpoints"] = "Missing"
    result.append(single_result_missing)
    # total
    result = pd.DataFrame(result)
    tmp_total = single_sample
    single_result_total = calc_columns(copy(base_single_result), tmp_total, Y, df)
    single_result_total["cutpoints"] = "Total"
    single_result_total["IV"] = round(result["IV"].sum(), 6)
    print(x, "--", single_result_total["IV"])
    result = result.append(pd.DataFrame([single_result_total, ]))[[
        "cutpoints", "total", "good", "bad", "goodDistr", "badDistr", "distr", "badRate", "Odds", "WOE", "IV"]]
    return result


def getReport(model, trainSet, X_train_ask, y_train_ask, testSet, X_test_ask, y_test_ask):
    print('训练集ks')
    X_pred = model.predict_proba(X_train_ask)
    trainSet['prob'] = X_pred[:, 1]
    good = trainSet[trainSet['fpd'] == 0]
    bad = trainSet[trainSet['fpd'] == 1]
    trainKS = stats.ks_2samp(good['prob'], bad['prob']).statistic
    print(trainKS)
    print('测试集ks')
    y_pred = model.predict_proba(X_test_ask)
    testSet['prob'] = y_pred[:, 1]
    good = testSet[testSet['fpd'] == 0]
    bad = testSet[testSet['fpd'] == 1]
    testKS = stats.ks_2samp(good['prob'], bad['prob']).statistic
    print(testKS)
    print('KS之差为:' + str(trainKS - testKS))
    print('==================================================')

    print('训练集auc')
    trainAUC = roc_auc_score(y_train_ask, trainSet['prob'])
    print(trainAUC)
    print('测试集auc')
    testAUC = roc_auc_score(y_test_ask, testSet['prob'])
    print(testAUC)
    print('AUC之差为:' + str(trainAUC - testAUC))
    print('==================================================')
    print('')
    return trainKS, testKS, (trainKS - testKS), trainAUC, testAUC, (trainAUC - testAUC)


if __name__ == '__main__':
    df = pd.read_csv("..\\data\\hl_test_clean.csv")
    print(len(df), ">>>>>>>>>>>>>>>>>>>>>>")
    res = data_split(df, 10, "flag")
    for i in res:
        print(len(i))
