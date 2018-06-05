# -*- coding:utf-8 -*-
import os
import pandas as pd
import math
import numpy as np
from copy import copy
from sklearn import tree


def smbinning_chaid(df, Y, x):
    y = df[Y]
    x_test = df[[x, ]]


def smbinning_test(df, Y, x):
    y = df[Y]
    x_test = df[[x, ]]
    mytree = tree.DecisionTreeClassifier(
        max_features=1,
        #min_weight_fraction_leaf=0.05,
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
    #print("total is {}".format(len(tmp)))
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


if __name__ == '__main__':
    os.chdir("E:\\mycode\\rocket\\smbinning")
    df = pd.read_csv("../data/hl_test_clean.csv")

    # res = calc_iv(df, "fpd", "hl_region_call_out_time_max_avg_call_in_time", [0.8074, 0.8759, 0.9665])
    # print(res)
    # cols = open("col", "r+").readlines()
    # for c in cols:
    #     tmp_col = str(c).strip()
    #     cuts = smbinning_test(df, "fpd", tmp_col)
    #     if len(cuts) == 0:
    #         print(tmp_col, "--", 0)
    #         continue
    #     calc_iv(df, "fpd", tmp_col, cuts)
    print(smbinning_test(df, "fpd", "hl_contact_morning_cnt_5m"))

