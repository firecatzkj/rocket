# -*- coding:utf-8 -*-
"""
User Define Feature Select Function
"""
import pandas as pd
from tools.mylogger import logger
from smbinning.smbinning import calc_iv
from sklearn.preprocessing import *
from multiprocessing import Pool


def drop_useless(df, *useless):
    """
    :param df: dataframe
    :param useless: useless columns
    :return: 
    """
    df = df.drop(list(useless), axis=1)
    return df.columns


def drop_by_iv(df, y, p=6):
    """
    select variables by information value
    :param df: 
    :param y:
    :param p: 线程数, < cpu核数/线程数
    :return: 
    """
    columns = list(df.columns)
    result = []
    # print(columns)
    # for col in tqdm(columns):
    #     tmp = {
    #         "columns": col,
    #         "IV": calc_iv(df, y, col)[1]
    #     }
    #     result.append(copy(tmp))
    logger.info("Start filter variables by IV, Current thread: {}".format(p))
    pool = Pool(processes=p)

    for col in columns:
        result.append(pool.apply_async(calc_iv, args=(df, y, col)))
    result = pd.DataFrame([s.get() for s in result])
    selected_vars = list(result[result.IV >= 0.02]["columns"])
    selected_vars.append("fpd")
    return selected_vars


def delMissing(df, thresold, cols=[], include=True):
    if len(cols) == 0:
        cols = df.columns
    total = len(df)
    for col in cols:
        missing = len(df[pd.isnull(df[col])])
        percent = round(missing / total, 2)
        if include is True and percent >= thresold:
            df = df.drop(col, axis=1)
        elif include is False and percent > thresold:
            df = df.drop(col, axis=1)


def handleMissing(df, missingValue, cols=[], strategies=[]):
    for i in range(len(cols)):
        imp = Imputer(missing_values=missingValue, strategy=strategies[i])
        df[cols[i]] = imp.fit_transform(df[cols[i]].values.reshape(-1, 1))


# 连续变量转离散变量
def turnDispersed(df, cols):
    le = LabelEncoder()
    for col in cols:
        df[col] = le.fit_transform(df[col])
    return df.columns


# 离散变量转哑变量
def turnOneHot(df, cols):
    for col in cols:
        fd = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, fd], axis=1)
    df = df.drop(cols, axis=1)
    return df.columns
