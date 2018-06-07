# -*- coding:utf-8 -*-
"""
模型评判方法
"""
import pandas as pd
from tools.mylogger import logger


def judge_function(result):
    """
    1. 测试AUC, KS表现
    2. 单个变量多次入选B1~B10
    3. 变量业务逻辑核查
    :param: train_all的返回值
    :return: 
    """
    # TODO: 对变量集对应的10组测试AUC均值和方差进行评判
    calc_res = []
    for model_res in result:
        current_auc = model_res[0]
        logger.info(current_auc)
        var_importance = model_res[1]
        var_importance["importance_plus"] = var_importance["importance"] * current_auc
        calc_res.append(var_importance)
    return pd.concat(calc_res)