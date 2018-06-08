# -*- coding:utf-8 -*-
"""
User Define Feature Select Function
"""

import pandas as pd
from tqdm import tqdm
from copy import copy
from smbinning.smbinning import calc_iv


def drop_useless(df, *useless):
    """
    :param df: dataframe
    :param useless: useless columns
    :return: 
    """
    df = df.drop(list(useless), axis=1)
    return df.columns


def drop_by_iv(df, y):
    """
    select variables by information value
    :param df: 
    :param y:
    :return: 
    """
    columns = list(df.columns)
    result = []
    # print(columns)
    for col in tqdm(columns):
        tmp = {
            "columns": col,
            "IV": calc_iv(df, y, col)[1]
        }
        result.append(copy(tmp))
    result = pd.DataFrame(result)
    selected_vars = list(result[result.IV >= 0.02]["columns"])
    selected_vars.append("fpd")
    return selected_vars
