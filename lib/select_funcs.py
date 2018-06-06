# -*- coding:utf-8 -*-
"""
User Define Feature Select Function
"""
import pandas as pd


def drop_useless(df, *useless):
    """
    :param df: dataframe
    :param useless: useless columns
    :return: 
    """
    df = df.drop(list(useless), axis=1)
    return df.columns
