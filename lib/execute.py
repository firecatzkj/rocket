# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from lib.utils import data_split
from sklearn import metrics
from abc import abstractclassmethod, ABCMeta
from tools.mylogger import logger
from copy import copy
from sklearn.tree import DecisionTreeClassifier


class Executor(metaclass=ABCMeta):
    def __init__(self, df, y, model):
        """
        :param df: 数据集
        :param y: target
        :param model: sklearn模型对象
        """
        self.y = y
        self.df_splited = data_split(df, threads=10, split_flag="split_flag")
        self.model = model

    @abstractclassmethod
    def feature_select(self, sub_train_set, y):
        """
        :param sub_train_set: 数据集
        :param y: target
        :return: 筛选之后的变量名列表
        """
        pass
        # sub_train_set = pd.DataFrame(sub_train_set).drop(useless, axis=1)
        # TODO: 返回筛选之后的变量列表
        # return columns
        # for i in self.filter_funcs:
        #     sub_train_set = i(sub_train_set, y)
        # return sub_train_set.columns

    def build_model(self, mymodel, train_x, train_y, test_x, test_y):
        """
        :param mymodel: sklearn对象
        :param train_x: 训练集_X
        :param train_y: 训练集_Y
        :param test_x:  测试集_X
        :param test_y:  测试集_Y
        :return: auc
        """
        # TODO: 训练模型,使用九份数据训练,在第十份数据上检验,得到相应的AUC,KS,  总共训练10次
        clf = mymodel
        clf.fit(train_x, train_y)
        predict_prob_y = clf.predict_proba(test_x)
        predict_prob_y = pd.DataFrame(predict_prob_y, columns=["fpd0", "fpd1"])
        test_auc = metrics.roc_auc_score(test_y, predict_prob_y["fpd1"])
        left_variables = train_x.columns[np.where(clf.feature_importances_ > 0)].tolist()
        logger.info(left_variables)
        this_feature_impoirtance = pd.DataFrame(list(zip(train_x.columns, clf.feature_importances_)),
                                                columns=["variable", "importance"])
        used_feature_importance = this_feature_impoirtance[this_feature_impoirtance.importance > 0]
        # print(used_feature_importance)
        return test_auc, used_feature_importance

    def train_by_feature(self, feature):
        current_data = []
        for single in self.df_splited:
            current_data.append(single[feature])
        # current_data = list(map(lambda x: x[feature], self.df_splited))
        result = []
        for i in range(len(current_data)):
            tmp = current_data.copy()
            test_data = pd.DataFrame(tmp.pop(i))
            train_data = pd.concat(tmp)
            train_x = train_data.drop(self.y, axis=1)
            train_y = train_data[self.y]
            test_x = test_data.drop(self.y, axis=1)
            test_y = test_data[self.y]
            current_res = self.build_model(self.model, train_x, train_y, test_x, test_y)
            result.append(current_res)
        return result

    def train_all(self):
        feature_list = []
        for i in range(len(self.df_splited)):
            tmp = self.df_splited.copy()
            tmp.pop(i)
            fea = self.feature_select(pd.concat(tmp), "fpd")
            feature_list.append(fea)
        for feature in feature_list:
            res = self.train_by_feature(feature)
            yield res

    @abstractclassmethod
    def judge_function(self, result):
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

    def get_result(self):
        final_result = []
        for i in self.train_all():
            final_result.append(self.judge_function(i))
        final_result = pd.concat(final_result)
        final_result = pd.DataFrame(final_result)
        res = {}
        for var, df in final_result.groupby(by="variable"):
            auc_p_imp = df["importance_plus"].sum()
            res[var] = auc_p_imp
        return pd.Series(res).sort_values(ascending=False)






