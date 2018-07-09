# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from lib.utils import data_split
from sklearn import metrics
from abc import abstractclassmethod, ABCMeta
from tools.mylogger import logger
from lib.judge_funcs import judge_auc_mean
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
        logger.info(len(left_variables))
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

    # @abstractclassmethod
    # def judge_function_model(self, result):
    #     """
    #     模型级别的筛选,筛选出auc均值和方差表现都比较好的模型
    #     1. 测试AUC, KS表现
    #     2. 单个变量多次入选B1~B10
    #     3. 变量业务逻辑核查
    #     :param: train_all的返回值
    #     :return:
    #     """
    #     # 对变量集对应的10组测试AUC均值和方差进行评判
    #     pass

    def judge_function_model(self, result, n_var):
        """
        变量筛选方法:
            对原有框架的几个环节做出如下改变：
            1. 将原来组与组之间用于比较的评分score,从计算方式上由"AUC平均值/AUC标准差",改为"AUC平均值"
            2. 将最后选取参数的评价指标由"score * 变量出现次数count"，改为"score * sum(Feature_Importance)"
        :param result: 结果集
        :param n_var: 筛选出的变量的个数
        :return: 筛选出来的变量
        """
        result = [i for i in result]
        model_score = []
        for i in result:
            current_auc_list = np.array(i)[:, 0]
            score = judge_auc_mean(current_auc_list.mean())
            model_score.append(score)
        model_score_sum = sum(model_score)
        model_weight = list(map(lambda x: round(x / model_score_sum, 8), model_score))
        var_weight_collection = []
        for mw, single_res in zip(model_weight, result):
            single_var_res = self.get_variable_importance_sum(single_res[1])
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

    def get_variable_importance_sum(self, single_result):
        var_df = pd.DataFrame(single_result[1])
        var_result = []
        for var_name, imp in var_df.groupby(by="variable"):
            tmp = {}
            tmp["variable"] = var_name
            tmp["cnt"] = pd.DataFrame(imp)["importance"].sum()
            var_result.append(copy(tmp))
        return pd.DataFrame(var_result)

    def get_variable_cnt(self, single_result):
        single_result = np.array(single_result)
        # var_df = pd.concat(single_result[:, 1])
        var_df = pd.DataFrame(single_result[1])
        var_result = []
        for var_name, _ in var_df.groupby(by="variable"):
            tmp = {}
            tmp["variable"] = var_name
            tmp["cnt"] = len(_)
            var_result.append(copy(tmp))
        return pd.DataFrame(var_result)

    def get_result(self, n_var):
        """
        随judge_function按情况重写
        :return: 
        """
        var_result = self.judge_function_model(self.train_all(), n_var)
        return var_result
