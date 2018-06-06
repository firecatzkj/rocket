# -*- coding:utf-8 -*-
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from lib.utils import data_split
from sklearn import metrics
from abc import abstractclassmethod, ABCMeta

class Executor(metaclass=ABCMeta):
    def __init__(self, df, y, model):
        """
        :param df: 数据集
        :param y: target
        :param model: sklearn模型对象
        :param filter_funcs: 变量初筛的方法 ==> list
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
        # clf = DecisionTreeRegressor()
        clf.fit(train_x, train_y)
        predict_prob_y = clf.predict_proba(test_x)
        predict_prob_y = pd.DataFrame(predict_prob_y, columns=["fpd0", "fpd1"])
        test_auc = metrics.roc_auc_score(test_y, predict_prob_y["fpd0"])
        # print(test_auc, ">>>>>>>>>>>>>>>>>>>>>>>>>>>")
        return test_auc

    def train_by_feature(self, feature):
        current_data = []
        for single in self.df_splited:
            current_data.append(single[feature])
        # current_data = list(map(lambda x: x[feature], self.df_splited))
        result_auc = []
        for i in range(len(current_data)):
            tmp = current_data.copy()
            test_data = pd.DataFrame(tmp.pop(i))
            train_data = pd.concat(tmp)
            train_x = train_data.drop(self.y, axis=1)
            train_y = train_data[self.y]
            test_x = test_data.drop(self.y, axis=1)
            test_y = test_data[self.y]
            current_auc = self.build_model(self.model, train_x, train_y, test_x, test_y)
            result_auc.append(current_auc)
        return result_auc

    def train_all(self):
        feature_list = []
        for i in range(len(self.df_splited)):
            tmp = self.df_splited.copy()
            tmp.pop(i)
            fea = self.feature_select(pd.concat(tmp), "fpd")
            feature_list.append(fea)
        for feature in feature_list:
            res = self.train_by_feature(feature)
            print(pd.Series(res).mean(), "\t", pd.Series(res).std())

    def judge_function(self, auc_mean_weight, auc_std_weight):
        pass
        # TODO: 对变量集对应的10组测试AUC均值和方差进行评判
        # 1. 测试AUC, KS表现
        # 2. 单个变量多次入选B1~B10
        # 3. 变量业务逻辑核查

