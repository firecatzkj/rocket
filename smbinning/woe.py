# -*- coding:utf-8 -*-
import numpy as np
import os
import pandas as pd




class WOE:
    def __init__(self):
        self.replace = True

    def woe(self, X, y, to_discretize=[], replace=True):
        '''
        Compute the WoE as well as the information values for each features in X.
        X: Features
        y: Target
        to_discretize: array containing the indexes of the features to discretize
        '''
        X = np.array(X)
        n_features = X.shape[1]
        IV = np.zeros((n_features, ))
        self.replace = replace

        for i in range(0, n_features):
            if np.in1d(i, to_discretize):
                x, iv = self.compute_woe(self.discretize(X[:, i]), y)
            else:
                x, iv = self.compute_woe(X[:, i], y)

            if self.replace:
                X[:, i] = x

            IV[i] = iv

        if self.replace:
            return X, IV
        else:
            return IV

    def compute_woe(self, x, y):
        n_0 = len(np.where(y == 0)[0])
        n_1 = len(np.where(y == 1)[0])

        attributes = np.unique(x)

        IV = 0

        woe = np.zeros((len(attributes), ))

        for i, attribute in enumerate(attributes):
            attr_event_index = np.where(x == attribute)[0]
            event_index = np.where(y == 1)[0]
            non_event_index = np.where(y == 0)[0]

            a = len(np.intersect1d(attr_event_index, event_index))
            b = len(np.intersect1d(attr_event_index, non_event_index))

            woe[i] = np.log(((a+0.5)/n_0)/((b+0.5)/n_1))

            IV = IV + (a/n_0 - b/n_1)*woe[i]

        if self.replace:
            x = self.replace_x(x, woe, attributes)

        return x, IV

    def discretize(self, x):
        # Discretize into 10 parts (using percentiles)
        percentiles = np.arange(0, 110, 10)

        return np.float32(np.digitize(x, percentiles))

    def replace_x(self, x, woe, attributes):
        for i, attribute in enumerate(attributes):
            x[np.where(x == attribute)[0]] = woe[i]

        return x

if __name__ == '__main__':
    os.chdir("E:\\mycode\\rocket\\smbinning")
    df = pd.read_csv("../data/hl_test_clean.csv")
    mywoe = WOE()
    y = df["fpd"]
    x_test = df[["hl_region_call_out_time_max_avg_call_in_time", ]]
    res = mywoe.woe(x_test, y, to_discretize=[157, 301])
    print(res[-1])