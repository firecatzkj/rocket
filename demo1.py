# -*- coding:utf-8 -*-
import os

import pandas as pd
from sklearn.utils import shuffle
os.chdir("E:\\mycode\\rocket\\test")
df = pd.read_csv("../data/hl_test_clean.csv")
df = pd.DataFrame(shuffle(df))
X = df.drop(["pre_apply_no", "book_date", "book_mon", "fpd"], axis=1)
y = df["fpd"]


res = mywoe.fit(X, y)


# mywoe = information_value.WOE()
#
# for col in X.columns:
#     print(col)
#     tmp = np.array(X[col])
#     y = np.array(y)
#     res = mywoe.woe_single_x(tmp, y)
#     print(res[1])
#     print("======================")
