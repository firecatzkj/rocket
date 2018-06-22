# -*- coding:utf-8 -*-
from sklearn.preprocessing import *


a = ["A", "A", "B", "C"]
le = LabelEncoder()
print(le.fit_transform(a))