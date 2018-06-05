# -*- coding:utf-8 -*-

def data_split(data, threads):
    total = len(data)
    sep = round(total / threads, 2)
    split_point = []
    for i in range(threads + 1):
        split_point.append(int(i * sep))
    print(split_point)
    split_res = []
    for j in range(len(split_point) - 1):
        t = []
        t.append(split_point[j])
        t.append(split_point[j + 1])
        split_res.append(t)
    res = []
    for s in split_res:
        content = data[s[0]:s[1]]
        if len(content) != 0:
            res.append(content)
    return res

