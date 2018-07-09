# -*- coding:utf-8 -*-
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def clean_data():
    mydata = pd.read_csv("app_data.csv", encoding="utf8")
    # 分类变量处理
    cols_encode = [
        "hl_region_call_cnt_max_loc",
        "hl_region_call_in_cnt_max_loc",
        "hl_region_call_in_time_max_loc",
        "hl_region_call_out_cnt_max_loc",
        "hl_region_call_out_time_max_loc",
        "hl_region_unique_num_max_loc",
        "hl_person_age2",
        "CRD_FLAG_IMPORT_BILL_MTHD"
    ]
    le = LabelEncoder()
    for ce in cols_encode:
        mydata[ce] = le.fit_transform(mydata[ce])

    # 删除无用变量
    del_cols = [
        "product_name",
        "hl_region_call_cnt_max_loc_code2",
        "hl_region_call_in_cnt_max_loc_code2",
        "hl_region_call_in_time_max_loc_code2",
        "hl_region_call_out_cnt_max_loc_code2",
        "hl_region_call_out_time_max_loc_code2",
        "hl_region_unique_num_max_loc_code2",
        "hl_region_call_cnt_max_loc_code3",
        "hl_region_call_in_cnt_max_loc_code3",
        "hl_region_call_in_time_max_loc_code3",
        "hl_region_call_out_cnt_max_loc_code3",
        "hl_region_call_out_time_max_loc_code3",
        "hl_region_unique_num_max_loc_code3",
        "gmt_create",
        "reg_time",
        "time1",
        "id",
        "id_id"
    ]
    for dc in del_cols:
        del mydata[dc]

    # 缺失值,特殊值处理
    for col in mydata.columns:
        if col in ["pre_apply_no", "book_mon", "book_date", "product_name"]:
            continue
        mydata[col].replace(-99999, None, inplace=True)
        mydata[col].replace(-2, -2.000001, inplace=True)
        # try:
        #     mydata[col] = mydata[col].fillna(mydata[col].mean())
        # except:
        #     print(col)

    # 存为csv格式
    mydata.to_csv("clean_data_new.csv", index=False, encoding="utf8")


def get_data():
    return pd.read_csv("clean_data.csv", encoding="utf8")

if __name__ == '__main__':
    clean_data()

