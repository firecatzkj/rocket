library(smbinning)
library(car)
library(rpart)
library(readxl)
library(compiler)
library(data.table)
library(usdm)
library(ggplot2)
library(ctree)

# load data
setwd("E:\\mycode\\rocket\\data")
mydata = read.csv("hl_test_clean.csv")

# 计算sumiv
sumiv = hb.smbinning.sumiv(df = mydata, y = "fpd")
sumiv = read.csv("sumiv.csv")
write.csv(sumiv, file = "sumiv.csv")
print(sumiv)

# smbinning
hb.smbinning(mydata,"fpd","hl_region_call_in_cnt_max_call_out_cnt")


hb.smbinning(mydata,"fpd","hl_phone_silent_frequentcy")

hb.smbinning(mydata,"fpd","hl_phone_num_used_time_months")

hb.smbinning(mydata,"fpd","hl_contact_holiday_cnt_5m")

ctree

hb.smbinning(mydata,"fpd", "hl_contact_holiday_cnt_5m", cuts=c(119.5, 212.5, 243.5,553.5))


