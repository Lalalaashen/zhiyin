# -*- coding: utf-8 -*-
import os
import pandas as pd

file_name = '私募机构'
file_ls = os.listdir(file_name)
# 读取文件夹
stat_owned_ls = [os.path.join(file_name, x) for x in file_ls if '-国有-' in x]
no_stat_owned_ls = [os.path.join(file_name, x) for x in file_ls if '-非国有-' in x]
# 合并
stat_owned_df = pd.concat([pd.read_excel(x, header=3) for x in stat_owned_ls], ignore_index=True)
no_stat_owned_df = pd.concat([pd.read_excel(x, header=3) for x in no_stat_owned_ls], ignore_index=True)
# 输出
stat_owned_df.to_excel('国有.xls', engine='openpyxl')
no_stat_owned_df.to_excel('非国有.xls', engine='openpyxl')
