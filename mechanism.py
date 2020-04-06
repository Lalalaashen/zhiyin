# -*- coding: utf-8 -*-
# %% 配置
import pandas as pd
import numpy as np


def read_and_chop(file_name, sheet_name, **kwargs) -> pd.DataFrame:
    """读取并截取有效数据"""
    df = pd.read_excel(file_name, sheet_name=sheet_name, **kwargs)
    if df.iloc[-1, 0] == '数据来源：Wind':
        df = df.iloc[:-2]
    return df


df = read_and_chop('影响机制研究.xlsx', 'Sheet1')
try:
    df['报告期'] = df['报告期'].apply(lambda x: x.strftime('%Y-%m-%d'))
except:
    pass
if '内部控制缺陷类型' in df.columns:
    df = df.drop(columns='内部控制缺陷类型')
df['股票代码'] = df['股票代码'].apply(lambda x: f'{x:0>6.0f}')

# %% 一般缺陷
# deficiency_df = read_and_chop('IC_DeficiencyInfo.xlsx', 'Sheet1')
#
# # 数据格式
# deficiency_df['报告期'] = deficiency_df["统计截止日期"]
# deficiency_df["股票代码"] = deficiency_df["股票代码"].apply(lambda x: f'{x:0>6.0f}')
# # 合并
# def_merge_df = df[['股票代码', '报告期']].reset_index().merge(
#     deficiency_df[['股票代码', '报告期', '缺陷类型编码', '缺陷类型']],
#     left_on=['股票代码', '报告期'],
#     right_on=['股票代码', '报告期'],
# ).set_index('index')
# # 写入
# def_merge_df.loc[def_merge_df['缺陷类型'] == '一般缺陷', '缺陷类型编码'] = 2
# def_merge_df.loc[def_merge_df['缺陷类型'] == '重要缺陷', '缺陷类型编码'] = 3
# df['缺陷类型编码'] = 1
# df.loc[def_merge_df.index, '缺陷类型编码'] = def_merge_df['缺陷类型编码'].values
# df.loc[def_merge_df.index, '缺陷类型'] = def_merge_df['缺陷类型'].values

# %% 关联交易金额
# opt_df = read_and_chop('RPT_Operation.xlsx', 'Sheet1')
# opt_df['报告期'] = opt_df["统计截止日期"]
# opt_df["股票代码"] = opt_df["证券代码"].apply(lambda x: f'{x:0>6.0f}')
#
# opt_group = opt_df.groupby(['股票代码', '报告期'])
# money_sum = opt_group['关联交易涉及的金额'].sum().reset_index()
# # 合并
# opt_merge_df = df[['股票代码', '报告期']].reset_index().merge(
#     money_sum[['股票代码', '报告期', '关联交易涉及的金额']],
#     left_on=['股票代码', '报告期'],
#     right_on=['股票代码', '报告期'],
# ).set_index('index')
# df['关联交易涉及的金额'] = 0
# df.loc[opt_merge_df.index, '关联交易涉及的金额'] = opt_merge_df['关联交易涉及的金额'].values

# %% 高管
director_df = read_and_chop('CG_Director.xlsx', 'Sheet1')
director_df['报告期'] = director_df["统计截止日期"]
director_df["股票代码"] = director_df["证券代码"].apply(lambda x: f'{x:0>6.0f}')

director_group = director_df.groupby(['股票代码', '报告期'])
director_sum = director_group['报告期报酬总额'].apply(lambda x: np.mean(x[:3])).reset_index()
# 合并
director_merge_df = df[['股票代码', '报告期']].reset_index().merge(
    director_sum[['股票代码', '报告期', '报告期报酬总额']],
    left_on=['股票代码', '报告期'],
    right_on=['股票代码', '报告期'],
).set_index('index')
df['报告期报酬总额前3均值'] = 0
df.loc[director_merge_df.index, '报告期报酬总额前3均值'] = director_merge_df['报告期报酬总额'].values

# %% 审计
audit_df = read_and_chop('FIN_Audit.xlsx', 'Sheet1')
audit_df['报告期'] = audit_df["会计截止日期"]
audit_df["股票代码"] = audit_df["证券代码"].apply(lambda x: f'{x:0>6.0f}')

audit_group = audit_df.groupby(['股票代码', '报告期'])
audit_sum = audit_group['审计意见类型'].apply(lambda x: x.values[0]).reset_index()
# 合并
audit_merge_df = df[['股票代码', '报告期']].reset_index().merge(
    audit_sum[['股票代码', '报告期', '审计意见类型']],
    left_on=['股票代码', '报告期'],
    right_on=['股票代码', '报告期'],
).set_index('index')
df['审计意见类型'] = None
df.loc[audit_merge_df.index, '审计意见类型'] = audit_merge_df['审计意见类型'].values

# %% 最后存储
# 存储
df.to_excel('影响机制研究.xlsx', index=False)
